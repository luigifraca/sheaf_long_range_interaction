"""Extraction of raw restriction maps and normalized sheaf transports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from sheaf_mpnn.nsd.nsd_layers import (
    DiagonalNSDConv,
    GeneralNSDConv,
    OrthogonalNSDConv,
)
from sheaf_mpnn.utils import attention_cayley, cayley, householder
from torch import nn
from torch_geometric.utils import add_self_loops

from slri.analysis.adapters import ModelTraceAdapter
from slri.models import (
    FrozenOrthogonalNSDConv,
    IdentityNSDConv,
)


@dataclass
class GeometrySnapshot:
    """Tabular geometry plus matrices needed for downstream path analysis."""

    table: pd.DataFrame
    normalized_transports: dict[tuple[int, int, int], torch.Tensor]
    strengths_by_layer: dict[int, dict[tuple[int, int], float]]


def matrix_diagnostics(
    matrix: torch.Tensor, epsilon: float = 1e-12
) -> dict[str, Any]:
    """Return norms and singular-spectrum diagnostics for one matrix."""
    singular = torch.linalg.svdvals(matrix.float())
    positive = singular[singular > epsilon]
    sigma_max = float(singular.max().item()) if singular.numel() else 0.0
    sigma_min = float(positive.min().item()) if positive.numel() else 0.0
    squared = singular.square()
    if squared.sum() > 0:
        probabilities = squared / squared.sum()
        effective_rank = float(
            torch.exp(
                -(probabilities * probabilities.clamp_min(epsilon).log()).sum()
            ).item()
        )
    else:
        effective_rank = 0.0
    return {
        "fro": float(torch.linalg.matrix_norm(matrix).item()),
        "spectral": sigma_max,
        "sigma_max": sigma_max,
        "sigma_min_nonzero": sigma_min,
        "condition_number": sigma_max / max(sigma_min, epsilon),
        "numerical_rank": int(positive.numel()),
        "effective_rank": effective_rank,
        "singular_values": singular.detach().cpu().tolist(),
    }


def _raw_maps(
    layer: nn.Module,
    hidden: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(layer, IdentityNSDConv):
        maps = layer.restriction_maps(
            edge_index.size(1), device=hidden.device, dtype=hidden.dtype
        )
        return maps, maps
    if isinstance(layer, FrozenOrthogonalNSDConv):
        layer.set_local_node_ids(
            torch.arange(hidden.size(0), device=hidden.device)
        )
        try:
            return layer.restriction_maps(edge_index)
        finally:
            layer.set_local_node_ids(None)
    raw_dst, raw_src = layer._bidirectional_input(hidden, edge_index)
    if isinstance(layer, DiagonalNSDConv):
        return torch.diag_embed(raw_dst), torch.diag_embed(raw_src)
    if isinstance(layer, GeneralNSDConv):
        dst = raw_dst.view(-1, layer.stalk_dim, layer.stalk_dim)
        src = raw_src.view(-1, layer.stalk_dim, layer.stalk_dim)
        if layer.use_attention:
            eye = torch.eye(
                layer.stalk_dim, device=hidden.device, dtype=hidden.dtype
            ).unsqueeze(0)
            dst = eye - torch.softmax(dst, dim=-1)
            src = eye - torch.softmax(src, dim=-1)
        return dst, src
    if isinstance(layer, OrthogonalNSDConv):
        if layer.use_attention:
            return (
                attention_cayley(
                    raw_dst, layer.stalk_dim, hidden.device, hidden.dtype
                ),
                attention_cayley(
                    raw_src, layer.stalk_dim, hidden.device, hidden.dtype
                ),
            )
        if layer.orth_strategy == "fasth":
            return (
                householder(raw_dst, layer.stalk_dim),
                householder(raw_src, layer.stalk_dim),
            )
        return (
            cayley(raw_dst, layer.stalk_dim, layer.clamp_val),
            cayley(raw_src, layer.stalk_dim, layer.clamp_val),
        )
    raise TypeError(f"Unsupported sheaf layer: {type(layer).__name__}")


def extract_sheaf_geometry(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    max_edges_per_layer: int | None = None,
) -> GeometrySnapshot | None:
    """Extract restriction, transport, and normalized transport diagnostics."""
    if not hasattr(model, "stalk_dim") or not hasattr(model, "layers"):
        return None
    model = model.to(dtype=torch.float32).eval()
    device = next(model.parameters()).device
    x = x.to(device=device, dtype=torch.float32)
    edge_index = edge_index.to(device)
    adapter = ModelTraceAdapter(model)
    states, _ = adapter.trace(x, edge_index)
    rows: list[dict[str, Any]] = []
    normalized: dict[tuple[int, int, int], torch.Tensor] = {}
    strengths: dict[int, dict[tuple[int, int], list[float]]] = {}
    for layer_index, layer in enumerate(model.layers):
        layer_edges = edge_index
        if layer.add_self_loops:
            layer_edges, _ = add_self_loops(
                layer_edges, num_nodes=x.size(0)
            )
        hidden = states[layer_index]
        dst_map, src_map = _raw_maps(layer, hidden, layer_edges)
        self_map_matrix = dst_map.transpose(-2, -1) @ dst_map
        cross_matrix = dst_map.transpose(-2, -1) @ src_map
        if isinstance(layer, DiagonalNSDConv):
            self_for_norm = torch.diagonal(self_map_matrix, dim1=-2, dim2=-1)
            cross_for_norm = torch.diagonal(cross_matrix, dim1=-2, dim2=-1)
            norm_self, norm_cross_raw = layer._apply_norm(
                self_for_norm,
                cross_for_norm,
                layer_edges,
                x.size(0),
            )
            norm_cross = torch.diag_embed(norm_cross_raw)
        elif isinstance(
            layer,
            (
                OrthogonalNSDConv,
                IdentityNSDConv,
                FrozenOrthogonalNSDConv,
            ),
        ):
            norm_self, norm_cross = layer._apply_norm(
                None,
                cross_matrix,
                layer_edges,
                x.size(0),
            )
        else:
            norm_self, norm_cross = layer._apply_norm(
                self_map_matrix,
                cross_matrix,
                layer_edges,
                x.size(0),
            )
        del norm_self
        alpha = float(layer.alpha.detach().abs().item())
        edge_count = layer_edges.size(1)
        if max_edges_per_layer and edge_count > max_edges_per_layer:
            selected_rows = torch.linspace(
                0,
                edge_count - 1,
                steps=max_edges_per_layer,
                device=layer_edges.device,
            ).round().long().unique()
        else:
            selected_rows = torch.arange(edge_count, device=layer_edges.device)
        selected_edges = layer_edges[:, selected_rows].detach().cpu().t().tolist()
        for edge_row_tensor, (source, target) in zip(
            selected_rows, selected_edges, strict=True
        ):
            edge_row = int(edge_row_tensor.item())
            dst_diag = matrix_diagnostics(dst_map[edge_row])
            src_diag = matrix_diagnostics(src_map[edge_row])
            cross_diag = matrix_diagnostics(cross_matrix[edge_row])
            normalized_diag = matrix_diagnostics(norm_cross[edge_row])
            omega = (
                alpha
                * normalized_diag["fro"]
                / max(layer.stalk_dim**0.5, 1.0)
            )
            row = {
                "layer": layer_index,
                "source": source,
                "target": target,
                "is_self_loop": source == target,
                "alpha": alpha,
                "omega": omega,
            }
            for prefix, diagnostics in (
                ("restriction_dst", dst_diag),
                ("restriction_src", src_diag),
                ("transport", cross_diag),
                ("normalized_transport", normalized_diag),
            ):
                row.update(
                    {
                        f"{prefix}_{key}": value
                        for key, value in diagnostics.items()
                    }
                )
            rows.append(row)
            normalized[(layer_index, source, target)] = (
                norm_cross[edge_row].detach().cpu()
            )
            if source != target:
                edge = (min(source, target), max(source, target))
                strengths.setdefault(layer_index, {}).setdefault(edge, []).append(
                    omega
                )
    reduced_strengths = {
        layer: {
            edge: float(sum(values) / len(values))
            for edge, values in layer_strengths.items()
        }
        for layer, layer_strengths in strengths.items()
    }
    return GeometrySnapshot(
        table=pd.DataFrame(rows),
        normalized_transports=normalized,
        strengths_by_layer=reduced_strengths,
    )


def canonical_transport_product(
    snapshot: GeometrySnapshot,
    path: list[int],
) -> tuple[dict[str, Any], torch.Tensor | None]:
    """Multiply normalized sheaf transports along a canonical path."""
    if len(path) < 2:
        return {}, None
    product: torch.Tensor | None = None
    path_strengths: list[float] = []
    used_layers = min(len(path) - 1, len(snapshot.strengths_by_layer))
    for layer_index in range(used_layers):
        key = (layer_index, path[layer_index], path[layer_index + 1])
        block = snapshot.normalized_transports.get(key)
        if block is None:
            return {}, None
        undirected = (
            min(path[layer_index], path[layer_index + 1]),
            max(path[layer_index], path[layer_index + 1]),
        )
        strength = snapshot.strengths_by_layer.get(layer_index, {}).get(
            undirected
        )
        if strength is not None:
            path_strengths.append(strength)
        product = block if product is None else block @ product
    if product is None:
        return {}, None
    metrics = {
        f"transport_path_{key}": value
        for key, value in matrix_diagnostics(product).items()
    }
    if path_strengths:
        metrics.update(
            {
                "omega_path_product": float(torch.tensor(path_strengths).prod()),
                "omega_path_min": min(path_strengths),
                "omega_path_mean": sum(path_strengths) / len(path_strengths),
            }
        )
    return metrics, product
