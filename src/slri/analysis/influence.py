"""Direct node-to-node Jacobian influence diagnostics."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from slri.models import forward_model


def shortest_path_distances(
    edge_index: torch.Tensor,
    source: int,
    num_nodes: int,
    max_hops: int | None = None,
) -> torch.Tensor:
    """Compute unweighted distances from one node by BFS."""
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for left, right in edge_index.detach().cpu().t().tolist():
        adjacency[left].append(right)
    distances = torch.full((num_nodes,), -1, dtype=torch.long)
    distances[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        if max_hops is not None and distances[node] >= max_hops:
            continue
        for neighbor in adjacency[node]:
            if distances[neighbor] < 0:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _jacrev_with_fallback(function, inputs: torch.Tensor) -> torch.Tensor:
    try:
        return torch.func.jacrev(function)(inputs)
    except (RuntimeError, NotImplementedError):
        return torch.autograd.functional.jacobian(
            function,
            inputs,
            vectorize=True,
        )


def jacobian_metrics(
    block: torch.Tensor,
    *,
    target_class: int | None = None,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Reduce one CxF Jacobian block to scalar and spectral diagnostics."""
    singular_values = torch.linalg.svdvals(block.float())
    positive = singular_values[singular_values > epsilon]
    sigma_max = float(singular_values.max().item()) if singular_values.numel() else 0.0
    ground_truth_l2 = float("nan")
    if target_class is not None and 0 <= target_class < block.size(0):
        ground_truth_l2 = float(
            torch.linalg.vector_norm(block[target_class]).item()
        )
    return {
        "influence_l1": float(block.abs().sum().item()),
        "influence_fro": float(torch.linalg.matrix_norm(block).item()),
        "influence_spectral": sigma_max,
        "ground_truth_l2": ground_truth_l2,
        "numerical_rank": int(positive.numel()),
        "singular_values": singular_values.detach().cpu().tolist(),
    }


def compute_target_jacobians(
    model: nn.Module,
    data: Data,
    *,
    target: int,
    max_hops: int,
    device: torch.device | str = "cpu",
    retain_matrices: bool = False,
    include_spectrum: bool = True,
    target_class: int | None = None,
    output_index: int | None = None,
    metric_scope: str = "full_logit_jacobian",
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    """Compute all source blocks influencing one target inside its ego graph."""
    device = torch.device(device)
    model = model.to(device=device, dtype=torch.float32).eval()
    edge_index = data.edge_index.to(device)
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        target,
        max_hops,
        edge_index,
        relabel_nodes=True,
        num_nodes=int(data.num_nodes),
    )
    root = int(mapping[0].item())
    sub_x = data.x[subset.cpu()].to(device=device, dtype=torch.float32)

    def target_output(features: torch.Tensor) -> torch.Tensor:
        return forward_model(model, features, sub_edge_index)[root]

    if output_index is None:
        jacobian = _jacrev_with_fallback(target_output, sub_x)
        metric_target_class = target_class
    else:
        def scalar_output(features: torch.Tensor) -> torch.Tensor:
            return target_output(features)[output_index]

        try:
            scalar, vjp = torch.func.vjp(scalar_output, sub_x)
            gradient = vjp(torch.ones_like(scalar))[0]
        except (RuntimeError, NotImplementedError):
            differentiable_x = sub_x.detach().requires_grad_(True)
            scalar = scalar_output(differentiable_x)
            gradient = torch.autograd.grad(scalar, differentiable_x)[0]
        jacobian = gradient.unsqueeze(0)
        metric_target_class = 0
    distances = shortest_path_distances(
        data.edge_index,
        target,
        int(data.num_nodes),
        max_hops=max_hops,
    )
    if (
        target_class is None
        and hasattr(data, "y")
        and data.y is not None
        and data.y.numel() > target
    ):
        if data.y.dtype in (torch.int32, torch.int64):
            target_class = int(data.y[target].item())
    rows: list[dict[str, Any]] = []
    matrices: dict[str, torch.Tensor] = {}
    for local_source, global_source in enumerate(subset.detach().cpu().tolist()):
        block = jacobian[:, local_source, :]
        metrics = jacobian_metrics(block, target_class=metric_target_class)
        if not include_spectrum:
            metrics["singular_values"] = None
        row = {
            "target": int(target),
            "source": int(global_source),
            "distance": int(distances[global_source].item()),
            "metric_scope": metric_scope,
            "output_index": output_index,
            **metrics,
        }
        if hasattr(data, "cluster"):
            source_cluster = int(data.cluster[global_source].item())
            target_cluster = int(data.cluster[target].item())
            row["pair_type"] = (
                "within_clique"
                if source_cluster == target_cluster
                else "cross_clique"
            )
        rows.append(row)
        if retain_matrices:
            matrices[f"{target}:{global_source}"] = block.detach().cpu()
    return pd.DataFrame(rows), matrices


def aggregate_hop_influence(
    pair_table: pd.DataFrame,
    *,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Aggregate pair influence into shell totals and receptive-field size."""
    if pair_table.empty:
        return pd.DataFrame(), {"influence_radius": float("nan")}
    target_columns = [
        column
        for column in ("sample_index", "target")
        if column in pair_table.columns
    ]
    grouped = (
        pair_table.groupby(target_columns + ["distance"], as_index=False)
        .agg(
            total_l1=("influence_l1", "sum"),
            total_fro=("influence_fro", "sum"),
            shell_size=("source", "count"),
        )
        .sort_values(target_columns + ["distance"])
    )
    zero_total = grouped[grouped["distance"] == 0][
        target_columns + ["total_l1"]
    ].rename(columns={"total_l1": "zero_total_l1"})
    grouped = grouped.merge(zero_total, on=target_columns, how="left")
    grouped["normalized_total_l1"] = (
        grouped["total_l1"] / grouped["zero_total_l1"].clip(lower=1e-30)
    )
    per_target_denominator = grouped.groupby(target_columns)["total_l1"].transform(
        "sum"
    )
    grouped["radius_contribution"] = (
        grouped["distance"]
        * grouped["total_l1"]
        / per_target_denominator.clip(lower=1e-30)
    )
    radius = float(
        grouped.groupby(target_columns)["radius_contribution"].sum().mean()
    )

    metric_columns = [
        "total_l1",
        "total_fro",
        "normalized_total_l1",
        "shell_size",
    ]
    summary = grouped.groupby("distance", as_index=False)[metric_columns].mean()
    rng = np.random.default_rng(seed)
    intervals: dict[int, tuple[float, float]] = {}
    values_by_target = {}
    for key, row_frame in grouped.groupby(target_columns):
        normalized_key = key if isinstance(key, tuple) else (key,)
        values_by_target[normalized_key] = row_frame.set_index("distance")[
            "normalized_total_l1"
        ]
    keys = list(values_by_target)
    distributions: dict[int, list[float]] = {
        int(distance): [] for distance in summary["distance"]
    }
    if keys and bootstrap_samples > 0:
        for _ in range(bootstrap_samples):
            sampled = rng.choice(len(keys), size=len(keys), replace=True)
            for distance in distributions:
                values = [
                    values_by_target[keys[index]].get(distance, np.nan)
                    for index in sampled
                ]
                distributions[distance].append(float(np.nanmean(values)))
        intervals = {
            distance: (
                float(np.nanpercentile(values, 2.5)),
                float(np.nanpercentile(values, 97.5)),
            )
            for distance, values in distributions.items()
        }
    summary["normalized_total_l1_ci_low"] = summary["distance"].map(
        lambda value, bounds=intervals: bounds.get(int(value), (float("nan"),))[0]
    )
    summary["normalized_total_l1_ci_high"] = summary["distance"].map(
        lambda value, bounds=intervals: bounds.get(
            int(value), (float("nan"), float("nan"))
        )[1]
    )

    positive = summary[
        (summary["distance"] > 0) & (summary["normalized_total_l1"] > 0)
    ]
    decay_slope = float("nan")
    if len(positive) >= 2:
        decay_slope = float(
            np.polyfit(
                positive["distance"],
                np.log10(positive["normalized_total_l1"].clip(lower=1e-30)),
                deg=1,
            )[0]
        )
    return summary, {
        "influence_radius": radius,
        "log10_total_decay_slope": decay_slope,
    }
