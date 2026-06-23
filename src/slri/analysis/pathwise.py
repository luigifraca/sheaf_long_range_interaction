"""Shortest-path chain-rule decomposition of node-to-node Jacobians."""

from __future__ import annotations

from collections import deque
from functools import cache
from typing import Any

import torch
from torch import nn

from slri.analysis.adapters import ModelTraceAdapter
from slri.analysis.influence import jacobian_metrics, shortest_path_distances


def _adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[set[int]]:
    result = [set() for _ in range(num_nodes)]
    for source, target in edge_index.detach().cpu().t().tolist():
        result[source].add(target)
    return result


def _replace_row(
    matrix: torch.Tensor, row: int, value: torch.Tensor
) -> torch.Tensor:
    indices = torch.tensor([row], device=matrix.device)
    return torch.index_copy(matrix, 0, indices, value.unsqueeze(0))


def compute_pathwise_jacobian(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    source: int,
    target: int,
    max_enumerated_paths: int = 10_000,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Compare the full Jacobian with geodesic chain-rule contributions."""
    model = model.to(dtype=torch.float32).eval()
    x = x.to(next(model.parameters()).device, dtype=torch.float32)
    edge_index = edge_index.to(x.device)
    adapter = ModelTraceAdapter(model)
    states, _ = adapter.trace(x, edge_index)
    num_nodes = x.size(0)
    distance_from_source = shortest_path_distances(
        edge_index, source, num_nodes
    )
    distance_to_target = shortest_path_distances(
        edge_index, target, num_nodes
    )
    distance = int(distance_from_source[target].item())
    if distance < 0:
        raise ValueError(f"No path from {source} to {target}")
    geodesic_nodes = {
        node
        for node in range(num_nodes)
        if int(distance_from_source[node] + distance_to_target[node]) == distance
    }
    adjacency = _adjacency(edge_index, num_nodes)

    @cache
    def local_block(layer_index: int, node: int, predecessor: int) -> torch.Tensor:
        hidden = states[layer_index].detach().requires_grad_(True)

        def output_node(value: torch.Tensor) -> torch.Tensor:
            return adapter.layer_step(layer_index, value, edge_index)[node]

        jacobian = torch.func.jacrev(output_node)(hidden)
        return jacobian[:, predecessor, :]

    hidden_size = adapter.hidden_size
    identity = torch.eye(hidden_size, device=x.device)
    current: dict[int, torch.Tensor] = {source: identity}
    for layer_index in range(adapter.num_layers):
        following: dict[int, torch.Tensor] = {}
        for predecessor, product in current.items():
            candidates = {predecessor} | adjacency[predecessor]
            for node in sorted(candidates & geodesic_nodes):
                progress = int(
                    distance_from_source[node]
                    - distance_from_source[predecessor]
                )
                if not (
                    progress == 1
                    or (progress == 0 and node == predecessor)
                ):
                    continue
                contribution = local_block(
                    layer_index, node, predecessor
                ) @ product
                following[node] = (
                    following[node] + contribution
                    if node in following
                    else contribution
                )
        current = following
    hidden_geodesic = current.get(
        target, torch.zeros_like(identity)
    )
    encoder = adapter.encoder_jacobian()
    decoder = adapter.decoder_jacobian(states[-1][target])
    geodesic = decoder @ hidden_geodesic @ encoder

    def full_output(features: torch.Tensor) -> torch.Tensor:
        return adapter.forward(features, edge_index)[target]

    full_all = torch.func.jacrev(full_output)(x)
    full = full_all[:, source, :]

    canonical_path = _canonical_shortest_path(
        adjacency, distance_to_target, source, target
    )
    extra_self_steps = max(adapter.num_layers - distance, 0)
    canonical_walk = [source] * (extra_self_steps + 1) + canonical_path[1:]
    if len(canonical_walk) < adapter.num_layers + 1:
        canonical_walk.extend(
            [target] * (adapter.num_layers + 1 - len(canonical_walk))
        )
    canonical_hidden = identity
    local_fro: list[float] = []
    local_spectral: list[float] = []
    if adapter.num_layers >= distance:
        for layer_index in range(adapter.num_layers):
            block = local_block(
                layer_index,
                canonical_walk[layer_index + 1],
                canonical_walk[layer_index],
            )
            local_fro.append(float(torch.linalg.matrix_norm(block).item()))
            local_spectral.append(float(torch.linalg.svdvals(block).max().item()))
            canonical_hidden = block @ canonical_hidden
        canonical = decoder @ canonical_hidden @ encoder
    else:
        canonical = torch.zeros_like(full)

    path_count = _count_monotone_walks(
        adjacency,
        geodesic_nodes,
        distance_from_source,
        source,
        target,
        adapter.num_layers,
    )
    path_norm_sum = float("nan")
    enumeration_truncated = path_count > max_enumerated_paths
    if not enumeration_truncated:
        products = _enumerate_products(
            adjacency=adjacency,
            geodesic_nodes=geodesic_nodes,
            distances=distance_from_source,
            source=source,
            target=target,
            num_layers=adapter.num_layers,
            local_block=local_block,
            identity=identity,
        )
        path_norm_sum = float(
            sum(
                torch.linalg.matrix_norm(decoder @ product @ encoder).item()
                for product in products
            )
        )

    residual = full - geodesic
    layerwise_hidden_fro: list[float] = []
    layerwise_hidden_spectral: list[float] = []
    for depth in range(adapter.num_layers + 1):
        def hidden_output(
            features: torch.Tensor, current_depth: int = depth
        ) -> torch.Tensor:
            hidden = adapter.initial_state(features)
            for layer_index in range(current_depth):
                hidden = adapter.layer_step(layer_index, hidden, edge_index)
            return hidden[target]

        hidden_jacobian = torch.func.jacrev(hidden_output)(x)[:, source, :]
        layerwise_hidden_fro.append(
            float(torch.linalg.matrix_norm(hidden_jacobian).item())
        )
        singular = torch.linalg.svdvals(hidden_jacobian)
        layerwise_hidden_spectral.append(
            float(singular.max().item()) if singular.numel() else 0.0
        )
    full_norm = float(torch.linalg.matrix_norm(full).item())
    geodesic_norm = float(torch.linalg.matrix_norm(geodesic).item())
    cancellation = (
        geodesic_norm / path_norm_sum
        if path_norm_sum and path_norm_sum > 0
        else float("nan")
    )
    row = {
        "source": source,
        "target": target,
        "distance": distance,
        "num_layers": adapter.num_layers,
        "path_count": path_count,
        "enumeration_truncated": enumeration_truncated,
        "path_cancellation": cancellation,
        "geodesic_to_full_ratio": geodesic_norm / max(full_norm, 1e-30),
        "non_geodesic_fro": float(torch.linalg.matrix_norm(residual).item()),
        **{
            f"full_{key}": value
            for key, value in jacobian_metrics(full).items()
            if key != "singular_values"
        },
        **{
            f"geodesic_{key}": value
            for key, value in jacobian_metrics(geodesic).items()
            if key != "singular_values"
        },
        **{
            f"canonical_{key}": value
            for key, value in jacobian_metrics(canonical).items()
            if key != "singular_values"
        },
        "canonical_path": canonical_path,
        "canonical_walk": canonical_walk,
        "canonical_local_fro": local_fro,
        "canonical_local_spectral": local_spectral,
        "layerwise_hidden_fro": layerwise_hidden_fro,
        "layerwise_hidden_spectral": layerwise_hidden_spectral,
        "full_singular_values": torch.linalg.svdvals(full).detach().cpu().tolist(),
        "geodesic_singular_values": (
            torch.linalg.svdvals(geodesic).detach().cpu().tolist()
        ),
    }
    matrices = {
        "full": full.detach().cpu(),
        "geodesic": geodesic.detach().cpu(),
        "canonical": canonical.detach().cpu(),
        "non_geodesic": residual.detach().cpu(),
    }
    return row, matrices


def _canonical_shortest_path(
    adjacency: list[set[int]],
    distance_to_target: torch.Tensor,
    source: int,
    target: int,
) -> list[int]:
    path = [source]
    node = source
    while node != target:
        candidates = [
            neighbor
            for neighbor in adjacency[node]
            if distance_to_target[neighbor] == distance_to_target[node] - 1
        ]
        if not candidates:
            raise ValueError("Could not reconstruct a shortest path")
        node = min(candidates)
        path.append(node)
    return path


def _count_monotone_walks(
    adjacency: list[set[int]],
    geodesic_nodes: set[int],
    distances: torch.Tensor,
    source: int,
    target: int,
    num_layers: int,
) -> int:
    counts = {source: 1}
    for _ in range(num_layers):
        following: dict[int, int] = {}
        for predecessor, count in counts.items():
            for node in ({predecessor} | adjacency[predecessor]) & geodesic_nodes:
                progress = int(distances[node] - distances[predecessor])
                if progress == 1 or (progress == 0 and node == predecessor):
                    following[node] = following.get(node, 0) + count
        counts = following
    return counts.get(target, 0)


def _enumerate_products(
    *,
    adjacency: list[set[int]],
    geodesic_nodes: set[int],
    distances: torch.Tensor,
    source: int,
    target: int,
    num_layers: int,
    local_block,
    identity: torch.Tensor,
) -> list[torch.Tensor]:
    products: list[torch.Tensor] = []
    queue = deque([(0, source, identity)])
    while queue:
        layer_index, predecessor, product = queue.popleft()
        if layer_index == num_layers:
            if predecessor == target:
                products.append(product)
            continue
        candidates = ({predecessor} | adjacency[predecessor]) & geodesic_nodes
        for node in sorted(candidates):
            progress = int(distances[node] - distances[predecessor])
            if progress == 1 or (progress == 0 and node == predecessor):
                queue.append(
                    (
                        layer_index + 1,
                        node,
                        local_block(layer_index, node, predecessor) @ product,
                    )
                )
    return products
