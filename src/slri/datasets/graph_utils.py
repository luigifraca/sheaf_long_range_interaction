"""Small graph-construction utilities."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import torch
from torch_geometric.utils import coalesce


def edge_index_from_undirected(
    edges: Iterable[tuple[int, int]],
    *,
    num_nodes: int,
    deduplicate: bool = True,
) -> torch.Tensor:
    """Build a bidirectional COO edge index from undirected pairs."""
    directed: list[tuple[int, int]] = []
    for source, target in edges:
        if source == target:
            continue
        directed.extend(((source, target), (target, source)))
    if not directed:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    if deduplicate:
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
    return edge_index


def shortest_path_distance(
    edge_index: torch.Tensor,
    source: int,
    target: int,
    num_nodes: int,
) -> int:
    """Return the unweighted shortest-path distance using BFS."""
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for start, end in edge_index.t().tolist():
        adjacency[start].append(end)
    queue: deque[tuple[int, int]] = deque([(source, 0)])
    seen = {source}
    while queue:
        node, distance = queue.popleft()
        if node == target:
            return distance
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    raise ValueError(f"No path from node {source} to node {target}")

