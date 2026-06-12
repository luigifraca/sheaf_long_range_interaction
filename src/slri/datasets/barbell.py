"""Bamberger-style barbell node-regression task."""

from __future__ import annotations

import math

import torch
from torch_geometric.data import Data

from slri.datasets.graph_utils import edge_index_from_undirected


def _barbell_topology(
    nodes_per_clique: int,
    num_bridge_edges: int,
    seed: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    if nodes_per_clique < 2:
        raise ValueError("nodes_per_clique must be at least 2")
    max_bridges = nodes_per_clique**2
    if not 1 <= num_bridge_edges <= max_bridges:
        raise ValueError(f"num_bridge_edges must be in [1, {max_bridges}]")

    edges: list[tuple[int, int]] = []
    for offset in (0, nodes_per_clique):
        for source in range(offset, offset + nodes_per_clique):
            for target in range(source + 1, offset + nodes_per_clique):
                edges.append((source, target))

    candidates = [
        (left, nodes_per_clique + right)
        for left in range(nodes_per_clique)
        for right in range(nodes_per_clique)
    ]
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(candidates), generator=generator)[
        :num_bridge_edges
    ].tolist()
    bridges = [candidates[index] for index in selected]
    edges.extend(bridges)
    return (
        edge_index_from_undirected(
            edges,
            num_nodes=2 * nodes_per_clique,
        ),
        bridges,
    )


def make_barbell_graph(
    *,
    nodes_per_clique: int = 10,
    num_bridge_edges: int = 1,
    feature_dim: int = 40,
    topology_seed: int = 0,
    feature_seed: int = 0,
) -> Data:
    """Generate one barbell sample with opposite-clique mean targets."""
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    edge_index, bridges = _barbell_topology(
        nodes_per_clique,
        num_bridge_edges,
        topology_seed,
    )
    generator = torch.Generator().manual_seed(feature_seed)
    scale = math.sqrt(3.0)
    left = torch.rand(
        (nodes_per_clique, feature_dim), generator=generator
    ) * scale - scale
    right = torch.rand(
        (nodes_per_clique, feature_dim), generator=generator
    ) * scale
    x = torch.cat((left, right), dim=0)
    left_target = right.mean(dim=0).expand(nodes_per_clique, -1)
    right_target = left.mean(dim=0).expand(nodes_per_clique, -1)
    y = torch.cat((left_target, right_target), dim=0)
    cluster = torch.cat(
        (
            torch.zeros(nodes_per_clique, dtype=torch.long),
            torch.ones(nodes_per_clique, dtype=torch.long),
        )
    )
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        cluster=cluster,
        bridge_edges=torch.tensor(bridges, dtype=torch.long),
    )


def generate_barbell_dataset(
    *,
    nodes_per_clique: int = 10,
    num_bridge_edges: int = 1,
    feature_dim: int = 40,
    samples: int,
    seed: int,
) -> list[Data]:
    """Generate deterministic barbell samples sharing one topology."""
    return [
        make_barbell_graph(
            nodes_per_clique=nodes_per_clique,
            num_bridge_edges=num_bridge_edges,
            feature_dim=feature_dim,
            topology_seed=seed,
            feature_seed=seed * 1_000_003 + index,
        )
        for index in range(samples)
    ]

