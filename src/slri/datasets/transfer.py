"""Graph-transfer classification tasks from the over-squashing literature."""

from __future__ import annotations

from collections import Counter

import torch
from torch_geometric.data import Data

from slri.datasets.graph_utils import (
    edge_index_from_undirected,
    shortest_path_distance,
)

TRANSFER_SIZES = (2, 6, 10, 20, 30)
TREE_DEPTHS = (2, 3, 4, 5, 6, 7, 8)
TRANSFER_NAMES = {"ring", "crossed_ring", "clique_path", "tree"}


def _ring_edges(nodes: int, crossed: bool) -> list[tuple[int, int]]:
    edges = [(index, (index + 1) % nodes) for index in range(nodes)]
    if crossed:
        for index in range(nodes // 2):
            partner = nodes - 1 - index
            if index != partner:
                edges.append((index, partner))
    return edges


def _clique_path_edges(nodes: int) -> list[tuple[int, int]]:
    clique_size = max(1, nodes // 2)
    edges = [
        (source, target)
        for source in range(clique_size)
        for target in range(source + 1, clique_size)
    ]
    for source in range(clique_size - 1, nodes - 1):
        edges.append((source, source + 1))
    return edges


def _tree_edges(depth: int, arity: int) -> tuple[list[tuple[int, int]], int]:
    if depth < 1:
        raise ValueError("tree depth must be positive")
    if arity < 2:
        raise ValueError("tree arity must be at least 2")
    num_nodes = (arity ** (depth + 1) - 1) // (arity - 1)
    edges: list[tuple[int, int]] = []
    for parent in range((num_nodes - 1) // arity):
        first_child = arity * parent + 1
        for child in range(first_child, min(first_child + arity, num_nodes)):
            edges.append((parent, child))
    return edges, num_nodes


def transfer_topology(
    name: str,
    *,
    size: int | None = None,
    depth: int | None = None,
    arity: int = 2,
    protocol: str = "clean",
) -> tuple[torch.Tensor, int, int, int]:
    """Return edge index, node count, source, and target for one task."""
    if name not in TRANSFER_NAMES:
        raise ValueError(f"Unknown transfer task {name!r}")
    if protocol not in {"clean", "legacy"}:
        raise ValueError("protocol must be 'clean' or 'legacy'")

    if name == "tree":
        if depth not in TREE_DEPTHS:
            raise ValueError(f"tree depth must be one of {TREE_DEPTHS}")
        edges, num_nodes = _tree_edges(depth, arity)
        source, target = 0, num_nodes - 1
    else:
        if size not in TRANSFER_SIZES:
            raise ValueError(f"{name} size must be one of {TRANSFER_SIZES}")
        num_nodes = size
        source = 0
        if name == "ring":
            edges = _ring_edges(num_nodes, crossed=False)
            target = num_nodes // 2
        elif name == "crossed_ring":
            edges = _ring_edges(num_nodes, crossed=True)
            target = num_nodes // 2
        else:
            if protocol == "legacy":
                clique_size = max(1, num_nodes // 2)
                edges = [
                    (source, target)
                    for source in range(clique_size)
                    for target in range(clique_size)
                    if source != target
                ]
                edges.extend(
                    (source, source + 1)
                    for source in range(clique_size - 1, num_nodes - 1)
                )
            else:
                edges = _clique_path_edges(num_nodes)
            target = num_nodes - 1

    edge_index = edge_index_from_undirected(
        edges,
        num_nodes=num_nodes,
        deduplicate=protocol == "clean",
    )
    return edge_index, num_nodes, source, target


def make_transfer_graph(
    name: str,
    *,
    label: int,
    classes: int = 5,
    size: int | None = None,
    depth: int | None = None,
    arity: int = 2,
    protocol: str = "clean",
) -> Data:
    """Generate one source-node graph-transfer example."""
    if not 0 <= label < classes:
        raise ValueError("label is outside the configured class range")
    edge_index, num_nodes, source, target = transfer_topology(
        name,
        size=size,
        depth=depth,
        arity=arity,
        protocol=protocol,
    )
    x = torch.ones((num_nodes, classes), dtype=torch.float32)
    x[source] = 0.0
    x[target] = torch.nn.functional.one_hot(
        torch.tensor(label), num_classes=classes
    ).float()
    source_mask = torch.zeros(num_nodes, dtype=torch.bool)
    source_mask[source] = True
    target_mask = torch.zeros(num_nodes, dtype=torch.bool)
    target_mask[target] = True
    distance = shortest_path_distance(edge_index, source, target, num_nodes)
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        source_mask=source_mask,
        target_mask=target_mask,
        source_index=torch.tensor([source]),
        target_index=torch.tensor([target]),
        source_target_distance=torch.tensor([distance]),
    )


def _balanced_labels(samples: int, classes: int, seed: int) -> list[int]:
    if samples < classes or samples % classes != 0:
        raise ValueError("samples must be a positive multiple of classes")
    labels = torch.arange(classes).repeat_interleave(samples // classes)
    order = torch.randperm(samples, generator=torch.Generator().manual_seed(seed))
    result = labels[order].tolist()
    assert set(Counter(result).values()) == {samples // classes}
    return result


def generate_transfer_dataset(
    name: str,
    *,
    samples: int,
    seed: int,
    classes: int = 5,
    size: int | None = None,
    depth: int | None = None,
    arity: int = 2,
    protocol: str = "clean",
) -> list[Data]:
    """Generate a deterministic, class-balanced transfer dataset."""
    return [
        make_transfer_graph(
            name,
            label=label,
            classes=classes,
            size=size,
            depth=depth,
            arity=arity,
            protocol=protocol,
        )
        for label in _balanced_labels(samples, classes, seed)
    ]
