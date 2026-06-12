from collections import Counter

import pytest
import torch

from slri.datasets.transfer import (
    TRANSFER_SIZES,
    TREE_DEPTHS,
    generate_transfer_dataset,
    make_transfer_graph,
)


@pytest.mark.parametrize("name", ["ring", "crossed_ring", "clique_path"])
@pytest.mark.parametrize("size", TRANSFER_SIZES)
def test_restricted_transfer_topologies(name, size):
    graph = make_transfer_graph(name, label=2, size=size)
    assert graph.num_nodes == size
    assert graph.source_mask.sum() == 1
    assert graph.target_mask.sum() == 1
    assert graph.source_target_distance.item() >= 1
    assert graph.y.item() == 2
    assert torch.equal(graph.x[graph.target_mask][0], torch.tensor([0, 0, 1, 0, 0]))


@pytest.mark.parametrize("depth", TREE_DEPTHS)
def test_binary_tree_distance_equals_depth(depth):
    graph = make_transfer_graph("tree", label=0, depth=depth, arity=2)
    assert graph.source_target_distance.item() == depth
    assert graph.num_nodes == 2 ** (depth + 1) - 1


def test_transfer_dataset_is_balanced_and_deterministic():
    first = generate_transfer_dataset(
        "ring", size=10, samples=100, seed=42
    )
    second = generate_transfer_dataset(
        "ring", size=10, samples=100, seed=42
    )
    assert Counter(graph.y.item() for graph in first) == {
        0: 20,
        1: 20,
        2: 20,
        3: 20,
        4: 20,
    }
    assert [graph.y.item() for graph in first] == [
        graph.y.item() for graph in second
    ]


def test_invalid_size_is_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        make_transfer_graph("ring", label=0, size=7)


def test_legacy_keeps_duplicate_edges():
    clean = make_transfer_graph("crossed_ring", label=0, size=10, protocol="clean")
    legacy = make_transfer_graph(
        "crossed_ring", label=0, size=10, protocol="legacy"
    )
    assert legacy.edge_index.size(1) > clean.edge_index.size(1)

