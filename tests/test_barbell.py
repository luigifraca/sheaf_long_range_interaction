import math

import torch

from slri.datasets.barbell import generate_barbell_dataset, make_barbell_graph


def test_barbell_topology_and_unique_bridges():
    graph = make_barbell_graph(
        nodes_per_clique=5,
        num_bridge_edges=4,
        feature_dim=3,
        topology_seed=7,
        feature_seed=9,
    )
    assert graph.num_nodes == 10
    assert graph.edge_index.size(1) == 2 * (2 * math.comb(5, 2) + 4)
    assert graph.bridge_edges.unique(dim=0).size(0) == 4
    assert torch.all(graph.bridge_edges[:, 0] < 5)
    assert torch.all(graph.bridge_edges[:, 1] >= 5)
    assert graph.x.shape == graph.y.shape == (10, 3)


def test_barbell_generation_is_deterministic():
    first = generate_barbell_dataset(
        nodes_per_clique=4,
        num_bridge_edges=2,
        feature_dim=5,
        samples=3,
        seed=11,
    )
    second = generate_barbell_dataset(
        nodes_per_clique=4,
        num_bridge_edges=2,
        feature_dim=5,
        samples=3,
        seed=11,
    )
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.edge_index, right.edge_index)
        assert torch.equal(left.x, right.x)
        assert torch.equal(left.y, right.y)


def test_barbell_default_baselines_match_paper_scale():
    dataset = generate_barbell_dataset(
        nodes_per_clique=10,
        num_bridge_edges=1,
        feature_dim=40,
        samples=500,
        seed=3,
    )
    targets = torch.stack([graph.y for graph in dataset])
    zero_error = targets.pow(2).sum(dim=-1).mean().item()

    expected = torch.where(
        dataset[0].cluster.view(1, -1, 1) == 0,
        torch.full_like(targets, math.sqrt(3) / 2),
        torch.full_like(targets, -math.sqrt(3) / 2),
    )
    cluster_error = (targets - expected).pow(2).sum(dim=-1).mean().item()
    assert 29.0 < zero_error < 33.0
    assert 0.8 < cluster_error < 1.2

