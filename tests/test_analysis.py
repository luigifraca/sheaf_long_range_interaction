from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import total_influence

from slri.analysis.curvature import curvature_payload
from slri.analysis.geometry import (
    canonical_transport_product,
    extract_orthogonal_restriction_rotations,
    extract_sheaf_geometry,
)
from slri.analysis.influence import (
    aggregate_hop_influence,
    compute_target_jacobians,
)
from slri.analysis.pathwise import compute_pathwise_jacobian
from slri.analysis.plotting import plot_orthogonal_restriction_rotations
from slri.analysis.runner import analyze_run, load_analysis_config
from slri.config import load_config
from slri.datasets.transfer import make_transfer_graph
from slri.grid import analysis_architectures, expand_grid
from slri.models import (
    FrozenOrthogonalNSDConv,
    build_model,
)
from slri.storage import Storage
from slri.training import run_spec

ROOT = Path(__file__).parents[1]


def _cycle_edges(nodes: int) -> torch.Tensor:
    pairs = []
    for index in range(nodes):
        neighbor = (index + 1) % nodes
        pairs.extend(((index, neighbor), (neighbor, index)))
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def test_city_l1_aggregation_matches_pyg_total_influence():
    torch.manual_seed(7)
    data = Data(
        x=torch.randn(4, 3),
        edge_index=_cycle_edges(4),
        y=torch.tensor([0, 1, 0, 1]),
    )
    model = build_model(
        variant="mlp",
        in_channels=3,
        out_channels=2,
        stalk_dim=1,
        hidden_dim=4,
        num_layers=2,
        normalize_output=False,
    ).eval()
    tables = []
    for target in range(data.num_nodes):
        table, _ = compute_target_jacobians(
            model, data, target=target, max_hops=2
        )
        table.insert(0, "sample_index", 0)
        tables.append(table)
    import pandas as pd

    pair_table = pd.concat(tables, ignore_index=True)
    assert {
        "influence_signed",
        "sigma_min_nonzero",
        "condition_number",
        "effective_rank",
    }.isdisjoint(pair_table.columns)
    ours, _ = aggregate_hop_influence(pair_table, bootstrap_samples=0)
    assert {"mean_l1", "normalized_mean_l1"}.isdisjoint(ours.columns)
    pyg, _ = total_influence(
        model,
        data,
        max_hops=2,
        num_samples=None,
        normalize=True,
        average=True,
        vectorize=True,
    )
    torch.testing.assert_close(
        torch.tensor(
            ours["normalized_total_l1"].to_numpy(), dtype=pyg.dtype
        ),
        pyg,
        rtol=1e-5,
        atol=1e-6,
    )


def test_scalar_vjp_matches_selected_full_jacobian_row():
    torch.manual_seed(11)
    data = Data(
        x=torch.randn(4, 3),
        edge_index=_cycle_edges(4),
        y=torch.tensor([0, 1, 0, 1]),
    )
    model = build_model(
        variant="mlp",
        in_channels=3,
        out_channels=2,
        stalk_dim=1,
        hidden_dim=4,
        num_layers=2,
        normalize_output=False,
    ).eval()
    full, _ = compute_target_jacobians(
        model, data, target=1, max_hops=2, target_class=1
    )
    scalar, _ = compute_target_jacobians(
        model,
        data,
        target=1,
        max_hops=2,
        target_class=1,
        output_index=1,
        metric_scope="ground_truth_logit",
    )
    expected = full.set_index("source")["ground_truth_l2"].sort_index()
    observed = scalar.set_index("source")["influence_fro"].sort_index()
    assert observed.index.equals(expected.index)
    torch.testing.assert_close(
        torch.tensor(observed.to_numpy()),
        torch.tensor(expected.to_numpy()),
    )


def test_shortest_path_sum_equals_full_jacobian_at_exact_depth():
    torch.manual_seed(3)
    graph = make_transfer_graph("ring", label=1, size=6)
    model = build_model(
        variant="identity",
        in_channels=5,
        out_channels=5,
        stalk_dim=2,
        hidden_dim=3,
        num_layers=3,
        normalize_output=False,
    )
    row, matrices = compute_pathwise_jacobian(
        model,
        graph.x,
        graph.edge_index,
        source=int(graph.target_index.item()),
        target=int(graph.source_index.item()),
    )
    assert row["distance"] == row["num_layers"] == 3
    assert row["path_count"] == 2
    torch.testing.assert_close(
        matrices["full"], matrices["geodesic"], rtol=1e-5, atol=1e-6
    )


class _CancellingPaths(nn.Module):
    num_layers = 2
    hidden_dim = 1
    normalize_output = False

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)
        self.decoder = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.encoder.weight.fill_(1)
            self.decoder.weight.fill_(1)
        first = torch.zeros(4, 4)
        first[1, 0] = 1
        first[2, 0] = 1
        second = torch.zeros(4, 4)
        second[3, 1] = 1
        second[3, 2] = -1
        self.register_buffer("first", first)
        self.register_buffer("second", second)

    def initial_state(self, x):
        return self.encoder(x)

    def layer_step(self, layer_index, hidden, edge_index):
        del edge_index
        matrix = self.first if layer_index == 0 else self.second
        return matrix @ hidden

    def decode_state(self, hidden):
        return self.decoder(hidden)

    def forward(self, x, edge_index):
        hidden = self.initial_state(x)
        for layer_index in range(self.num_layers):
            hidden = self.layer_step(layer_index, hidden, edge_index)
        return self.decode_state(hidden)


def test_parallel_shortest_paths_report_exact_cancellation():
    edges = torch.tensor(
        [
            [0, 1, 0, 2, 1, 3, 2, 3],
            [1, 0, 2, 0, 3, 1, 3, 2],
        ],
        dtype=torch.long,
    )
    row, matrices = compute_pathwise_jacobian(
        _CancellingPaths(),
        torch.ones(4, 1),
        edges,
        source=0,
        target=3,
    )
    assert row["path_count"] == 2
    assert "path_norm_sum" not in row
    assert row["path_cancellation"] == 0.0
    assert torch.equal(matrices["full"], torch.zeros(1, 1))
    assert torch.equal(matrices["geodesic"], torch.zeros(1, 1))


def test_frozen_orthogonal_maps_are_fixed_and_parameter_free():
    layer = FrozenOrthogonalNSDConv(
        stalk_dim=3,
        in_channels=2,
        hidden_dim=2,
        map_seed=17,
    )
    edges = _cycle_edges(5)
    first_dst, first_src = layer.restriction_maps(edges)
    second_dst, second_src = layer.restriction_maps(edges)
    torch.testing.assert_close(first_dst, second_dst)
    torch.testing.assert_close(first_src, second_src)
    eye = torch.eye(3).expand(first_dst.size(0), -1, -1)
    torch.testing.assert_close(
        first_dst.transpose(-2, -1) @ first_dst,
        eye,
        rtol=1e-5,
        atol=1e-5,
    )
    assert not any("map" in name for name, _ in layer.named_parameters())


def test_geometry_extracts_all_sheaf_objects():
    graph = make_transfer_graph("ring", label=0, size=6)
    model = build_model(
        variant="general",
        in_channels=5,
        out_channels=5,
        stalk_dim=2,
        hidden_dim=3,
        num_layers=2,
        normalize_output=False,
    )
    snapshot = extract_sheaf_geometry(model, graph.x, graph.edge_index)
    assert snapshot is not None
    assert {
        "restriction_dst_fro",
        "transport_fro",
        "normalized_transport_fro",
        "laplacian_entry_fro",
        "diffusion_alpha",
        "omega",
    } <= set(snapshot.table)
    assert set(snapshot.strengths_by_layer) == {0, 1}
    assert set(snapshot.laplacian_by_layer) == {0, 1}
    first_row = snapshot.table[~snapshot.table["is_self_loop"]].iloc[0]
    assert first_row["laplacian_entry_fro"] == first_row["normalized_transport_fro"]
    metrics, product = canonical_transport_product(snapshot, [0, 1, 2])
    assert product is not None
    assert {
        "transport_path_fro",
        "omega_path_product",
        "omega_path_min",
        "omega_path_mean",
    } <= set(metrics)


def test_geometry_supports_deterministic_edge_sampling():
    graph = make_transfer_graph("ring", label=0, size=10)
    model = build_model(
        variant="general",
        in_channels=5,
        out_channels=5,
        stalk_dim=2,
        hidden_dim=3,
        num_layers=2,
        normalize_output=False,
    )
    first = extract_sheaf_geometry(
        model,
        graph.x,
        graph.edge_index,
        max_edges_per_layer=5,
        required_edges={(4, 5)},
    )
    second = extract_sheaf_geometry(
        model,
        graph.x,
        graph.edge_index,
        max_edges_per_layer=5,
        required_edges={(4, 5)},
    )
    assert first is not None and second is not None
    assert len(first.table) >= 10
    assert first.table[["layer", "source", "target"]].equals(
        second.table[["layer", "source", "target"]]
    )
    required = first.table[
        first.table.apply(
            lambda row: {int(row.source), int(row.target)} == {4, 5},
            axis=1,
        )
    ]
    assert set(required["layer"]) == {0, 1}


def test_curvature_payload_contains_multi_alpha_weight_schemes():
    graph = make_transfer_graph("ring", label=0, size=6)
    model = build_model(
        variant="general",
        in_channels=5,
        out_channels=5,
        stalk_dim=2,
        hidden_dim=3,
        num_layers=2,
        normalize_output=False,
    )
    snapshot = extract_sheaf_geometry(model, graph.x, graph.edge_index)
    assert snapshot is not None
    payload = curvature_payload(
        graph.edge_index,
        snapshot,
        alphas=[0.0, 0.5, 1.0],
    )
    assert payload["alphas"] == [0.0, 0.5, 1.0]
    assert set(payload["omega_by_layer"]) == {"0", "1"}
    assert set(payload["laplacian_by_layer"]) == {"0", "1"}
    first_layer = payload["laplacian_by_layer"]["0"]
    assert all(value >= 0 for value in first_layer.values())


def test_orthogonal_rotation_plots_for_d2_and_d3(tmp_path):
    graph = make_transfer_graph("ring", label=0, size=6)
    path = [3, 2, 1, 0]
    for stalk_dim in (2, 3):
        model = build_model(
            variant="orthogonal",
            in_channels=5,
            out_channels=5,
            stalk_dim=stalk_dim,
            hidden_dim=3,
            num_layers=2,
            normalize_output=False,
        )
        table = extract_orthogonal_restriction_rotations(
            model,
            graph.x,
            graph.edge_index,
            paths=[path],
        )
        assert not table.empty
        assert set(table["stalk_dim"]) == {stalk_dim}
        assert {"restriction_dst", "restriction_src", "transport"} <= set(
            table["map_kind"]
        )
        output = tmp_path / f"orthogonal-d{stalk_dim}.pdf"
        plot_orthogonal_restriction_rotations(table, output)
        assert output.exists()


def test_analysis_preset_contains_requested_controls():
    architectures = analysis_architectures()
    assert len(architectures) == 46
    assert {
        item["variant"] for item in architectures
    } >= {"gcn", "gat", "graphsage", "mlp", "frozen_orthogonal"}
    controlled = [
        item
        for item in architectures
        if item["analysis_group"] == "total_width_60"
        and item["variant"] == "general"
    ]
    assert {
        item["stalk_dim"] * item["hidden_dim"] for item in controlled
    } == {60}


def test_smoke_analysis_is_indexed_and_exportable(tmp_path):
    storage = Storage(tmp_path / "storage")
    runs = expand_grid(load_config(ROOT / "configs/transfer.yaml", "smoke"))
    spec = next(
        run
        for run in runs
        if run["model"]["variant"] == "identity"
        and run["model"]["stalk_dim"] == 2
        and run["model"]["hidden_dim"] == 16
    )
    trained = run_spec(spec, storage, device_name="cpu")
    config = load_analysis_config(ROOT / "configs/analysis.yaml", "smoke")
    result = analyze_run(
        spec["run_id"],
        storage,
        checkpoint="best",
        config=config,
        device_name="cpu",
    )
    analysis_id = result["analysis_id"]
    assert result["metric_name"] == trained["metric_name"]
    assert result["test_metric"] == trained["test_metric"]
    assert storage.show(spec["run_id"])["analyses"][0]["analysis_id"] == analysis_id
    files = storage.analysis_files(analysis_id)
    names = {item["relative_path"] for item in files}
    assert "tables/influence_pairs.parquet" in names
    assert "tables/path_jacobians.parquet" in names
    assert "tables/orthogonal_restriction_rotations.parquet" in names
    assert "figures/distance_influence.pdf" in names
    assert "figures/orthogonal_restriction_rotations.pdf" in names
    artifact_path = Path(storage.show_analysis(analysis_id)["artifact_path"])
    matrices = torch.load(
        artifact_path / "matrices" / "synthetic_jacobians.pt",
        weights_only=False,
    )
    assert matrices["layerwise_embeddings"]["node_indices"].numel() > 0
    assert matrices["layerwise_embeddings"]["states"]
    archive = storage.export_analyses(
        analysis_id, tmp_path / "analysis.tar.gz"
    )
    assert archive.exists()
