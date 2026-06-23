"""End-to-end orchestration for one stored run analysis."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from slri.analysis.adapters import ModelTraceAdapter
from slri.analysis.curvature import curvature_payload, run_curvature_sidecar
from slri.analysis.geometry import (
    GeometrySnapshot,
    canonical_transport_product,
    extract_orthogonal_restriction_rotations,
    extract_sheaf_geometry,
)
from slri.analysis.influence import (
    aggregate_hop_influence,
    compute_target_jacobians,
    shortest_path_distances,
)
from slri.analysis.pathwise import compute_pathwise_jacobian
from slri.analysis.plotting import (
    plot_anisotropy,
    plot_bottleneck,
    plot_curvature,
    plot_distance_influence,
    plot_orthogonal_restriction_rotations,
    plot_pathwise,
)
from slri.config import deep_merge
from slri.datasets import DatasetBundle, load_experiment_data
from slri.grid import config_hash
from slri.models import build_model
from slri.storage import Storage
from slri.training import resolve_device, set_seed

DEFAULT_ANALYSIS = {
    "max_hops": {"barbell": 16, "transfer": 16, "cities": 16},
    "influence": {
        "barbell_samples": 100,
        "transfer_samples": 500,
        "transfer_all_pairs_samples": 100,
        "city_targets": 10_000,
        "rich_targets": 256,
        "retain_matrix_targets": 8,
        "bootstrap_samples": 500,
    },
    "pathwise": {
        "barbell_samples": 20,
        "transfer_samples": 100,
        "city_targets": 16,
        "max_enumerated_paths": 10_000,
    },
    "geometry": {"enabled": True},
    "curvature": {
        "enabled": True,
        "alphas": [0.0, 0.5, 1.0],
        "epsilon": 1e-8,
        "proc": 1,
    },
    "layerwise": {"retain_embedding_nodes": 256},
}


def _normalize_analysis_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize backward-compatible analysis config fields."""
    result = copy.deepcopy(config)
    curvature = result.setdefault("curvature", {})
    if "alpha" in curvature:
        curvature["alphas"] = [float(curvature.pop("alpha"))]
    else:
        curvature["alphas"] = [
            float(value) for value in curvature.get("alphas", [0.0, 0.5, 1.0])
        ]
    return result


def load_analysis_config(
    path: str | Path | None = None,
    profile: str = "benchmark",
) -> dict[str, Any]:
    """Load analysis-only configuration without changing training grids."""
    if path is None:
        return _normalize_analysis_config(DEFAULT_ANALYSIS)
    raw = yaml.safe_load(Path(path).read_text())
    profiles = raw.pop("profiles", {})
    if profile not in profiles:
        raise ValueError(f"Unknown analysis profile {profile!r}")
    return _normalize_analysis_config(
        deep_merge(deep_merge(DEFAULT_ANALYSIS, raw), profiles[profile])
    )


def _write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


def _checkpoint_model(
    spec: dict[str, Any],
    bundle: DatasetBundle,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(
        variant=spec["model"]["variant"],
        in_channels=bundle.in_channels,
        out_channels=bundle.out_channels,
        stalk_dim=int(spec["model"]["stalk_dim"]),
        hidden_dim=int(spec["model"]["hidden_dim"]),
        num_layers=int(bundle.num_layers),
        alpha=float(spec["model"].get("alpha", 1.0)),
        orth_strategy=spec["model"].get("orth_strategy", "cayley"),
        input_dropout=float(spec["model"].get("input_dropout", 0.0)),
        dropout=float(spec["model"].get("dropout", 0.0)),
        normalize_output=bool(spec["model"].get("normalize_output", True)),
        seed=int(spec["seed"]),
    ).to(device=device, dtype=torch.float32)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.eval()


def _test_graphs(bundle: DatasetBundle) -> list[Data]:
    return bundle.test if isinstance(bundle.test, list) else [bundle.test]


def _target_class(data: Data, bundle: DatasetBundle, target: int) -> int | None:
    if bundle.task_type == "source_classification":
        return int(data.y.view(-1)[0].item())
    if bundle.task_type == "node_classification":
        return int(data.y[target].item())
    return None


def _city_targets(data: Data, count: int, seed: int) -> list[int]:
    candidates = torch.where(data.test_mask)[0]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[: min(count, candidates.numel())]].tolist()


def _influence_tables(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    config: dict[str, Any],
    *,
    task: str,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, torch.Tensor], dict[str, float]]:
    settings = config["influence"]
    max_hops = int(config["max_hops"][task])
    graphs = _test_graphs(bundle)
    pair_tables: list[pd.DataFrame] = []
    matrices: dict[str, torch.Tensor] = {}
    target_counter = 0
    if task == "barbell":
        selected = graphs[: int(settings["barbell_samples"])]
        schedules = [
            (index, graph, list(range(graph.num_nodes)))
            for index, graph in enumerate(selected)
        ]
    elif task == "transfer":
        selected = graphs[: int(settings["transfer_samples"])]
        all_pair_count = int(settings["transfer_all_pairs_samples"])
        schedules = []
        for index, graph in enumerate(selected):
            targets = (
                list(range(graph.num_nodes))
                if index < all_pair_count
                else [int(graph.source_index.item())]
            )
            schedules.append((index, graph, targets))
    else:
        graph = graphs[0]
        schedules = [
            (
                0,
                graph,
                _city_targets(graph, int(settings["city_targets"]), seed),
            )
        ]
    rich_targets = int(settings["rich_targets"])
    matrix_targets = int(settings["retain_matrix_targets"])
    for sample_index, graph, targets in schedules:
        for target in targets:
            rich = target_counter < rich_targets
            retain = target_counter < matrix_targets
            target_class = _target_class(graph, bundle, target)
            if task == "cities":
                scalar_table, _ = compute_target_jacobians(
                    model,
                    graph,
                    target=target,
                    max_hops=max_hops,
                    device=device,
                    target_class=target_class,
                    output_index=target_class,
                    metric_scope="ground_truth_logit",
                )
                scalar_table.insert(0, "sample_index", sample_index)
                pair_tables.append(scalar_table)
            if task != "cities" or rich:
                table, target_matrices = compute_target_jacobians(
                    model,
                    graph,
                    target=target,
                    max_hops=max_hops,
                    device=device,
                    retain_matrices=retain,
                    include_spectrum=rich,
                    target_class=target_class,
                )
                table.insert(0, "sample_index", sample_index)
                pair_tables.append(table)
                matrices.update(
                    {
                        f"{sample_index}:{key}": value
                        for key, value in target_matrices.items()
                    }
                )
            target_counter += 1
    pair_table = (
        pd.concat(pair_tables, ignore_index=True)
        if pair_tables
        else pd.DataFrame()
    )
    hop_tables: list[pd.DataFrame] = []
    influence_summary: dict[str, float] = {}
    for scope, frame in pair_table.groupby("metric_scope", sort=False):
        scope_hops, scope_summary = aggregate_hop_influence(
            frame,
            bootstrap_samples=int(settings["bootstrap_samples"]),
            seed=seed,
        )
        scope_hops.insert(0, "metric_scope", scope)
        hop_tables.append(scope_hops)
        influence_summary.update(
            {f"{key}_{scope}": value for key, value in scope_summary.items()}
        )
    hop_table = (
        pd.concat(hop_tables, ignore_index=True)
        if hop_tables
        else pd.DataFrame()
    )
    return pair_table, hop_table, matrices, influence_summary


def _pathwise_jobs(
    bundle: DatasetBundle,
    config: dict[str, Any],
    task: str,
    seed: int,
) -> list[tuple[int, Data, int, int]]:
    graphs = _test_graphs(bundle)
    jobs: list[tuple[int, Data, int, int]] = []
    if task == "barbell":
        for index, graph in enumerate(
            graphs[: int(config["pathwise"]["barbell_samples"])]
        ):
            left, right = graph.bridge_edges[0].tolist()
            jobs.extend(
                [(index, graph, left, right), (index, graph, right, left)]
            )
    elif task == "transfer":
        for index, graph in enumerate(
            graphs[: int(config["pathwise"]["transfer_samples"])]
        ):
            jobs.append(
                (
                    index,
                    graph,
                    int(graph.target_index.item()),
                    int(graph.source_index.item()),
                )
            )
    else:
        graph = graphs[0]
        targets = _city_targets(
            graph, int(config["pathwise"]["city_targets"]), seed + 9176
        )
        max_hops = int(config["max_hops"]["cities"])
        for index, target in enumerate(targets):
            subset, sub_edges, mapping, _ = k_hop_subgraph(
                target,
                max_hops,
                graph.edge_index,
                relabel_nodes=True,
                num_nodes=int(graph.num_nodes),
            )
            distances = shortest_path_distances(
                sub_edges,
                int(mapping[0].item()),
                subset.numel(),
            )
            source_local = int(torch.argmax(distances).item())
            subgraph = Data(
                x=graph.x[subset],
                edge_index=sub_edges,
                y=graph.y[subset],
            )
            jobs.append(
                (
                    index,
                    subgraph,
                    source_local,
                    int(mapping[0].item()),
                )
            )
    return jobs


def _canonical_path(
    edge_index: torch.Tensor,
    source: int,
    target: int,
    num_nodes: int,
) -> list[int]:
    distances = shortest_path_distances(edge_index, target, num_nodes)
    if int(distances[source].item()) < 0:
        return []
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for left, right in edge_index.detach().cpu().t().tolist():
        adjacency[left].append(right)
    path = [source]
    node = source
    while node != target:
        candidates = [
            neighbor
            for neighbor in adjacency[node]
            if int(distances[neighbor].item()) == int(distances[node].item()) - 1
        ]
        if not candidates:
            return []
        node = min(candidates)
        path.append(node)
    return path


def _critical_paths(
    bundle: DatasetBundle,
    config: dict[str, Any],
    task: str,
    seed: int,
) -> list[list[int]]:
    graphs = _test_graphs(bundle)
    if not graphs:
        return []
    graph = graphs[0]
    if task == "barbell" and hasattr(graph, "bridge_edges"):
        paths = []
        for left, right in graph.bridge_edges.detach().cpu().tolist():
            paths.extend(([int(left), int(right)], [int(right), int(left)]))
        return paths
    if task == "transfer":
        return [
            _canonical_path(
                graph.edge_index,
                int(graph.target_index.item()),
                int(graph.source_index.item()),
                int(graph.num_nodes),
            )
        ]
    if task == "cities":
        paths = []
        max_hops = int(config["max_hops"]["cities"])
        targets = _city_targets(
            graph, int(config["pathwise"]["city_targets"]), seed + 9176
        )
        for target in targets:
            subset, sub_edges, mapping, _ = k_hop_subgraph(
                target,
                max_hops,
                graph.edge_index,
                relabel_nodes=True,
                num_nodes=int(graph.num_nodes),
            )
            root = int(mapping[0].item())
            distances = shortest_path_distances(
                sub_edges,
                root,
                subset.numel(),
            )
            reachable = torch.where(distances >= 0)[0]
            if reachable.numel() == 0:
                continue
            source_local = int(
                reachable[torch.argmax(distances[reachable])].item()
            )
            local_path = _canonical_path(
                sub_edges, source_local, root, int(subset.numel())
            )
            if local_path:
                paths.append([int(subset[node].item()) for node in local_path])
        return paths
    return []


def _required_edges(paths: list[list[int]]) -> set[tuple[int, int]]:
    return {
        (min(source, target), max(source, target))
        for path in paths
        for source, target in zip(path, path[1:], strict=False)
    }


def _pathwise_table(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    config: dict[str, Any],
    *,
    task: str,
    seed: int,
    geometry: GeometrySnapshot | None,
) -> tuple[pd.DataFrame, dict[str, dict[str, torch.Tensor]]]:
    rows = []
    matrices = {}
    for sample_index, graph, source, target in _pathwise_jobs(
        bundle, config, task, seed
    ):
        row, result_matrices = compute_pathwise_jacobian(
            model,
            graph.x,
            graph.edge_index,
            source=source,
            target=target,
            max_enumerated_paths=int(
                config["pathwise"]["max_enumerated_paths"]
            ),
        )
        row["sample_index"] = sample_index
        if geometry is not None and sample_index == 0:
            transport_metrics, transport_matrix = canonical_transport_product(
                geometry, row["canonical_path"]
            )
            row.update(transport_metrics)
            if transport_matrix is not None:
                result_matrices["transport_path"] = transport_matrix
        rows.append(row)
        matrices[f"{sample_index}:{source}:{target}"] = result_matrices
    return pd.DataFrame(rows), matrices


def _layerwise_metrics(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    graph = _test_graphs(bundle)[0].to(device)
    adapter = ModelTraceAdapter(model)
    states, _ = adapter.trace(graph.x.float(), graph.edge_index)
    retain_count = min(
        int(config["layerwise"]["retain_embedding_nodes"]),
        int(graph.num_nodes),
    )
    candidates = (
        torch.where(graph.test_mask)[0]
        if bundle.task_type == "node_classification"
        else torch.arange(graph.num_nodes, device=device)
    ).detach().cpu()
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(candidates.numel(), generator=generator)
    retained_nodes_cpu = candidates[
        order[: min(retain_count, candidates.numel())]
    ]
    retained_nodes = retained_nodes_cpu.to(device)
    rows = []
    source, target = graph.edge_index
    for layer_index, hidden in enumerate(states):
        differences = hidden[source] - hidden[target]
        dirichlet = float(differences.square().sum(dim=-1).mean().item())
        row = {
            "layer": layer_index,
            "mean_node_norm": float(
                torch.linalg.vector_norm(hidden, dim=-1).mean().item()
            ),
            "feature_variance": float(hidden.var(dim=0).mean().item()),
            "dirichlet_energy": dirichlet,
        }
        rows.append(row)
    embeddings = {
        "node_indices": retained_nodes_cpu,
        "states": {
            str(layer_index): hidden[retained_nodes].detach().cpu()
            for layer_index, hidden in enumerate(states)
        },
    }
    return pd.DataFrame(rows), embeddings


def _correlations(path_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if path_table.empty:
        return pd.DataFrame()
    outcome = "full_influence_fro"
    candidates = [
        "geodesic_influence_fro",
        "canonical_influence_fro",
        "transport_path_fro",
        "transport_path_spectral",
        "omega_path_product",
        "omega_path_min",
        "omega_path_mean",
        "path_unit_curvature_mean",
        "path_effective_curvature_mean",
        "path_curvature_change_from_unit_mean",
        "path_effective_curvature_min",
        "path_cancellation",
        "distance",
    ]
    for predictor in candidates:
        if predictor not in path_table:
            continue
        frame = path_table[[outcome, predictor, "distance"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(frame) < 3 or frame[predictor].nunique() < 2:
            continue
        correlation = spearmanr(frame[outcome], frame[predictor]).statistic
        transformed = np.log10(frame[outcome].clip(lower=1e-30))
        design = frame[["distance", predictor]].to_numpy()
        regression = LinearRegression().fit(design, transformed)
        rows.append(
            {
                "outcome": outcome,
                "predictor": predictor,
                "spearman": float(correlation),
                "distance_controlled_coefficient": float(regression.coef_[1]),
                "distance_controlled_r2": float(
                    regression.score(design, transformed)
                ),
                "samples": len(frame),
            }
        )
    return pd.DataFrame(rows)


def _attach_path_curvature(
    path_table: pd.DataFrame, curvature: pd.DataFrame
) -> pd.DataFrame:
    if path_table.empty or curvature.empty:
        return path_table
    updated = path_table.copy()
    primary = curvature
    if "curvature_alpha" in primary:
        primary = primary[primary["curvature_alpha"] == 0.5]
    if "length_scheme" in primary:
        primary = primary[primary["length_scheme"] == "omega_inverse"]
    lookup = {
        (
            int(row.layer),
            min(int(row.source), int(row.target)),
            max(int(row.source), int(row.target)),
        ): row
        for row in primary.itertuples()
    }
    for index, row in updated.iterrows():
        path = row["canonical_path"]
        records = []
        for layer, (source, target) in enumerate(
            zip(path, path[1:], strict=False)
        ):
            record = lookup.get((layer, min(source, target), max(source, target)))
            if record is not None:
                records.append(record)
        if records:
            updated.loc[index, "path_unit_curvature_mean"] = np.mean(
                [record.unit_curvature for record in records]
            )
            updated.loc[index, "path_effective_curvature_mean"] = np.mean(
                [record.curvature for record in records]
            )
            updated.loc[index, "path_curvature_change_from_unit_mean"] = np.mean(
                [record.curvature_change_from_unit for record in records]
            )
            updated.loc[index, "path_effective_curvature_min"] = np.min(
                [record.curvature for record in records]
            )
    return updated


def analyze_run(
    run_id: str,
    storage: Storage,
    *,
    checkpoint: str = "best",
    config: dict[str, Any] | None = None,
    device_name: str = "auto",
    force: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze one checkpoint and persist all tables, matrices, and figures."""
    config = _normalize_analysis_config(deep_merge(DEFAULT_ANALYSIS, config or {}))
    run = storage.show(run_id)
    run_path = Path(run["artifact_path"])
    run_summary = run.get("summary") or {}
    spec = yaml.safe_load((run_path / "resolved_config.yaml").read_text())
    checkpoint_path = run_path / "checkpoints" / f"{checkpoint}.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    analysis_payload = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "analysis": config,
    }
    analysis_hash = config_hash(analysis_payload)
    analysis_id = f"{run_id}-{checkpoint}-{analysis_hash[:10]}"
    record = {
        "analysis_id": analysis_id,
        "run_id": run_id,
        "config_hash": analysis_hash,
        "task": run["task"],
        "dataset": run["dataset"],
        "variant": run["variant"],
        "checkpoint": checkpoint,
    }
    previous = storage.query_analyses(f"analysis_id={analysis_id}")
    if previous and previous[0]["status"] == "completed" and not force:
        return {"analysis_id": analysis_id, "status": "skipped"}
    resolved = {
        **analysis_payload,
        "analysis_id": analysis_id,
        "config_hash": analysis_hash,
        "task": run["task"],
        "dataset": run["dataset"],
        "model": spec["model"],
        "seed": spec["seed"],
    }
    path = storage.begin_analysis(record, resolved)
    set_seed(int(spec["seed"]))
    device = resolve_device(device_name)
    project_root = Path(project_root or Path(__file__).parents[3]).resolve()
    try:
        bundle = load_experiment_data(spec, storage)
        model = _checkpoint_model(spec, bundle, checkpoint_path, device)
        graphs = _test_graphs(bundle)
        critical_paths = _critical_paths(
            bundle, config, run["task"], int(spec["seed"])
        )
        geometry = None
        rotation_table = pd.DataFrame()
        if config["geometry"]["enabled"]:
            geometry = extract_sheaf_geometry(
                model,
                graphs[0].x,
                graphs[0].edge_index,
                max_edges_per_layer=config["geometry"].get(
                    "max_edges_per_layer"
                ),
                required_edges=_required_edges(critical_paths),
            )
            rotation_table = extract_orthogonal_restriction_rotations(
                model,
                graphs[0].x,
                graphs[0].edge_index,
                paths=critical_paths,
            )
        pair_table, hop_table, influence_matrices, influence_summary = (
            _influence_tables(
                model,
                bundle,
                config,
                task=run["task"],
                seed=int(spec["seed"]),
                device=device,
            )
        )
        path_table, path_matrices = _pathwise_table(
            model,
            bundle,
            config,
            task=run["task"],
            seed=int(spec["seed"]),
            geometry=geometry,
        )
        layer_table, layerwise_embeddings = _layerwise_metrics(
            model,
            bundle,
            config,
            device,
            int(spec["seed"]),
        )
        geometry_table = (
            geometry.table if geometry is not None else pd.DataFrame()
        )
        curvature_table = pd.DataFrame()
        curvature_status = "disabled"
        if config["curvature"]["enabled"]:
            payload = curvature_payload(
                graphs[0].edge_index,
                geometry,
                alphas=config["curvature"]["alphas"],
                epsilon=float(config["curvature"]["epsilon"]),
                proc=int(config["curvature"]["proc"]),
                shortest_path=(
                    "pairwise" if run["task"] == "cities" else "all_pairs"
                ),
            )
            try:
                curvature_table = run_curvature_sidecar(
                    payload,
                    path / "tables" / "curvature_edges.csv",
                    project_root=project_root,
                )
                curvature_status = "completed"
            except RuntimeError as exc:
                curvature_status = "unavailable"
                (path / "logs" / "curvature_error.txt").write_text(f"{exc}\n")
        path_table = _attach_path_curvature(path_table, curvature_table)
        correlation_table = _correlations(path_table)

        _write_table(pair_table, path / "tables" / "influence_pairs.parquet")
        _write_table(hop_table, path / "tables" / "influence_hops.parquet")
        _write_table(path_table, path / "tables" / "path_jacobians.parquet")
        _write_table(
            geometry_table, path / "tables" / "sheaf_edge_geometry.parquet"
        )
        _write_table(
            rotation_table,
            path / "tables" / "orthogonal_restriction_rotations.parquet",
        )
        _write_table(
            curvature_table, path / "tables" / "curvature_edges.parquet"
        )
        _write_table(
            correlation_table,
            path / "tables" / "geometry_influence_correlations.parquet",
        )
        _write_table(
            layer_table, path / "tables" / "layerwise_metrics.parquet"
        )
        matrix_name = (
            "sampled_city_jacobians.pt"
            if run["task"] == "cities"
            else "synthetic_jacobians.pt"
        )
        torch.save(
            {
                "influence": influence_matrices,
                "pathwise": path_matrices,
                "layerwise_embeddings": layerwise_embeddings,
            },
            path / "matrices" / matrix_name,
        )
        plot_distance_influence(
            hop_table, path / "figures" / "distance_influence.pdf"
        )
        plot_pathwise(
            path_table, path / "figures" / "pathwise_vs_full.pdf"
        )
        plot_curvature(
            curvature_table, path / "figures" / "curvature_influence.pdf"
        )
        plot_bottleneck(
            geometry_table, path / "figures" / "bottleneck_evolution.pdf"
        )
        plot_anisotropy(
            geometry_table, path / "figures" / "anisotropy_spectra.pdf"
        )
        plot_orthogonal_restriction_rotations(
            rotation_table,
            path / "figures" / "orthogonal_restriction_rotations.pdf",
        )
        summary = {
            "analysis_id": analysis_id,
            "run_id": run_id,
            "checkpoint": checkpoint,
            "status": "completed",
            "curvature_status": curvature_status,
            "metric_name": run_summary.get("metric_name"),
            "test_metric": run_summary.get("test_metric"),
            "test_metrics": run_summary.get("test_metrics", {}),
            "pair_rows": len(pair_table),
            "hop_rows": len(hop_table),
            "pathwise_rows": len(path_table),
            "geometry_rows": len(geometry_table),
            "orthogonal_rotation_rows": len(rotation_table),
            "curvature_rows": len(curvature_table),
            "layerwise_rows": len(layer_table),
            **influence_summary,
            "files": {
                "tables": "tables/",
                "matrices": f"matrices/{matrix_name}",
                "figures": "figures/",
            },
        }
        storage.complete_analysis(record, summary, path)
        return summary
    except BaseException as exc:
        storage.fail_analysis(record, path, exc)
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def compare_analyses(
    storage: Storage,
    query: str | None,
    output: str | Path,
) -> Path:
    """Build a portable summary CSV from indexed analyses."""
    rows = storage.query_analyses(query)
    records = []
    for row in rows:
        summary_path = Path(row["artifact_path"]) / "summary.json"
        summary = (
            json.loads(summary_path.read_text()) if summary_path.exists() else {}
        )
        records.append({**row, **summary})
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() != ".csv":
        output.mkdir(parents=True, exist_ok=True)
        output = output / "analysis_summary.csv"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    return output
