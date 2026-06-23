#!/usr/bin/env python3
"""Run pinned GraphRicciCurvature from the isolated Python 3.12 environment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "GraphRicciCurvature"))

from GraphRicciCurvature.OllivierRicci import OllivierRicci  # noqa: E402


def _curvature(
    graph: nx.Graph,
    *,
    alpha: float,
    method: str,
    exp_power: int,
    proc: int,
    shortest_path: str,
) -> dict[tuple[int, int], float]:
    orc = OllivierRicci(
        graph,
        weight="weight",
        alpha=alpha,
        method=method,
        exp_power=exp_power,
        proc=proc,
        shortest_path=shortest_path,
        verbose="ERROR",
    )
    orc.compute_ricci_curvature()
    return {
        (min(source, target), max(source, target)): float(
            orc.G[source][target]["ricciCurvature"]
        )
        for source, target in orc.G.edges()
    }


def _safe_curvature(
    graph: nx.Graph,
    *,
    alpha: float,
    method: str,
    exp_power: int,
    proc: int,
    shortest_path: str,
) -> dict[tuple[int, int], float]:
    try:
        return _curvature(
            graph,
            alpha=alpha,
            method=method,
            exp_power=exp_power,
            proc=proc,
            shortest_path=shortest_path,
        )
    except Exception:
        return {
            (min(source, target), max(source, target)): float("nan")
            for source, target in graph.edges()
        }


def _graph(num_nodes: int, edges: list[tuple[int, int]], lengths) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for edge in edges:
        graph.add_edge(edge[0], edge[1], weight=float(lengths(edge)))
    return graph


def _edge_values(
    values: dict[str, float],
) -> dict[tuple[int, int], float]:
    return {
        tuple(map(int, key.split(":"))): float(value)
        for key, value in values.items()
    }


def _median_normalized(
    edges: list[tuple[int, int]],
    raw_lengths: dict[tuple[int, int], float],
    *,
    epsilon: float,
) -> dict[tuple[int, int], float]:
    values = [max(float(raw_lengths.get(edge, 1.0)), epsilon) for edge in edges]
    median = statistics.median(values) if values else 1.0
    scale = max(float(median), epsilon)
    return {
        edge: max(float(raw_lengths.get(edge, 1.0)), epsilon) / scale
        for edge in edges
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    edges = [tuple(edge) for edge in payload["edges"]]
    num_nodes = int(payload["num_nodes"])
    alphas = payload.get("alphas", [payload.get("alpha", 0.5)])
    common = {
        "method": payload["method"],
        "exp_power": int(payload["exp_power"]),
        "proc": int(payload["proc"]),
        "shortest_path": payload["shortest_path"],
    }
    unit = _graph(num_nodes, edges, lambda _edge: 1.0)
    rows = []
    epsilon = float(payload["epsilon"])
    omega_layers = payload.get(
        "omega_by_layer", payload.get("strengths_by_layer", {})
    )
    laplacian_layers = payload.get("laplacian_by_layer", omega_layers)
    all_layers = sorted(
        set(omega_layers) | set(laplacian_layers),
        key=lambda value: int(value),
    )
    for alpha in [float(value) for value in alphas]:
        unit_curvature = _safe_curvature(unit, alpha=alpha, **common)
        for layer_text in all_layers:
            omega_values = _edge_values(omega_layers.get(layer_text, {}))
            laplacian_values = _edge_values(laplacian_layers.get(layer_text, {}))
            raw_by_scheme = {
                "unit": {edge: 1.0 for edge in edges},
                "omega_inverse": {
                    edge: 1.0 / (epsilon + max(omega_values.get(edge, 0.0), 0.0))
                    if edge in omega_values
                    else 1.0
                    for edge in edges
                },
                "laplacian_fro": {
                    edge: laplacian_values.get(edge, 1.0) for edge in edges
                },
                "laplacian_fro_inverse": {
                    edge: 1.0
                    / (epsilon + max(laplacian_values.get(edge, 0.0), 0.0))
                    if edge in laplacian_values
                    else 1.0
                    for edge in edges
                },
            }
            references = {
                "unit": {edge: 1.0 for edge in edges},
                "omega_inverse": omega_values,
                "laplacian_fro": laplacian_values,
                "laplacian_fro_inverse": laplacian_values,
            }
            for scheme, raw_lengths in raw_by_scheme.items():
                lengths = _median_normalized(
                    edges, raw_lengths, epsilon=epsilon
                )
                graph = _graph(
                    num_nodes,
                    edges,
                    lambda edge, current=lengths: current[edge],
                )
                curvature = (
                    unit_curvature
                    if scheme == "unit"
                    else _safe_curvature(graph, alpha=alpha, **common)
                )
                for edge in edges:
                    unit_value = unit_curvature[edge]
                    value = curvature[edge]
                    default_reference = 1.0 if scheme == "unit" else 0.0
                    rows.append(
                        {
                            "curvature_alpha": alpha,
                            "length_scheme": scheme,
                            "layer": int(layer_text),
                            "source": edge[0],
                            "target": edge[1],
                            "reference_value": float(
                                references[scheme].get(edge, default_reference)
                            ),
                            "length": float(lengths[edge]),
                            "curvature": value,
                            "unit_curvature": unit_value,
                            "curvature_change_from_unit": value - unit_value,
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "curvature_alpha",
        "length_scheme",
        "layer",
        "source",
        "target",
        "reference_value",
        "length",
        "curvature",
        "unit_curvature",
        "curvature_change_from_unit",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
