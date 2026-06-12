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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    edges = [tuple(edge) for edge in payload["edges"]]
    original = nx.Graph()
    original.add_nodes_from(range(payload["num_nodes"]))
    original.add_edges_from(edges, weight=1.0)
    common = {
        "alpha": float(payload["alpha"]),
        "method": payload["method"],
        "exp_power": int(payload["exp_power"]),
        "proc": int(payload["proc"]),
        "shortest_path": payload["shortest_path"],
    }
    original_curvature = _curvature(original, **common)
    rows = []
    epsilon = float(payload["epsilon"])
    for layer_text, strengths in payload["strengths_by_layer"].items():
        raw_lengths = {
            tuple(map(int, key.split(":"))): 1.0 / (epsilon + float(value))
            for key, value in strengths.items()
        }
        median = statistics.median(raw_lengths.values()) if raw_lengths else 1.0
        effective = nx.Graph()
        effective.add_nodes_from(range(payload["num_nodes"]))
        for edge in edges:
            length = raw_lengths.get(edge, 1.0) / max(median, epsilon)
            effective.add_edge(edge[0], edge[1], weight=length)
        effective_curvature = _curvature(effective, **common)
        for edge in edges:
            original_value = original_curvature[edge]
            effective_value = effective_curvature[edge]
            rows.append(
                {
                    "layer": int(layer_text),
                    "source": edge[0],
                    "target": edge[1],
                    "strength": float(
                        strengths.get(f"{edge[0]}:{edge[1]}", 0.0)
                    ),
                    "length": float(effective[edge[0]][edge[1]]["weight"]),
                    "original_curvature": original_value,
                    "effective_curvature": effective_value,
                    "curvature_change": effective_value - original_value,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer",
        "source",
        "target",
        "strength",
        "length",
        "original_curvature",
        "effective_curvature",
        "curvature_change",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
