"""Deterministic experiment-grid expansion."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize structured values deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for configuration or dataset metadata."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def analysis_architectures() -> list[dict[str, Any]]:
    """Return the representative and capacity-controlled analysis suite."""
    architectures: list[dict[str, Any]] = []
    for variant in ("general", "diagonal", "identity", "frozen_orthogonal"):
        for stalk_dim in (1, 2, 3, 5):
            architectures.append(
                {
                    "variant": variant,
                    "stalk_dim": stalk_dim,
                    "hidden_dim": 32,
                    "analysis_group": "representative",
                }
            )
    for stalk_dim in (2, 3, 5):
        architectures.append(
            {
                "variant": "orthogonal",
                "stalk_dim": stalk_dim,
                "hidden_dim": 32,
                "analysis_group": "representative",
            }
        )
    for variant in ("gcn", "gat", "graphsage", "mlp"):
        architectures.append(
            {
                "variant": variant,
                "stalk_dim": 1,
                "hidden_dim": 32,
                "analysis_group": "representative",
            }
        )
    controlled_widths = {1: 60, 2: 30, 3: 20, 5: 12}
    for variant in ("general", "diagonal", "identity", "frozen_orthogonal"):
        for stalk_dim, hidden_dim in controlled_widths.items():
            architectures.append(
                {
                    "variant": variant,
                    "stalk_dim": stalk_dim,
                    "hidden_dim": hidden_dim,
                    "analysis_group": "total_width_60",
                }
            )
    for stalk_dim in (2, 3, 5):
        architectures.append(
            {
                "variant": "orthogonal",
                "stalk_dim": stalk_dim,
                "hidden_dim": controlled_widths[stalk_dim],
                "analysis_group": "total_width_60",
            }
        )
    for variant in ("gcn", "gat", "graphsage", "mlp"):
        architectures.append(
            {
                "variant": variant,
                "stalk_dim": 1,
                "hidden_dim": 60,
                "analysis_group": "total_width_60",
            }
        )
    return architectures


def _dataset_settings(dataset: dict[str, Any]) -> Iterable[dict[str, Any]]:
    parameter = dataset.get("sweep_parameter")
    values = dataset.get("sweep_values")
    if parameter is None:
        yield copy.deepcopy(dataset)
        return
    if not values:
        raise ValueError(f"{dataset['name']} has no sweep_values")
    for value in values:
        setting = copy.deepcopy(dataset)
        setting.pop("sweep_parameter", None)
        setting.pop("sweep_values", None)
        setting.setdefault("params", {})[parameter] = value
        setting["setting"] = f"{parameter}={value}"
        yield setting


def expand_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand datasets, model combinations, and seeds in stable order."""
    runs: list[dict[str, Any]] = []
    grid = config["grid"]
    architectures = grid.get("architectures")
    if grid.get("preset") == "analysis":
        architectures = analysis_architectures()
    if architectures is None:
        architectures = [
            {
                "variant": variant,
                "stalk_dim": stalk_dim,
                "hidden_dim": hidden_dim,
            }
            for variant in grid["variants"]
            for stalk_dim in grid["stalk_dims"]
            for hidden_dim in grid["hidden_dims"]
        ]
    for dataset in config["datasets"]:
        for setting in _dataset_settings(dataset):
            for architecture in architectures:
                variant = architecture["variant"]
                stalk_dim = int(architecture["stalk_dim"])
                hidden_dim = int(architecture["hidden_dim"])
                for seed in config["seeds"]:
                    spec = {
                        "task": config["task"],
                        "profile": config["profile"],
                        "dataset": copy.deepcopy(setting),
                        "model": {
                            "variant": variant,
                            "stalk_dim": stalk_dim,
                            "hidden_dim": hidden_dim,
                            "num_layers": setting.get(
                                "num_layers",
                                config.get("model", {}).get(
                                    "num_layers", "auto"
                                ),
                            ),
                            "alpha": config.get("model", {}).get("alpha", 1.0),
                            "orth_strategy": "cayley",
                            "input_dropout": config.get("model", {}).get(
                                "input_dropout", 0.0
                            ),
                            "dropout": config.get("model", {}).get(
                                "dropout", 0.0
                            ),
                            "normalize_output": config.get("model", {}).get(
                                "normalize_output", True
                            ),
                            "analysis_group": architecture.get(
                                "analysis_group", "main"
                            ),
                        },
                        "training": copy.deepcopy(config["training"]),
                        "tracking": copy.deepcopy(
                            config.get("tracking", {"wandb": False})
                        ),
                        "seed": seed,
                    }
                    spec["config_hash"] = config_hash(spec)
                    setting_name = setting.get("setting", "default").replace(
                        "=", "-"
                    )
                    spec["run_id"] = (
                        f"{config['task']}-{setting['name']}-{setting_name}-"
                        f"{variant}-d{stalk_dim}-h{hidden_dim}-s{seed}-"
                        f"{spec['config_hash'][:10]}"
                    )
                    runs.append(spec)
    return runs


def count_runs_per_setting(runs: list[dict[str, Any]]) -> Counter:
    """Count expanded runs by dataset setting."""
    return Counter(
        (
            run["dataset"]["name"],
            run["dataset"].get("setting", "default"),
        )
        for run in runs
    )
