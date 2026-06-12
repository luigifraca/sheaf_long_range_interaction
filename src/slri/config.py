"""Configuration loading and profile resolution."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from slri.models import MODEL_VARIANTS

DEFAULT_STORAGE_ROOT = Path(
    os.environ.get("SLRI_STORAGE_ROOT", "./slri-storage")
).expanduser()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, profile: str = "benchmark") -> dict[str, Any]:
    """Load a YAML configuration and apply one named profile."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    profiles = raw.pop("profiles", {})
    if profile not in profiles:
        known = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown profile {profile!r}; available: {known}")

    resolved = deep_merge(raw, profiles[profile])
    resolved["profile"] = profile
    validate_config(resolved)
    return resolved


def validate_config(config: dict[str, Any]) -> None:
    """Validate the cross-task configuration contract."""
    required = {"task", "datasets", "grid", "training", "seeds"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")

    if config["task"] not in {"barbell", "transfer", "cities"}:
        raise ValueError(f"Unsupported task: {config['task']!r}")

    grid = config["grid"]
    architectures = grid.get("architectures")
    if grid.get("preset") == "analysis":
        architectures = [
            {
                "variant": "general",
                "stalk_dim": 1,
                "hidden_dim": 32,
            }
        ]
    if architectures is not None:
        if not architectures or not isinstance(architectures, list):
            raise ValueError("grid.architectures must be a non-empty list")
        for architecture in architectures:
            variant = architecture.get("variant")
            stalk_dim = architecture.get("stalk_dim")
            hidden_dim = architecture.get("hidden_dim")
            if variant not in MODEL_VARIANTS:
                raise ValueError(f"Unknown model variant: {variant!r}")
            if not isinstance(stalk_dim, int) or stalk_dim <= 0:
                raise ValueError("architecture stalk_dim must be positive")
            if not isinstance(hidden_dim, int) or hidden_dim <= 0:
                raise ValueError("architecture hidden_dim must be positive")
            if variant == "orthogonal" and stalk_dim == 1:
                raise ValueError("orthogonal d=1 degenerates to Identity")
    else:
        expected_variants = {"general", "orthogonal", "diagonal", "identity"}
        variants = set(grid.get("variants", []))
        if variants != expected_variants:
            raise ValueError(
                "grid.variants must contain exactly general, orthogonal, "
                "diagonal, and identity"
            )

        stalk_dims = grid.get("stalk_dims")
        hidden_dims = grid.get("hidden_dims")
        if stalk_dims != [2, 3, 5]:
            raise ValueError("grid.stalk_dims must be [2, 3, 5]")
        if hidden_dims != [16, 32]:
            raise ValueError("grid.hidden_dims must be [16, 32]")

    seeds = config["seeds"]
    if not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be a non-empty list of integers")
