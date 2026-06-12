"""Experiment data assembly and optional materialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from slri.datasets.barbell import generate_barbell_dataset
from slri.datasets.base import DatasetBundle
from slri.datasets.cities import load_city
from slri.datasets.transfer import generate_transfer_dataset
from slri.grid import config_hash
from slri.storage import Storage


def _split_seeds(seed: int) -> tuple[int, int, int]:
    return seed * 10_000 + 11, seed * 10_000 + 23, seed * 10_000 + 37


def _cache_key(spec: dict[str, Any]) -> str:
    payload = {
        "task": spec["task"],
        "dataset": spec["dataset"],
        "seed": spec["seed"],
        "samples": spec["training"].get("samples"),
    }
    return config_hash(payload)


def materialize_dataset(
    spec: dict[str, Any],
    storage: Storage,
    *,
    force: bool = False,
) -> Path:
    """Generate and save the exact data bundle used by one run."""
    cache_dir = (
        storage.data_generated / spec["task"] / _cache_key(spec)
    )
    bundle_path = cache_dir / "dataset.pt"
    if bundle_path.exists() and not force:
        return bundle_path
    bundle = load_experiment_data(spec, storage, use_cache=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, bundle_path)
    metadata = {
        "task": spec["task"],
        "name": spec["dataset"]["name"],
        "dataset": spec["dataset"],
        "seed": spec["seed"],
        "cache_hash": _cache_key(spec),
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    return bundle_path


def load_experiment_data(
    spec: dict[str, Any],
    storage: Storage,
    *,
    use_cache: bool = True,
) -> DatasetBundle:
    """Build or load the data bundle for a resolved run specification."""
    cache_path = (
        storage.data_generated / spec["task"] / _cache_key(spec) / "dataset.pt"
    )
    if use_cache and cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    if spec["task"] == "barbell":
        return _load_barbell(spec)
    if spec["task"] == "transfer":
        return _load_transfer(spec)
    if spec["task"] == "cities":
        return _load_city_bundle(spec, storage)
    raise ValueError(f"Unsupported task: {spec['task']}")


def _load_barbell(spec: dict[str, Any]) -> DatasetBundle:
    params = spec["dataset"].get("params", {})
    counts = spec["training"]["samples"]
    train_seed, val_seed, test_seed = _split_seeds(spec["seed"])
    common = {
        "nodes_per_clique": params.get("nodes_per_clique", 10),
        "num_bridge_edges": params.get("num_bridge_edges", 1),
        "feature_dim": params.get("feature_dim", 40),
    }
    train = generate_barbell_dataset(
        **common, samples=counts["train"], seed=train_seed
    )
    val = generate_barbell_dataset(
        **common, samples=counts["val"], seed=val_seed
    )
    test = generate_barbell_dataset(
        **common, samples=counts["test"], seed=test_seed
    )
    return DatasetBundle(
        train=train,
        val=val,
        test=test,
        in_channels=common["feature_dim"],
        out_channels=common["feature_dim"],
        num_layers=int(spec["model"]["num_layers"]),
        task_type="node_regression",
        metric_name="mse",
        metadata=common,
    )


def _load_transfer(spec: dict[str, Any]) -> DatasetBundle:
    dataset = spec["dataset"]
    params = dataset.get("params", {})
    counts = spec["training"]["samples"]
    protocol = dataset.get("protocol", "clean")
    classes = params.get("classes", 5)
    kwargs = {
        "name": dataset["name"],
        "classes": classes,
        "size": params.get("size"),
        "depth": params.get("depth"),
        "arity": params.get("arity", 2),
        "protocol": protocol,
    }
    train_seed, val_seed, test_seed = _split_seeds(spec["seed"])
    train = generate_transfer_dataset(
        **kwargs, samples=counts["train"], seed=train_seed
    )
    if protocol == "legacy":
        test = generate_transfer_dataset(
            **kwargs, samples=counts["test"], seed=test_seed
        )
        val = test
    else:
        val = generate_transfer_dataset(
            **kwargs, samples=counts["val"], seed=val_seed
        )
        test = generate_transfer_dataset(
            **kwargs, samples=counts["test"], seed=test_seed
        )
    distance = int(train[0].source_target_distance.item())
    return DatasetBundle(
        train=train,
        val=val,
        test=test,
        in_channels=classes,
        out_channels=classes,
        num_layers=distance,
        task_type="source_classification",
        metric_name="accuracy",
        metadata={**kwargs, "source_target_distance": distance},
    )


def _load_city_bundle(spec: dict[str, Any], storage: Storage) -> DatasetBundle:
    data: Data = load_city(
        spec["dataset"]["name"],
        raw_root=storage.data_raw,
        processed_root=storage.data_processed,
    )
    return DatasetBundle(
        train=data,
        val=data,
        test=data,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
        num_layers=int(spec["model"]["num_layers"]),
        task_type="node_classification",
        metric_name="accuracy",
        metadata={
            "name": spec["dataset"]["name"],
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
        },
    )

