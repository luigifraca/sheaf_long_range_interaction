"""Paris and Shanghai City-Network loaders."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import CityNetwork

CITY_METADATA = {
    "paris": {
        "nodes": 114_127,
        "edges": 182_511,
        "features": 37,
        "classes": 10,
        "split": "10%/10%/80%",
    },
    "shanghai": {
        "nodes": 183_917,
        "edges": 262_092,
        "features": 37,
        "classes": 10,
        "split": "10%/10%/80%",
    },
}


def load_city(
    name: str,
    *,
    raw_root: str | Path,
    processed_root: str | Path,
) -> Data:
    """Load one city, retaining a project-owned processed cache."""
    name = name.lower()
    if name not in CITY_METADATA:
        raise ValueError(f"Unsupported city {name!r}")

    processed_dir = Path(processed_root) / "city_networks" / name
    processed_path = processed_dir / "data.pt"
    if processed_path.exists():
        return torch.load(processed_path, weights_only=False)

    dataset = CityNetwork(
        root=str(Path(raw_root) / "city_networks"),
        name=name,
        augmented=True,
    )
    data = dataset[0]
    processed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, processed_path)
    metadata = {"task": "cities", "name": name, **CITY_METADATA[name]}
    (processed_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    return data

