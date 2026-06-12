"""Shared dataset structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch_geometric.data import Data


@dataclass
class DatasetBundle:
    """Data and model dimensions required by one experiment run."""

    train: list[Data] | Data
    val: list[Data] | Data
    test: list[Data] | Data
    in_channels: int
    out_channels: int
    num_layers: int
    task_type: Literal[
        "node_regression",
        "source_classification",
        "node_classification",
    ]
    metric_name: str
    metadata: dict
