from pathlib import Path

from slri.config import load_config
from slri.grid import config_hash, count_runs_per_setting, expand_grid

ROOT = Path(__file__).parents[1]


def test_barbell_grid_has_72_seeded_runs_per_setting():
    runs = expand_grid(load_config(ROOT / "configs/barbell.yaml", "benchmark"))
    assert len(runs) == 72
    assert set(count_runs_per_setting(runs).values()) == {72}


def test_transfer_grid_uses_only_requested_sizes():
    runs = expand_grid(load_config(ROOT / "configs/transfer.yaml", "benchmark"))
    counts = count_runs_per_setting(runs)
    assert len(counts) == 22
    assert set(counts.values()) == {72}
    sizes = {
        run["dataset"]["params"]["size"]
        for run in runs
        if run["dataset"]["name"] != "tree"
    }
    assert sizes == {2, 6, 10, 20, 30}
    depths = {
        run["dataset"]["params"]["depth"]
        for run in runs
        if run["dataset"]["name"] == "tree"
    }
    assert depths == {2, 3, 4, 5, 6, 7, 8}


def test_smoke_transfer_is_one_setting_and_all_architectures():
    runs = expand_grid(load_config(ROOT / "configs/transfer.yaml", "smoke"))
    assert len(runs) == 72
    assert {run["model"]["variant"] for run in runs} == {
        "general",
        "orthogonal",
        "diagonal",
        "identity",
    }


def test_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
