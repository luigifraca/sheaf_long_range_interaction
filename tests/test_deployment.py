import sys
from argparse import Namespace
from types import SimpleNamespace

from slri.cli import _resolved_runs
from slri.training import _tracking


def _args(**overrides):
    values = {
        "config": "configs/transfer.yaml",
        "profile": "analysis",
        "seeds": [0, 1, 2],
        "wandb": False,
        "wandb_project": None,
        "wandb_entity": None,
        "dataset": None,
        "setting": None,
        "shard_count": 1,
        "shard_index": 0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_transfer_setting_shards_are_disjoint_and_complete():
    full = _resolved_runs(_args())
    shards = [
        _resolved_runs(_args(shard_count=3, shard_index=index))
        for index in range(3)
    ]
    full_ids = {run["run_id"] for run in full}
    shard_ids = [{run["run_id"] for run in shard} for shard in shards]
    assert [len(shard) for shard in shards] == [1104, 966, 966]
    assert set.union(*shard_ids) == full_ids
    assert not (shard_ids[0] & shard_ids[1])
    assert not (shard_ids[0] & shard_ids[2])
    assert not (shard_ids[1] & shard_ids[2])


def test_dataset_selector_is_applied_before_sharding():
    runs = _resolved_runs(
        _args(
            config="configs/cities.yaml",
            dataset=["paris"],
            shard_count=1,
        )
    )
    assert len(runs) == 138
    assert {run["dataset"]["name"] for run in runs} == {"paris"}


def test_wandb_run_name_is_readable(monkeypatch, tmp_path):
    captured = {}

    def init(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    spec = _resolved_runs(
        _args(
            dataset=["ring"],
            setting=["size=20"],
            shard_count=1,
        )
    )[0]
    spec["tracking"]["wandb"] = True
    _tracking(spec, tmp_path)
    assert captured["name"].startswith(
        "transfer-ring-size20-analysis-representative-"
    )
    assert captured["id"] == spec["run_id"]
    assert captured["group"] == "transfer/ring/size20/analysis"
