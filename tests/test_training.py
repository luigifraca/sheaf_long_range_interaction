from pathlib import Path

from slri.config import load_config
from slri.grid import expand_grid
from slri.storage import Storage
from slri.training import run_spec

ROOT = Path(__file__).parents[1]


def test_one_epoch_transfer_run_writes_retrievable_artifacts(tmp_path):
    runs = expand_grid(load_config(ROOT / "configs/transfer.yaml", "smoke"))
    spec = next(run for run in runs if run["model"]["variant"] == "identity")
    result = run_spec(
        spec,
        Storage(tmp_path / "storage"),
        device_name="cpu",
        precision="32",
    )
    assert result["status"] == "completed"
    assert 0.0 <= result["test_metric"] <= 1.0
    run_path = (
        tmp_path / "storage" / "runs" / "transfer" / spec["run_id"]
    )
    assert (run_path / "resolved_config.yaml").exists()
    assert (run_path / "provenance.json").exists()
    assert (run_path / "metrics.jsonl").exists()
    assert (run_path / "summary.json").exists()
    assert (run_path / "checkpoints" / "initial.ckpt").exists()
    assert (run_path / "checkpoints" / "best.ckpt").exists()
