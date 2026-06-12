import json
import tarfile

from slri.storage import Storage


def fake_spec():
    return {
        "run_id": "barbell-test",
        "config_hash": "abc123",
        "task": "barbell",
        "dataset": {"name": "barbell", "setting": "default"},
        "model": {
            "variant": "identity",
            "stalk_dim": 2,
            "hidden_dim": 16,
        },
        "seed": 0,
    }


def test_storage_index_query_export_and_summary(tmp_path):
    storage = Storage(tmp_path / "store")
    spec = fake_spec()
    path = storage.begin_run(spec)
    storage.append_metric(path, {"epoch": 1, "val_mse": 1.0})
    storage.complete_run(
        spec,
        {
            "metric_name": "mse",
            "test_metric": 0.5,
            "status": "completed",
        },
        path,
    )

    assert storage.is_completed(spec["run_id"])
    rows = storage.query("task=barbell,variant=identity")
    assert len(rows) == 1
    assert rows[0]["metric_value"] == 0.5
    assert storage.show(spec["run_id"])["summary"]["test_metric"] == 0.5

    archive = storage.export(spec["run_id"], tmp_path / "run.tar.gz")
    with tarfile.open(archive) as handle:
        assert "manifest.json" in handle.getnames()
        assert any(name.endswith("summary.json") for name in handle.getnames())

    csv_path = storage.write_summary_csv(
        "status=completed", tmp_path / "summary.csv"
    )
    assert "barbell-test" in csv_path.read_text()
    assert json.loads((path / "summary.json").read_text())["test_metric"] == 0.5

