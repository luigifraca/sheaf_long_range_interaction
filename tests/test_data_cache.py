from pathlib import Path

from slri.config import load_config
from slri.datasets import materialize_dataset
from slri.grid import expand_grid
from slri.storage import Storage

ROOT = Path(__file__).parents[1]


def test_materialized_cache_depends_on_data_not_model(tmp_path):
    runs = expand_grid(load_config(ROOT / "configs/transfer.yaml", "smoke"))
    first = runs[0]
    different_model = next(
        run
        for run in runs
        if run["seed"] == first["seed"]
        and run["model"]["variant"] != first["model"]["variant"]
    )
    storage = Storage(tmp_path / "storage")
    first_path = materialize_dataset(first, storage)
    second_path = materialize_dataset(different_model, storage)
    assert first_path == second_path
    assert first_path.exists()
    assert (first_path.parent / "metadata.json").exists()
