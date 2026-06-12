import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "run_barbell_gpu.sh",
        "run_transfer_gpu.sh",
        "run_cities_gpu.sh",
        "run_analysis_gpu.sh",
    ],
)
def test_task_launcher_dry_run(script, tmp_path):
    env = {
        **os.environ,
        "SLRI_PYTHON": sys.executable,
        "SLRI_STORAGE_ROOT": str(tmp_path / script),
    }
    result = subprocess.run(
        [
            str(ROOT / "scripts" / script),
            "--profile",
            "smoke",
            "--seeds",
            "0",
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"dry_run": true' in result.stdout


def test_global_launcher_parallel_dry_run(tmp_path):
    env = {
        **os.environ,
        "SLRI_PYTHON": sys.executable,
        "SLRI_STORAGE_ROOT": str(tmp_path / "all"),
    }
    result = subprocess.run(
        [
            str(ROOT / "scripts" / "run_all_gpu.sh"),
            "--gpus",
            "0,1,2",
            "--parallel",
            "--profile",
            "smoke",
            "--seeds",
            "0",
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('"dry_run": true') == 3
