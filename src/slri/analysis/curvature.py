"""Python 3.12 sidecar interface for Ollivier--Ricci curvature."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from slri.analysis.geometry import GeometrySnapshot


def curvature_payload(
    edge_index: torch.Tensor,
    snapshot: GeometrySnapshot | None,
    *,
    alpha: float = 0.5,
    epsilon: float = 1e-8,
    proc: int = 1,
    shortest_path: str = "all_pairs",
) -> dict[str, Any]:
    """Build the sidecar request with deduplicated non-self-loop edges."""
    edges = sorted(
        {
            (min(int(source), int(target)), max(int(source), int(target)))
            for source, target in edge_index.detach().cpu().t().tolist()
            if source != target
        }
    )
    layers = {}
    if snapshot is not None:
        layers = {
            str(layer): {
                f"{source}:{target}": strength
                for (source, target), strength in strengths.items()
            }
            for layer, strengths in snapshot.strengths_by_layer.items()
        }
    if not layers:
        layers = {
            "-1": {
                f"{source}:{target}": 1.0 for source, target in edges
            }
        }
    return {
        "num_nodes": int(edge_index.max().item()) + 1,
        "edges": edges,
        "strengths_by_layer": layers,
        "alpha": alpha,
        "epsilon": epsilon,
        "method": "OTDSinkhornMix",
        "exp_power": 2,
        "proc": proc,
        "shortest_path": shortest_path,
    }


def run_curvature_sidecar(
    payload: dict[str, Any],
    output: str | Path,
    *,
    project_root: str | Path,
    python: str | Path | None = None,
) -> pd.DataFrame:
    """Execute the pinned GraphRicciCurvature code in its Python 3.12 env."""
    project_root = Path(project_root).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    request = output.with_suffix(".request.json")
    request.write_text(json.dumps(payload, indent=2, sort_keys=True))
    executable = Path(
        python
        or os.environ.get(
            "SLRI_CURVATURE_PYTHON",
            project_root / ".venv-curvature" / "bin" / "python",
        )
    )
    if not executable.exists():
        raise RuntimeError(
            "Curvature environment is missing. Run "
            "scripts/setup_curvature_env.sh first or set "
            "SLRI_CURVATURE_PYTHON."
        )
    command = [
        str(executable),
        str(project_root / "scripts" / "curvature_sidecar.py"),
        "--input",
        str(request),
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            "GraphRicciCurvature sidecar failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return pd.read_csv(output)


def current_python_supports_curvature() -> bool:
    """Return whether this interpreter is intentionally used as the sidecar."""
    return sys.version_info[:2] == (3, 12)
