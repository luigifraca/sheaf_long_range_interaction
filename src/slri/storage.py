"""Persistent experiment artifacts and SQLite indexing."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import socket
import sqlite3
import subprocess
import sys
import tarfile
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL,
    task TEXT NOT NULL,
    dataset TEXT NOT NULL,
    setting TEXT NOT NULL,
    variant TEXT NOT NULL,
    stalk_dim INTEGER NOT NULL,
    hidden_dim INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    metric_name TEXT,
    metric_value REAL,
    artifact_path TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_query
ON runs(task, dataset, variant, status);
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    task TEXT NOT NULL,
    dataset TEXT NOT NULL,
    variant TEXT NOT NULL,
    checkpoint TEXT NOT NULL,
    status TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_analyses_query
ON analyses(run_id, task, dataset, variant, checkpoint, status);
"""

QUERY_FIELDS = {
    "run_id",
    "task",
    "dataset",
    "setting",
    "variant",
    "stalk_dim",
    "hidden_dim",
    "seed",
    "status",
}

ANALYSIS_QUERY_FIELDS = {
    "analysis_id",
    "run_id",
    "task",
    "dataset",
    "variant",
    "checkpoint",
    "status",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def parse_query(query: str | None) -> dict[str, str]:
    """Parse comma-separated `field=value` filters."""
    if not query:
        return {}
    result: dict[str, str] = {}
    for part in query.split(","):
        if "=" not in part:
            raise ValueError("Queries use comma-separated field=value filters")
        key, value = (item.strip() for item in part.split("=", 1))
        if key not in QUERY_FIELDS:
            raise ValueError(f"Unsupported query field {key!r}")
        result[key] = value
    return result


def parse_analysis_query(query: str | None) -> dict[str, str]:
    """Parse filters accepted by the analyses index."""
    if not query:
        return {}
    result: dict[str, str] = {}
    for part in query.split(","):
        if "=" not in part:
            raise ValueError("Queries use comma-separated field=value filters")
        key, value = (item.strip() for item in part.split("=", 1))
        if key not in ANALYSIS_QUERY_FIELDS:
            raise ValueError(f"Unsupported analysis query field {key!r}")
        result[key] = value
    return result


class Storage:
    """Manage the `SLRI_STORAGE_ROOT` directory and run index."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.data_raw = self.root / "data" / "raw"
        self.data_processed = self.root / "data" / "processed"
        self.data_generated = self.root / "data" / "generated"
        self.runs_dir = self.root / "runs"
        self.summaries_dir = self.root / "summaries"
        self.db_path = self.root / "runs.sqlite"
        for path in (
            self.data_raw,
            self.data_processed,
            self.data_generated,
            self.runs_dir,
            self.summaries_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _initialize_database(self) -> None:
        for attempt in range(10):
            try:
                with self.connect() as con:
                    con.execute("PRAGMA journal_mode=WAL")
                    con.executescript(SCHEMA)
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 9:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30.0)
        con.execute("PRAGMA busy_timeout=30000")
        con.row_factory = sqlite3.Row
        return con

    def run_dir(self, spec: dict[str, Any]) -> Path:
        path = self.runs_dir / spec["task"] / spec["run_id"]
        path.mkdir(parents=True, exist_ok=True)
        (path / "checkpoints").mkdir(exist_ok=True)
        (path / "logs").mkdir(exist_ok=True)
        return path

    def is_completed(self, run_id: str) -> bool:
        with self.connect() as con:
            row = con.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return bool(row and row["status"] == "completed")

    def begin_run(self, spec: dict[str, Any]) -> Path:
        path = self.run_dir(spec)
        (path / "resolved_config.yaml").write_text(
            yaml.safe_dump(spec, sort_keys=False)
        )
        (path / "provenance.json").write_text(
            json.dumps(self.provenance(spec), indent=2, sort_keys=True)
        )
        self.upsert(spec, status="running", artifact_path=path)
        return path

    def append_metric(self, path: Path, metric: dict[str, Any]) -> None:
        with (path / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metric, sort_keys=True) + "\n")

    def complete_run(
        self,
        spec: dict[str, Any],
        summary: dict[str, Any],
        path: Path,
    ) -> None:
        (path / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        self.upsert(
            spec,
            status="completed",
            artifact_path=path,
            metric_name=summary.get("metric_name"),
            metric_value=summary.get("test_metric"),
        )

    def fail_run(self, spec: dict[str, Any], path: Path, error: BaseException) -> None:
        (path / "logs" / "error.txt").write_text(f"{type(error).__name__}: {error}\n")
        self.upsert(spec, status="failed", artifact_path=path)

    def upsert(
        self,
        spec: dict[str, Any],
        *,
        status: str,
        artifact_path: Path,
        metric_name: str | None = None,
        metric_value: float | None = None,
    ) -> None:
        dataset = spec["dataset"]
        model = spec["model"]
        values = (
            spec["run_id"],
            spec["config_hash"],
            spec["task"],
            dataset["name"],
            dataset.get("setting", "default"),
            model["variant"],
            model["stalk_dim"],
            model["hidden_dim"],
            spec["seed"],
            status,
            metric_name,
            metric_value,
            str(artifact_path),
            _now(),
        )
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    metric_name=excluded.metric_name,
                    metric_value=excluded.metric_value,
                    artifact_path=excluded.artifact_path,
                    updated_at=excluded.updated_at
                """,
                values,
            )

    def query(self, query: str | None = None) -> list[dict[str, Any]]:
        filters = parse_query(query)
        clauses: list[str] = []
        values: list[str] = []
        for key, value in filters.items():
            clauses.append(f"{key} = ?")
            values.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as con:
            rows = con.execute(
                f"SELECT * FROM runs{where} ORDER BY updated_at DESC",  # noqa: S608
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def show(self, run_id: str) -> dict[str, Any]:
        rows = self.query(f"run_id={run_id}")
        if not rows:
            raise KeyError(f"Unknown run ID: {run_id}")
        row = rows[0]
        summary_path = Path(row["artifact_path"]) / "summary.json"
        row["summary"] = (
            json.loads(summary_path.read_text()) if summary_path.exists() else None
        )
        row["analyses"] = self.query_analyses(f"run_id={run_id}")
        return row

    def analysis_dir(self, run_id: str, analysis_id: str) -> Path:
        """Return and initialize one analysis artifact directory."""
        rows = self.query(f"run_id={run_id}")
        if not rows:
            raise KeyError(f"Unknown run ID: {run_id}")
        path = Path(rows[0]["artifact_path"]) / "analysis" / analysis_id
        for child in ("tables", "matrices", "figures", "logs"):
            (path / child).mkdir(parents=True, exist_ok=True)
        return path

    def begin_analysis(
        self,
        record: dict[str, Any],
        resolved: dict[str, Any],
    ) -> Path:
        """Create files and index a running analysis."""
        path = self.analysis_dir(record["run_id"], record["analysis_id"])
        (path / "resolved_analysis.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False)
        )
        (path / "provenance.json").write_text(
            json.dumps(self.provenance(resolved), indent=2, sort_keys=True)
        )
        self.upsert_analysis(record, status="running", artifact_path=path)
        return path

    def upsert_analysis(
        self,
        record: dict[str, Any],
        *,
        status: str,
        artifact_path: Path,
    ) -> None:
        """Insert or update one analysis index record."""
        values = (
            record["analysis_id"],
            record["run_id"],
            record["config_hash"],
            record["task"],
            record["dataset"],
            record["variant"],
            record["checkpoint"],
            status,
            str(artifact_path),
            _now(),
        )
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(analysis_id) DO UPDATE SET
                    status=excluded.status,
                    artifact_path=excluded.artifact_path,
                    updated_at=excluded.updated_at
                """,
                values,
            )

    def complete_analysis(
        self,
        record: dict[str, Any],
        summary: dict[str, Any],
        path: Path,
    ) -> None:
        """Write an analysis summary and mark it completed."""
        (path / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        self.upsert_analysis(record, status="completed", artifact_path=path)

    def fail_analysis(
        self,
        record: dict[str, Any],
        path: Path,
        error: BaseException,
    ) -> None:
        """Retain a failed analysis and its error."""
        (path / "logs" / "error.txt").write_text(
            f"{type(error).__name__}: {error}\n"
        )
        self.upsert_analysis(record, status="failed", artifact_path=path)

    def query_analyses(self, query: str | None = None) -> list[dict[str, Any]]:
        """Query derived analyses using comma-separated equality filters."""
        filters = parse_analysis_query(query)
        clauses: list[str] = []
        values: list[str] = []
        for key, value in filters.items():
            clauses.append(f"{key} = ?")
            values.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as con:
            rows = con.execute(
                f"SELECT * FROM analyses{where} ORDER BY updated_at DESC",  # noqa: S608
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def show_analysis(self, analysis_id: str) -> dict[str, Any]:
        """Return one analysis record with its summary."""
        rows = self.query_analyses(f"analysis_id={analysis_id}")
        if not rows:
            raise KeyError(f"Unknown analysis ID: {analysis_id}")
        row = rows[0]
        summary_path = Path(row["artifact_path"]) / "summary.json"
        row["summary"] = (
            json.loads(summary_path.read_text()) if summary_path.exists() else None
        )
        return row

    def analysis_files(self, analysis_id: str) -> list[dict[str, Any]]:
        """List all source-of-truth files for one analysis."""
        row = self.show_analysis(analysis_id)
        root = Path(row["artifact_path"])
        return [
            {
                "analysis_id": analysis_id,
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]

    def export_analyses(self, selector: str, output: str | Path) -> Path:
        """Export one analysis ID or a filtered set of analyses."""
        filters = (
            selector if "=" in selector else f"analysis_id={selector}"
        )
        rows = self.query_analyses(filters)
        if not rows:
            raise KeyError(f"No analyses match {selector!r}")
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output_path, "w:gz") as archive:
            manifest = json.dumps(rows, indent=2, sort_keys=True).encode()
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest)
            info.mtime = int(time.time())
            import io

            archive.addfile(info, io.BytesIO(manifest))
            for row in rows:
                path = Path(row["artifact_path"])
                archive.add(
                    path,
                    arcname=f"analyses/{row['analysis_id']}",
                )
        return output_path

    def export(self, selector: str, output: str | Path) -> Path:
        filters = selector if "=" in selector else f"run_id={selector}"
        rows = self.query(filters)
        if not rows:
            raise KeyError(f"No runs match {selector!r}")
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = output_path.with_suffix(".manifest.json")
        manifest.write_text(json.dumps(rows, indent=2, sort_keys=True))
        with tarfile.open(output_path, "w:gz") as archive:
            archive.add(manifest, arcname="manifest.json")
            for row in rows:
                path = Path(row["artifact_path"])
                archive.add(path, arcname=f"runs/{path.name}")
        manifest.unlink()
        return output_path

    def write_summary_csv(
        self, query: str | None, output: str | Path
    ) -> Path:
        rows = self.query(query)
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "run_id",
            "task",
            "dataset",
            "setting",
            "variant",
            "stalk_dim",
            "hidden_dim",
            "seed",
            "status",
            "metric_name",
            "metric_value",
            "artifact_path",
        ]
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})
        return output_path

    def list_data(self) -> list[dict[str, Any]]:
        records = []
        for metadata in self.root.glob("data/**/metadata.json"):
            payload = json.loads(metadata.read_text())
            payload["path"] = str(metadata.parent)
            records.append(payload)
        return sorted(records, key=lambda item: (item.get("task", ""), item["name"]))

    def clear_run(self, spec: dict[str, Any]) -> None:
        path = self.runs_dir / spec["task"] / spec["run_id"]
        if path.exists():
            shutil.rmtree(path)
        with self.connect() as con:
            con.execute("DELETE FROM runs WHERE run_id = ?", (spec["run_id"],))

    @staticmethod
    def provenance(spec: dict[str, Any]) -> dict[str, Any]:
        packages = {}
        for package in (
            "torch",
            "torch-geometric",
            "lightning",
            "sheaf-mpnn",
            "pandas",
            "pyarrow",
        ):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = "not-installed"
        return {
            "created_at": _now(),
            "config_hash": spec["config_hash"],
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "packages": packages,
            "environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "SLRI_STORAGE_ROOT": os.environ.get("SLRI_STORAGE_ROOT"),
            },
            "git": {
                "project": _git_sha("."),
                "sheaf_mpnn": _git_sha("external/sheaf-mpnn"),
                "graph_ricci_curvature": _git_sha(
                    "external/GraphRicciCurvature"
                ),
            },
        }


def _git_sha(path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def print_rows(rows: Iterable[dict[str, Any]]) -> None:
    """Print records as compact JSON, one per line."""
    for row in rows:
        print(json.dumps(row, sort_keys=True, default=str))
