"""Command-line interface for experiment execution and retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slri.analysis.runner import (
    analyze_run,
    compare_analyses,
    load_analysis_config,
)
from slri.config import DEFAULT_STORAGE_ROOT, load_config
from slri.datasets import CITY_METADATA, load_city, materialize_dataset
from slri.grid import count_runs_per_setting, expand_grid
from slri.storage import Storage, print_rows
from slri.training import run_spec, write_manifest


def _seeds(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from exc


def _common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", default="benchmark")
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--seeds", type=_seeds)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=["32", "16-mixed", "bf16-mixed"],
        default="32",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")


def _resolved_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = load_config(args.config, args.profile)
    if args.seeds is not None:
        config["seeds"] = args.seeds
    if args.wandb:
        config.setdefault("tracking", {})["wandb"] = True
    if args.wandb_project:
        config.setdefault("tracking", {})["project"] = args.wandb_project
    if args.wandb_entity:
        config.setdefault("tracking", {})["entity"] = args.wandb_entity
    return expand_grid(config)


def _cmd_grid(args: argparse.Namespace) -> int:
    runs = _resolved_runs(args)
    storage = Storage(args.storage_root)
    manifest = (
        storage.summaries_dir
        / f"{runs[0]['task']}-{args.profile}-manifest.jsonl"
    )
    write_manifest(runs, manifest)
    counts = count_runs_per_setting(runs)
    print(
        json.dumps(
            {
                "task": runs[0]["task"],
                "profile": args.profile,
                "runs": len(runs),
                "runs_per_setting": sorted(set(counts.values())),
                "manifest": str(manifest),
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    if args.dry_run:
        return 0
    failures = 0
    for index, spec in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] {spec['run_id']}", flush=True)
        try:
            result = run_spec(
                spec,
                storage,
                force=args.force,
                device_name=args.device,
                precision=args.precision,
            )
            print(json.dumps(result, sort_keys=True))
        except Exception as exc:
            failures += 1
            print(f"FAILED {spec['run_id']}: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise
    return 1 if failures else 0


def _cmd_run(args: argparse.Namespace) -> int:
    runs = _resolved_runs(args)
    selected = [run for run in runs if run["run_id"] == args.run_id]
    if not selected:
        raise SystemExit(f"Run ID {args.run_id!r} is not in the resolved grid")
    result = run_spec(
        selected[0],
        Storage(args.storage_root),
        force=args.force,
        device_name=args.device,
        precision=args.precision,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_data_prepare(args: argparse.Namespace) -> int:
    storage = Storage(args.storage_root)
    if args.dataset:
        names = list(CITY_METADATA) if args.dataset == "all" else [args.dataset]
        for name in names:
            data = load_city(
                name,
                raw_root=storage.data_raw,
                processed_root=storage.data_processed,
            )
            print(
                json.dumps(
                    {
                        "name": name,
                        "nodes": int(data.num_nodes),
                        "edges": int(data.num_edges),
                    }
                )
            )
        return 0
    if args.config:
        config = load_config(args.config, args.profile)
        if args.seeds is not None:
            config["seeds"] = args.seeds
        unique: dict[str, dict[str, Any]] = {}
        for spec in expand_grid(config):
            key = json.dumps(
                [spec["task"], spec["dataset"], spec["seed"]],
                sort_keys=True,
            )
            unique.setdefault(key, spec)
        for spec in unique.values():
            print(materialize_dataset(spec, storage, force=args.force))
        return 0
    raise SystemExit("Provide --dataset or --config")


def _cmd_data_list(args: argparse.Namespace) -> int:
    print_rows(Storage(args.storage_root).list_data())
    return 0


def _cmd_data_describe(args: argparse.Namespace) -> int:
    records = [
        record
        for record in Storage(args.storage_root).list_data()
        if record.get("name") == args.name
    ]
    if not records and args.name in CITY_METADATA:
        records = [{"name": args.name, **CITY_METADATA[args.name]}]
    if not records:
        raise SystemExit(f"No stored or built-in metadata for {args.name!r}")
    print_rows(records)
    return 0


def _query_from_args(args: argparse.Namespace) -> str | None:
    filters = []
    if args.query:
        filters.append(args.query)
    for field in ("task", "dataset", "variant", "status"):
        value = getattr(args, field, None)
        if value:
            filters.append(f"{field}={value}")
    return ",".join(filters) or None


def _cmd_runs_list(args: argparse.Namespace) -> int:
    print_rows(Storage(args.storage_root).query(_query_from_args(args)))
    return 0


def _cmd_runs_show(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            Storage(args.storage_root).show(args.run_id),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_runs_export(args: argparse.Namespace) -> int:
    print(Storage(args.storage_root).export(args.selector, args.output))
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    print(
        Storage(args.storage_root).write_summary_csv(
            args.query,
            args.output,
        )
    )
    return 0


def _cmd_analyze_run(args: argparse.Namespace) -> int:
    storage = Storage(args.storage_root)
    config = load_analysis_config(args.config, args.profile)
    failures = 0
    for checkpoint in args.checkpoints.split(","):
        checkpoint = checkpoint.strip()
        if not checkpoint:
            continue
        try:
            result = analyze_run(
                args.run_id,
                storage,
                checkpoint=checkpoint,
                config=config,
                device_name=args.device,
                force=args.force,
            )
            print(json.dumps(result, sort_keys=True))
        except Exception as exc:
            failures += 1
            print(
                f"FAILED {args.run_id} ({checkpoint}): {exc}",
                file=sys.stderr,
            )
            if args.fail_fast:
                raise
    return 1 if failures else 0


def _cmd_analyze_grid(args: argparse.Namespace) -> int:
    storage = Storage(args.storage_root)
    config = load_analysis_config(args.config, args.profile)
    queries = config.get("queries", ["status=completed"])
    checkpoints = args.checkpoints or ",".join(
        config.get("checkpoints", ["best"])
    )
    runs: dict[str, dict[str, Any]] = {}
    for query in queries:
        for row in storage.query(query):
            runs[row["run_id"]] = row
    manifest = {
        "runs": sorted(runs),
        "checkpoints": checkpoints.split(","),
        "profile": args.profile,
        "dry_run": args.dry_run,
    }
    print(json.dumps(manifest, sort_keys=True))
    if args.dry_run:
        return 0
    failures = 0
    for run_id in sorted(runs):
        for checkpoint in checkpoints.split(","):
            try:
                result = analyze_run(
                    run_id,
                    storage,
                    checkpoint=checkpoint.strip(),
                    config=config,
                    device_name=args.device,
                    force=args.force,
                )
                print(json.dumps(result, sort_keys=True))
            except Exception as exc:
                failures += 1
                print(
                    f"FAILED {run_id} ({checkpoint}): {exc}",
                    file=sys.stderr,
                )
                if args.fail_fast:
                    raise
    return 1 if failures else 0


def _cmd_analyze_compare(args: argparse.Namespace) -> int:
    print(
        compare_analyses(
            Storage(args.storage_root),
            args.query,
            args.output,
        )
    )
    return 0


def _cmd_analyses_list(args: argparse.Namespace) -> int:
    print_rows(Storage(args.storage_root).query_analyses(args.query))
    return 0


def _cmd_analyses_show(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            Storage(args.storage_root).show_analysis(args.analysis_id),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_analyses_files(args: argparse.Namespace) -> int:
    print_rows(Storage(args.storage_root).analysis_files(args.analysis_id))
    return 0


def _cmd_analyses_export(args: argparse.Namespace) -> int:
    print(
        Storage(args.storage_root).export_analyses(
            args.selector, args.output
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slri")
    subparsers = parser.add_subparsers(dest="command", required=True)

    grid = subparsers.add_parser("grid", help="Run or inspect a complete grid")
    _common_run_args(grid)
    grid.add_argument("--dry-run", action="store_true")
    grid.add_argument("--fail-fast", action="store_true")
    grid.set_defaults(func=_cmd_grid)

    run = subparsers.add_parser("run", help="Run one ID from a resolved grid")
    _common_run_args(run)
    run.add_argument("--run-id", required=True)
    run.set_defaults(func=_cmd_run)

    data = subparsers.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    prepare = data_sub.add_parser("prepare")
    prepare.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    prepare.add_argument("--dataset", choices=["paris", "shanghai", "all"])
    prepare.add_argument("--config", type=Path)
    prepare.add_argument("--profile", default="benchmark")
    prepare.add_argument("--seeds", type=_seeds)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=_cmd_data_prepare)
    data_list = data_sub.add_parser("list")
    data_list.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    data_list.set_defaults(func=_cmd_data_list)
    describe = data_sub.add_parser("describe")
    describe.add_argument("name")
    describe.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    describe.set_defaults(func=_cmd_data_describe)

    runs = subparsers.add_parser("runs")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_sub.add_parser("list")
    runs_list.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    runs_list.add_argument("--query")
    runs_list.add_argument("--task")
    runs_list.add_argument("--dataset")
    runs_list.add_argument("--variant")
    runs_list.add_argument("--status")
    runs_list.set_defaults(func=_cmd_runs_list)
    show = runs_sub.add_parser("show")
    show.add_argument("run_id")
    show.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    show.set_defaults(func=_cmd_runs_show)
    export = runs_sub.add_parser("export")
    export.add_argument("selector")
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    export.set_defaults(func=_cmd_runs_export)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--query")
    summarize.add_argument("--output", required=True, type=Path)
    summarize.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    summarize.set_defaults(func=_cmd_summarize)

    analyze = subparsers.add_parser("analyze")
    analyze_sub = analyze.add_subparsers(dest="analyze_command", required=True)
    analyze_run_parser = analyze_sub.add_parser("run")
    analyze_run_parser.add_argument("run_id")
    analyze_run_parser.add_argument(
        "--config", type=Path, default=Path("configs/analysis.yaml")
    )
    analyze_run_parser.add_argument("--profile", default="benchmark")
    analyze_run_parser.add_argument("--checkpoints", default="best")
    analyze_run_parser.add_argument("--device", default="auto")
    analyze_run_parser.add_argument(
        "--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT
    )
    analyze_run_parser.add_argument("--force", action="store_true")
    analyze_run_parser.add_argument("--fail-fast", action="store_true")
    analyze_run_parser.set_defaults(func=_cmd_analyze_run)

    analyze_grid = analyze_sub.add_parser("grid")
    analyze_grid.add_argument(
        "--config", type=Path, default=Path("configs/analysis.yaml")
    )
    analyze_grid.add_argument("--profile", default="benchmark")
    analyze_grid.add_argument("--checkpoints")
    analyze_grid.add_argument("--device", default="auto")
    analyze_grid.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    analyze_grid.add_argument("--force", action="store_true")
    analyze_grid.add_argument("--fail-fast", action="store_true")
    analyze_grid.add_argument("--dry-run", action="store_true")
    analyze_grid.set_defaults(func=_cmd_analyze_grid)

    analyze_compare = analyze_sub.add_parser("compare")
    analyze_compare.add_argument("--query")
    analyze_compare.add_argument("--output", required=True, type=Path)
    analyze_compare.add_argument(
        "--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT
    )
    analyze_compare.set_defaults(func=_cmd_analyze_compare)

    analyses = subparsers.add_parser("analyses")
    analyses_sub = analyses.add_subparsers(
        dest="analyses_command", required=True
    )
    analyses_list = analyses_sub.add_parser("list")
    analyses_list.add_argument("--query")
    analyses_list.add_argument(
        "--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT
    )
    analyses_list.set_defaults(func=_cmd_analyses_list)
    analyses_show = analyses_sub.add_parser("show")
    analyses_show.add_argument("analysis_id")
    analyses_show.add_argument(
        "--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT
    )
    analyses_show.set_defaults(func=_cmd_analyses_show)
    analyses_files = analyses_sub.add_parser("files")
    analyses_files.add_argument("analysis_id")
    analyses_files.add_argument(
        "--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT
    )
    analyses_files.set_defaults(func=_cmd_analyses_files)
    analyses_export = analyses_sub.add_parser("export")
    analyses_export.add_argument("selector")
    analyses_export.add_argument("--output", required=True, type=Path)
    analyses_export.add_argument(
        "--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT
    )
    analyses_export.set_defaults(func=_cmd_analyses_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
