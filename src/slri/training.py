"""Task-aware PyTorch training loop for resolved NSD runs."""

from __future__ import annotations

import contextlib
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from slri.datasets import DatasetBundle, load_experiment_data
from slri.models import build_model, forward_model
from slri.storage import Storage


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve `auto`, CUDA, MPS, or CPU device requests."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "32":
        return contextlib.nullcontext()
    if precision == "bf16-mixed":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "16-mixed":
        return torch.autocast("cuda", dtype=torch.float16)
    raise ValueError("precision must be 32, 16-mixed, or bf16-mixed")


def _make_loader(
    data: list[Data],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _forward_loss(
    model: nn.Module,
    batch: Data,
    task_type: str,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = forward_model(
        model,
        batch.x,
        batch.edge_index,
        getattr(batch, "batch", None),
    )
    if task_type == "node_regression":
        target = batch.y
        loss = (logits - target).pow(2).sum(dim=-1).mean()
        return loss, logits, target
    if task_type == "source_classification":
        selected = logits[batch.source_mask]
        target = batch.y.view(-1)
        return F.cross_entropy(selected, target), selected, target
    if task_type == "node_classification":
        mask = getattr(batch, f"{split}_mask")
        selected = logits[mask]
        target = batch.y[mask]
        return F.cross_entropy(selected, target), selected, target
    raise ValueError(f"Unsupported task type: {task_type}")


def _metrics(
    loss: float,
    outputs: list[torch.Tensor],
    targets: list[torch.Tensor],
    task_type: str,
) -> dict[str, float]:
    if task_type == "node_regression":
        return {"loss": loss, "mse": loss}
    logits = torch.cat(outputs, dim=0)
    labels = torch.cat(targets, dim=0)
    predictions = logits.argmax(dim=-1)
    accuracy = predictions.eq(labels).float().mean().item()
    macro_f1 = f1_score(
        labels.numpy(),
        predictions.numpy(),
        average="macro",
        zero_division=0,
    )
    return {"loss": loss, "accuracy": accuracy, "macro_f1": float(macro_f1)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: list[Data] | Data,
    bundle: DatasetBundle,
    *,
    split: str,
    device: torch.device,
    precision: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, float]:
    """Evaluate one split and return loss plus task metrics."""
    model.eval()
    if isinstance(data, Data):
        iterable = [data]
    else:
        iterable = _make_loader(
            data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    total_loss = 0.0
    total_weight = 0
    outputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in iterable:
        batch = batch.to(device)
        with _autocast(device, precision):
            loss, output, target = _forward_loss(
                model, batch, bundle.task_type, split
            )
        weight = int(target.size(0))
        total_loss += float(loss.detach().cpu()) * weight
        total_weight += weight
        outputs.append(output.detach().float().cpu())
        targets.append(target.detach().cpu())
    return _metrics(
        total_loss / max(total_weight, 1),
        outputs,
        targets,
        bundle.task_type,
    )


def _tracking(spec: dict[str, Any], path: Path):
    if not spec.get("tracking", {}).get("wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking requested; install with `uv sync --extra wandb`"
        ) from exc
    return wandb.init(
        project=spec["tracking"].get("project", "sheaf-long-range"),
        entity=spec["tracking"].get("entity"),
        name=spec["run_id"],
        config=spec,
        dir=str(path / "logs"),
        reinit=True,
    )


def run_spec(
    original_spec: dict[str, Any],
    storage: Storage,
    *,
    force: bool = False,
    device_name: str = "auto",
    precision: str = "32",
    use_cache: bool = True,
) -> dict[str, Any]:
    """Train, validate, test, checkpoint, and index one resolved run."""
    if storage.is_completed(original_spec["run_id"]) and not force:
        return {"run_id": original_spec["run_id"], "status": "skipped"}
    if force:
        storage.clear_run(original_spec)

    set_seed(original_spec["seed"])
    bundle = load_experiment_data(original_spec, storage, use_cache=use_cache)
    spec = copy.deepcopy(original_spec)
    spec["model"]["num_layers"] = bundle.num_layers
    spec["dataset_metadata"] = bundle.metadata
    path = storage.begin_run(spec)
    tracker = None
    started = time.perf_counter()

    try:
        device = resolve_device(device_name)
        model = build_model(
            variant=spec["model"]["variant"],
            in_channels=bundle.in_channels,
            out_channels=bundle.out_channels,
            stalk_dim=spec["model"]["stalk_dim"],
            hidden_dim=spec["model"]["hidden_dim"],
            num_layers=bundle.num_layers,
            alpha=spec["model"].get("alpha", 1.0),
            orth_strategy=spec["model"].get("orth_strategy", "cayley"),
            input_dropout=spec["model"].get("input_dropout", 0.0),
            dropout=spec["model"].get("dropout", 0.0),
            normalize_output=spec["model"].get("normalize_output", True),
            seed=spec["seed"],
        ).to(device)
        training = spec["training"]
        optimizer_name = training.get("optimizer", "adamw").lower()
        optimizer_cls = (
            torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
        )
        optimizer = optimizer_cls(
            model.parameters(),
            lr=float(training["lr"]),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=device.type == "cuda" and precision == "16-mixed",
        )
        tracker = _tracking(spec, path)

        if isinstance(bundle.train, Data):
            train_iterable = [bundle.train]
        else:
            train_iterable = _make_loader(
                bundle.train,
                batch_size=int(training.get("batch_size", 1)),
                shuffle=True,
                num_workers=int(training.get("num_workers", 0)),
            )

        monitor = "loss" if bundle.task_type == "node_regression" else "accuracy"
        mode = "min" if monitor == "loss" else "max"
        best = float("inf") if mode == "min" else -float("inf")
        best_epoch = 0
        stale_epochs = 0
        checkpoint = path / "checkpoints" / "best.ckpt"
        initial_checkpoint = path / "checkpoints" / "initial.ckpt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": 0,
                "val_metrics": None,
                "spec": spec,
            },
            initial_checkpoint,
        )

        for epoch in range(1, int(training["epochs"]) + 1):
            model.train()
            epoch_loss = 0.0
            batches = 0
            for batch in train_iterable:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, precision):
                    loss, _, _ = _forward_loss(
                        model, batch, bundle.task_type, "train"
                    )
                scaler.scale(loss).backward()
                if training.get("gradient_clip"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(training["gradient_clip"]),
                    )
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += float(loss.detach().cpu())
                batches += 1

            val_metrics = evaluate(
                model,
                bundle.val,
                bundle,
                split="val",
                device=device,
                precision=precision,
                batch_size=int(training.get("eval_batch_size", 512)),
                num_workers=int(training.get("num_workers", 0)),
            )
            record = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(batches, 1),
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            storage.append_metric(path, record)
            if tracker is not None:
                tracker.log(record, step=epoch)

            score = val_metrics[monitor]
            improved = score < best if mode == "min" else score > best
            if improved:
                best = score
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "spec": spec,
                    },
                    checkpoint,
                )
            else:
                stale_epochs += 1
            if stale_epochs >= int(training.get("patience", training["epochs"])):
                break

        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        test_metrics = evaluate(
            model,
            bundle.test,
            bundle,
            split="test",
            device=device,
            precision=precision,
            batch_size=int(training.get("eval_batch_size", 512)),
            num_workers=int(training.get("num_workers", 0)),
        )
        summary = {
            "run_id": spec["run_id"],
            "status": "completed",
            "metric_name": bundle.metric_name,
            "test_metric": test_metrics[bundle.metric_name],
            "test_metrics": test_metrics,
            "best_epoch": best_epoch,
            "best_val_metric": best,
            "num_layers": bundle.num_layers,
            "parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "runtime_seconds": time.perf_counter() - started,
            "device": str(device),
            "precision": precision,
        }
        storage.complete_run(spec, summary, path)
        if tracker is not None:
            tracker.log({"final": summary})
        return summary
    except BaseException as exc:
        storage.fail_run(spec, path, exc)
        raise
    finally:
        if tracker is not None:
            tracker.finish()


def write_manifest(runs: list[dict[str, Any]], output: Path) -> None:
    """Write a JSONL manifest suitable for schedulers and auditing."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for run in runs:
            handle.write(json.dumps(run, sort_keys=True) + "\n")
