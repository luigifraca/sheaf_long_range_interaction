#!/usr/bin/env python3
"""Deploy the six-worker A40 experiment suite through RunPod's REST API."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API = "https://rest.runpod.io/v1"
ROLES = (
    "barbell",
    "transfer-0",
    "transfer-1",
    "transfer-2",
    "cities-paris",
    "cities-shanghai",
)


def request(
    method: str,
    path: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{API}/{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            if response.status == 204:
                return None
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        raise RuntimeError(f"RunPod {method} {path}: {exc.code} {body}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-id", required=True)
    parser.add_argument("--volume-size", type=int, default=300)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--deployment-id")
    parser.add_argument(
        "--source-root",
        help=(
            "Preloaded source tree on the network volume "
            "(recommended for private repos)."
        ),
    )
    parser.add_argument(
        "--image",
        default="runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404",
    )
    parser.add_argument("--output", type=Path, default=Path(".runpod-deployment.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ["RUNPOD_API_KEY"]
    wandb_key = os.environ["WANDB_API_KEY"]
    deployment_id = args.deployment_id or datetime.now(
        timezone.utc  # noqa: UP017
    ).strftime(
        "%Y%m%d-%H%M%S"
    )
    volume = request("GET", f"networkvolumes/{args.volume_id}", api_key)
    if int(volume["size"]) < args.volume_size and not args.dry_run:
        volume = request(
            "PATCH",
            f"networkvolumes/{args.volume_id}",
            api_key,
            {"name": "sheaf-lri-storage", "size": args.volume_size},
        )

    pods = []
    for role in ROLES:
        if args.source_root:
            bootstrap = (
                'cp "$SLRI_SOURCE_ROOT/scripts/bootstrap_runpod_worker.sh" '
                "/tmp/bootstrap.sh && "
                "chmod +x /tmp/bootstrap.sh && "
                "/tmp/bootstrap.sh"
            )
        else:
            bootstrap = (
                "cd /root && "
                "curl -fsSL "
                "https://raw.githubusercontent.com/luigifraca/"
                f"sheaf_long_range_interaction/{args.revision}/"
                "scripts/bootstrap_runpod_worker.sh "
                "-o /tmp/bootstrap.sh && "
                "chmod +x /tmp/bootstrap.sh && "
                "/tmp/bootstrap.sh"
            )
        payload = {
            "name": f"slri-{role}-{deployment_id}",
            "imageName": args.image,
            "computeType": "GPU",
            "cloudType": "SECURE",
            "gpuTypeIds": ["NVIDIA A40"],
            "gpuTypePriority": "custom",
            "gpuCount": 1,
            "dataCenterIds": [volume["dataCenterId"]],
            "dataCenterPriority": "custom",
            "networkVolumeId": args.volume_id,
            "volumeMountPath": "/workspace",
            "containerDiskInGb": 80,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "interruptible": False,
            "dockerEntrypoint": ["bash", "-lc"],
            "dockerStartCmd": [bootstrap],
            "env": {
                "SLRI_WORKER_ROLE": role,
                "SLRI_DEPLOYMENT_ID": deployment_id,
                "SLRI_VOLUME_ROOT": "/workspace/sheaf-lri-storage",
                "SLRI_REVISION": args.revision,
                "SLRI_SOURCE_ROOT": args.source_root or "",
                "WANDB_API_KEY": wandb_key,
                "WANDB_PROJECT": "sheaf-long-range-full",
                "PYTHONUNBUFFERED": "1",
            },
        }
        if args.dry_run:
            pod = {"id": None, "name": payload["name"], "payload": payload}
        else:
            pod = request("POST", "pods", api_key, payload)
        pods.append(
            {
                "id": pod.get("id"),
                "name": pod.get("name", payload["name"]),
                "role": role,
                "cost_per_hour": pod.get("costPerHr"),
                "gpu": (pod.get("gpu") or {}).get("displayName"),
                "desired_status": pod.get("desiredStatus"),
            }
        )

    record = {
        "deployment_id": deployment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "revision": args.revision,
        "volume": volume,
        "pods": pods,
    }
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
