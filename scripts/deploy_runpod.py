#!/usr/bin/env python3
"""Deploy the six-worker A40 experiment suite through RunPod's REST API."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

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


def log(message: str) -> None:
    print(message, flush=True)


def is_capacity_error(error: RuntimeError) -> bool:
    return "no instances currently available" in str(error).lower()


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")  # noqa: UP017


def delete_created_pods(api_key: str, pods: list[dict[str, Any]]) -> None:
    for pod in pods:
        pod_id = pod.get("id")
        if not pod_id:
            continue
        try:
            request("DELETE", f"pods/{pod_id}", api_key)
            log(f"deleted partial pod role={pod.get('role')} id={pod_id}")
        except RuntimeError as exc:
            log(f"warning: failed to delete partial pod id={pod_id}: {exc}")


def build_payload(
    *,
    role: str,
    deployment_id: str,
    revision: str,
    image: str,
    gpu_type: str,
    volume: dict[str, Any],
    volume_id: str,
    source_root: str | None,
    wandb_key: str,
    github_token: str | None,
) -> dict[str, Any]:
    if source_root:
        bootstrap = (
            'until test -x "$SLRI_SOURCE_ROOT/scripts/'
            'bootstrap_runpod_worker.sh"; do sleep 10; done && '
            'cp "$SLRI_SOURCE_ROOT/scripts/bootstrap_runpod_worker.sh" '
            "/tmp/bootstrap.sh && "
            "chmod +x /tmp/bootstrap.sh && "
            "/tmp/bootstrap.sh"
        )
    elif github_token:
        bootstrap = (
            "cd /root && "
            'curl -fsSL -H "Authorization: Bearer $SLRI_GITHUB_TOKEN" '
            "https://raw.githubusercontent.com/luigifraca/"
            f"sheaf_long_range_interaction/{revision}/"
            "scripts/bootstrap_runpod_worker.sh "
            "-o /tmp/bootstrap.sh && "
            "chmod +x /tmp/bootstrap.sh && "
            "/tmp/bootstrap.sh"
        )
    else:
        bootstrap = (
            "cd /root && "
            "curl -fsSL "
            "https://raw.githubusercontent.com/luigifraca/"
            f"sheaf_long_range_interaction/{revision}/"
            "scripts/bootstrap_runpod_worker.sh "
            "-o /tmp/bootstrap.sh && "
            "chmod +x /tmp/bootstrap.sh && "
            "/tmp/bootstrap.sh"
        )
    env = {
        "SLRI_WORKER_ROLE": role,
        "SLRI_DEPLOYMENT_ID": deployment_id,
        "SLRI_VOLUME_ROOT": "/workspace/sheaf-lri-storage",
        "SLRI_REVISION": revision,
        "SLRI_SOURCE_ROOT": source_root or "",
        "WANDB_API_KEY": wandb_key,
        "WANDB_PROJECT": "sheaf-long-range-full",
        "PYTHONUNBUFFERED": "1",
    }
    if github_token:
        env["SLRI_GITHUB_TOKEN"] = github_token
        env["SLRI_REPOSITORY"] = (
            "https://x-access-token:"
            f"{quote(github_token, safe='')}"
            "@github.com/luigifraca/sheaf_long_range_interaction.git"
        )

    return {
        "name": f"slri-{role}-{deployment_id}",
        "imageName": image,
        "computeType": "GPU",
        "cloudType": "SECURE",
        "gpuTypeIds": [gpu_type],
        "gpuTypePriority": "custom",
        "gpuCount": 1,
        "dataCenterIds": [volume["dataCenterId"]],
        "dataCenterPriority": "custom",
        "networkVolumeId": volume_id,
        "volumeMountPath": "/workspace",
        "containerDiskInGb": 80,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "interruptible": False,
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [bootstrap],
        "env": env,
    }


def create_deployment_attempt(
    *,
    api_key: str,
    wandb_key: str,
    roles: tuple[str, ...],
    volume: dict[str, Any],
    volume_id: str,
    deployment_id: str,
    revision: str,
    image: str,
    gpu_type: str,
    source_root: str | None,
    github_token: str | None,
    dry_run: bool,
) -> list[dict[str, Any]]:
    pods = []
    for role in roles:
        payload = build_payload(
            role=role,
            deployment_id=deployment_id,
            revision=revision,
            image=image,
            gpu_type=gpu_type,
            volume=volume,
            volume_id=volume_id,
            source_root=source_root,
            wandb_key=wandb_key,
            github_token=github_token,
        )
        if dry_run:
            pod = {"id": None, "name": payload["name"], "payload": payload}
        else:
            log(f"creating role={role} deployment={deployment_id}")
            try:
                pod = request("POST", "pods", api_key, payload)
            except RuntimeError:
                delete_created_pods(api_key, pods)
                raise
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
    return pods


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-id", required=True)
    parser.add_argument("--volume-size", type=int, default=300)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--deployment-id")
    parser.add_argument(
        "--roles",
        default=",".join(ROLES),
        help="Comma-separated worker roles to deploy.",
    )
    parser.add_argument("--gpu-type", default="NVIDIA A40")
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
    parser.add_argument(
        "--public-source",
        action="store_true",
        help="Ignore GitHub tokens and fetch the public GitHub raw bootstrap.",
    )
    parser.add_argument(
        "--wait-for-capacity",
        action="store_true",
        help=(
            "Retry until every requested role can be allocated. Partial "
            "attempts are deleted before sleeping."
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="Seconds between capacity retries when --wait-for-capacity is set.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum capacity attempts; 0 means retry indefinitely.",
    )
    args = parser.parse_args()
    roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
    unknown_roles = sorted(set(roles) - set(ROLES))
    if unknown_roles:
        parser.error(f"unknown roles: {', '.join(unknown_roles)}")

    api_key = os.environ["RUNPOD_API_KEY"]
    wandb_key = os.environ["WANDB_API_KEY"]
    github_token = (
        os.environ.get("SLRI_GITHUB_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
    )
    if args.public_source:
        github_token = None
    volume = request("GET", f"networkvolumes/{args.volume_id}", api_key)
    if int(volume["size"]) < args.volume_size and not args.dry_run:
        volume = request(
            "PATCH",
            f"networkvolumes/{args.volume_id}",
            api_key,
            {"name": "sheaf-lri-storage", "size": args.volume_size},
        )

    attempt = 0
    while True:
        attempt += 1
        deployment_id = args.deployment_id or timestamp_id()
        pods: list[dict[str, Any]] = []
        try:
            log(
                f"attempt={attempt} deployment={deployment_id} "
                f"roles={len(roles)} gpu={args.gpu_type}"
            )
            pods = create_deployment_attempt(
                api_key=api_key,
                wandb_key=wandb_key,
                roles=roles,
                volume=volume,
                volume_id=args.volume_id,
                deployment_id=deployment_id,
                revision=args.revision,
                image=args.image,
                gpu_type=args.gpu_type,
                source_root=args.source_root,
                github_token=github_token,
                dry_run=args.dry_run,
            )
            break
        except RuntimeError as exc:
            delete_created_pods(api_key, pods)
            if not args.wait_for_capacity or not is_capacity_error(exc):
                raise
            if args.max_attempts and attempt >= args.max_attempts:
                raise
            log(
                f"capacity unavailable; retrying in {args.poll_seconds}s "
                f"(attempt {attempt})"
            )
            time.sleep(args.poll_seconds)

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
