#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="0"
PROFILE="benchmark"
STORAGE_ROOT="${SLRI_STORAGE_ROOT:-/workspace/sheaf-lri-storage}"
CHECKPOINTS=""
FORCE=0
DRY_RUN=0

usage() {
  echo "Usage: $0 [--gpu ID] [--profile benchmark|smoke]"
  echo "          [--storage-root PATH] [--checkpoints initial,best]"
  echo "          [--seeds LIST]  # accepted for task-launcher compatibility"
  echo "          [--force] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --storage-root) STORAGE_ROOT="$2"; shift 2 ;;
    --checkpoints) CHECKPOINTS="$2"; shift 2 ;;
    --seeds) shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PYTHON="${SLRI_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python3"
CMD=("$PYTHON" -m slri.cli analyze grid
  --config "$ROOT/configs/analysis.yaml"
  --profile "$PROFILE"
  --storage-root "$STORAGE_ROOT"
  --device cuda:0)
[[ -n "$CHECKPOINTS" ]] && CMD+=(--checkpoints "$CHECKPOINTS")
[[ $FORCE -eq 1 ]] && CMD+=(--force)
[[ $DRY_RUN -eq 1 ]] && CMD+=(--dry-run)

export PYTHONPATH="$ROOT/src:$ROOT/external/sheaf-mpnn/src${PYTHONPATH:+:$PYTHONPATH}"
export SLRI_STORAGE_ROOT="$STORAGE_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "${CMD[@]}"
