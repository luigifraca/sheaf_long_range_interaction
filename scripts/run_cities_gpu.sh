#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="0"
PROFILE="benchmark"
SEEDS="0,1,2"
STORAGE_ROOT="${SLRI_STORAGE_ROOT:-/workspace/sheaf-lri-storage}"
PRECISION="32"
FORCE=0
DRY_RUN=0
WANDB=0
WANDB_PROJECT=""
WANDB_ENTITY=""

usage() {
  echo "Usage: $0 [--gpu ID] [--profile benchmark|smoke] [--seeds 0,1,2]"
  echo "          [--storage-root PATH] [--precision bf16-mixed|16-mixed|32]"
  echo "          [--force] [--dry-run] [--wandb]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --storage-root) STORAGE_ROOT="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --resume) shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --wandb) WANDB=1; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PYTHON="${SLRI_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python3"
CMD=("$PYTHON" -m slri.cli grid
  --config "$ROOT/configs/cities.yaml"
  --profile "$PROFILE"
  --seeds "$SEEDS"
  --storage-root "$STORAGE_ROOT"
  --device cuda:0
  --precision "$PRECISION")
[[ $FORCE -eq 1 ]] && CMD+=(--force)
[[ $DRY_RUN -eq 1 ]] && CMD+=(--dry-run)
[[ $WANDB -eq 1 ]] && CMD+=(--wandb)
[[ -n "$WANDB_PROJECT" ]] && CMD+=(--wandb-project "$WANDB_PROJECT")
[[ -n "$WANDB_ENTITY" ]] && CMD+=(--wandb-entity "$WANDB_ENTITY")

export PYTHONPATH="$ROOT/src:$ROOT/external/sheaf-mpnn/src${PYTHONPATH:+:$PYTHONPATH}"
export SLRI_STORAGE_ROOT="$STORAGE_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "${CMD[@]}"
