#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS="0"
PARALLEL=0
PROFILE="benchmark"
SEEDS="43"
STORAGE_ROOT="${SLRI_STORAGE_ROOT:-/workspace/sheaf-lri-storage}"
PRECISION="32"
FORCE=0
DRY_RUN=0
WANDB=0
WANDB_PROJECT=""
WANDB_ENTITY=""

usage() {
  echo "Usage: $0 [--gpus 0|0,1,2] [--parallel|--sequential]"
  echo "          [--profile benchmark|smoke] [--seeds 43]"
  echo "          [--storage-root PATH] [--precision ...] [--force]"
  echo "          [--dry-run] [--wandb] [--wandb-project NAME]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --parallel) PARALLEL=1; shift ;;
    --sequential) PARALLEL=0; shift ;;
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

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
COMMON=(--profile "$PROFILE" --seeds "$SEEDS"
  --storage-root "$STORAGE_ROOT" --precision "$PRECISION")
[[ $FORCE -eq 1 ]] && COMMON+=(--force)
[[ $DRY_RUN -eq 1 ]] && COMMON+=(--dry-run)
[[ $WANDB -eq 1 ]] && COMMON+=(--wandb)
[[ -n "$WANDB_PROJECT" ]] && COMMON+=(--wandb-project "$WANDB_PROJECT")
[[ -n "$WANDB_ENTITY" ]] && COMMON+=(--wandb-entity "$WANDB_ENTITY")

TASK_SCRIPTS=(
  "$ROOT/scripts/run_barbell_gpu.sh"
  "$ROOT/scripts/run_transfer_gpu.sh"
  "$ROOT/scripts/run_cities_gpu.sh"
)

if [[ $PARALLEL -eq 0 ]]; then
  GPU="${GPU_IDS[0]}"
  for script in "${TASK_SCRIPTS[@]}"; do
    "$script" --gpu "$GPU" "${COMMON[@]}"
  done
  exit 0
fi

if [[ ${#GPU_IDS[@]} -lt 3 ]]; then
  echo "Parallel mode needs at least three GPU IDs, for example --gpus 0,1,2" >&2
  exit 2
fi

PIDS=()
for index in 0 1 2; do
  "${TASK_SCRIPTS[$index]}" --gpu "${GPU_IDS[$index]}" "${COMMON[@]}" &
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    STATUS=1
  fi
done
exit "$STATUS"
