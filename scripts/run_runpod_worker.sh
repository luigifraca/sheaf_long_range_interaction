#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROLE="${1:?usage: run_runpod_worker.sh ROLE}"
DEPLOYMENT_ID="${SLRI_DEPLOYMENT_ID:-$(date -u +%Y%m%d-%H%M%S)}"
VOLUME_ROOT="${SLRI_VOLUME_ROOT:-/workspace/sheaf-lri-storage}"
STORAGE_ROOT="$VOLUME_ROOT/deployments/$DEPLOYMENT_ID/workers/$ROLE"
LOG_ROOT="$VOLUME_ROOT/deployments/$DEPLOYMENT_ID/logs"
PROJECT="${WANDB_PROJECT:-sheaf-long-range-full}"
COMMON=(
  --profile analysis
  --seeds 43
  --storage-root "$STORAGE_ROOT"
  --device cuda:0
  --precision 32
  --wandb
  --wandb-project "$PROJECT"
)

mkdir -p "$STORAGE_ROOT" "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/$ROLE.log") 2>&1
cd "$ROOT"

echo "[$(date -u +%FT%TZ)] starting role=$ROLE storage=$STORAGE_ROOT"
nvidia-smi

case "$ROLE" in
  barbell)
    CONFIG="configs/barbell.yaml"
    ANALYSIS_CONFIG="configs/analysis_runpod.yaml"
    SELECTORS=()
    ;;
  transfer-0|transfer-1|transfer-2)
    CONFIG="configs/transfer.yaml"
    ANALYSIS_CONFIG="configs/analysis_runpod.yaml"
    SHARD="${ROLE##*-}"
    SELECTORS=(--shard-count 3 --shard-index "$SHARD")
    ;;
  cities-paris)
    CONFIG="configs/cities.yaml"
    ANALYSIS_CONFIG="configs/analysis_cities_runpod.yaml"
    SELECTORS=(--dataset paris)
    ;;
  cities-shanghai)
    CONFIG="configs/cities.yaml"
    ANALYSIS_CONFIG="configs/analysis_cities_runpod.yaml"
    SELECTORS=(--dataset shanghai)
    ;;
  *)
    echo "Unknown worker role: $ROLE" >&2
    exit 2
    ;;
esac

python -m slri.cli grid \
  --config "$CONFIG" \
  "${COMMON[@]}" \
  "${SELECTORS[@]}" \
  --fail-fast

python -m slri.cli analyze grid \
  --config "$ANALYSIS_CONFIG" \
  --profile benchmark \
  --storage-root "$STORAGE_ROOT" \
  --device cuda:0 \
  --checkpoints initial,best \
  --fail-fast

python -m slri.cli analyze compare \
  --query status=completed \
  --storage-root "$STORAGE_ROOT" \
  --output "$VOLUME_ROOT/deployments/$DEPLOYMENT_ID/summaries/$ROLE"

touch "$STORAGE_ROOT/WORKER_COMPLETE"
echo "[$(date -u +%FT%TZ)] completed role=$ROLE"
