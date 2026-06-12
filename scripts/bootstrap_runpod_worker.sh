#!/usr/bin/env bash
set -euo pipefail

ROLE="${SLRI_WORKER_ROLE:?SLRI_WORKER_ROLE is required}"
ROLES="${SLRI_WORKER_ROLES:-$ROLE}"
REPOSITORY="${SLRI_REPOSITORY:-https://github.com/luigifraca/sheaf_long_range_interaction.git}"
REVISION="${SLRI_REVISION:-main}"
CODE_ROOT="/root/sheaf_long_range_interaction"
SOURCE_ROOT="${SLRI_SOURCE_ROOT:-}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends curl git ca-certificates
rm -rf /var/lib/apt/lists/*

if [[ -n "$SOURCE_ROOT" ]]; then
  if [[ ! -d "$SOURCE_ROOT/src/slri" ]]; then
    echo "SLRI_SOURCE_ROOT does not contain an SLRI source tree: $SOURCE_ROOT" >&2
    exit 2
  fi
  rm -rf "$CODE_ROOT"
  cp -a "$SOURCE_ROOT" "$CODE_ROOT"
elif [[ ! -d "$CODE_ROOT/.git" ]]; then
  git clone --recurse-submodules "$REPOSITORY" "$CODE_ROOT"
fi
cd "$CODE_ROOT"
if [[ -d .git ]]; then
  git fetch origin "$REVISION"
  git checkout --force "$REVISION"
  git submodule update --init --recursive
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.13
uv sync --extra wandb
scripts/setup_curvature_env.sh

export PATH="$CODE_ROOT/.venv/bin:$PATH"
export SLRI_PYTHON="$CODE_ROOT/.venv/bin/python"
export SLRI_CURVATURE_PYTHON="$CODE_ROOT/.venv-curvature/bin/python"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/external/sheaf-mpnn/src"
export WANDB_DIR="${SLRI_VOLUME_ROOT:-/workspace/sheaf-lri-storage}/wandb"
mkdir -p "$WANDB_DIR"

IFS=',' read -r -a WORKER_ROLES <<< "$ROLES"
for worker_role in "${WORKER_ROLES[@]}"; do
  scripts/run_runpod_worker.sh "$worker_role"
done
