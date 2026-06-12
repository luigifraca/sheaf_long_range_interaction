#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${SLRI_CURVATURE_PYTHON_VERSION:-3.12}"
ENV_DIR="${SLRI_CURVATURE_ENV:-$ROOT/.venv-curvature}"
UV="${SLRI_UV:-uv}"
if ! command -v "$UV" >/dev/null 2>&1 && [[ -x "$ROOT/.venv/bin/uv" ]]; then
  UV="$ROOT/.venv/bin/uv"
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$UV" venv --python "$PYTHON_VERSION" "$ENV_DIR"
fi
"$UV" pip install --python "$ENV_DIR/bin/python" \
  -r "$ROOT/external/GraphRicciCurvature/requirements.txt"

echo "Curvature sidecar ready: $ENV_DIR/bin/python"
echo "export SLRI_CURVATURE_PYTHON=$ENV_DIR/bin/python"
