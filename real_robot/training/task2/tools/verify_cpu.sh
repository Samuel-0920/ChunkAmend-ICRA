#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CPU_PY="$BUNDLE_ROOT/training/openpi/.venv/bin/python"
if [[ ! -x "$CPU_PY" ]]; then
  CPU_PY="${UF850_TRAIN_PYTHON:-python3}"
fi
if [[ ! -x "$CPU_PY" ]]; then echo 'Run bash training/server.sh setup first' >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="$BUNDLE_ROOT/.cache/datasets"
export HF_LEROBOT_HOME="$BUNDLE_ROOT/datasets"
export PYTHONPATH="$BUNDLE_ROOT/training/openpi/src:$BUNDLE_ROOT/training/openpi/packages/openpi-client/src"
cd "$BUNDLE_ROOT/training/openpi"
exec "$CPU_PY" "$@"
