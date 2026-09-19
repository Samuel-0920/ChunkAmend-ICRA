#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_LEROBOT_HOME="$BUNDLE_ROOT/datasets"
export PYTHONPATH="$BUNDLE_ROOT/training/openpi/src:$BUNDLE_ROOT/training/openpi/packages/openpi-client/src"
cd "$BUNDLE_ROOT/training/openpi"
exec "$BUNDLE_ROOT/.verify-env/bin/python" "$@"
