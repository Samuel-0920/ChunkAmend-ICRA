#!/usr/bin/env bash
# Run on the training server after copying this entire bundle. No robot connection.
set -euo pipefail
BUNDLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$BUNDLE_ROOT/datasets}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd "$BUNDLE_ROOT/training/openpi"
MODE="${1:-help}"
case "$MODE" in
  setup)
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
    ;;
  smoke|full|lora)
    if [[ ! -x .venv/bin/python ]]; then echo '先执行 bash training/server.sh setup'; exit 1; fi
    .venv/bin/python -c 'import jax; d=jax.devices(); print(d); assert len(d)==4 and all(x.platform=="gpu" for x in d), "需要单机4张可用GPU"'
    CFG=pi05_uf850_real_lemon_to_basket
    EXP=pi05_uf850_real_lemon_to_basket_v001_full_s42
    EXTRA=()
    if [[ "$MODE" == lora ]]; then
      CFG=pi05_uf850_real_lemon_to_basket_lora
      EXP=pi05_uf850_real_lemon_to_basket_v001_lora_s42
    elif [[ "$MODE" == smoke ]]; then
      EXP=pi05_uf850_real_lemon_to_basket_v001_full_smoke_s42
      EXTRA=(--num-train-steps 50 --save-interval 25 --keep-period 25)
    fi
    # No --overwrite. Existing runs cannot be silently replaced.
    exec .venv/bin/python scripts/train.py "$CFG" --exp-name "$EXP" "${EXTRA[@]}"
    ;;
  *) echo '用法: bash training/server.sh setup | smoke | full | lora';;
esac
