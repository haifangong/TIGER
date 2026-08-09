#!/usr/bin/env bash
# External eval of a trained MIC experiment on:
#   - LL37 neighbor-509 pairs
#   - APEX-GO geo3 template-centric 200 pairs
# Default experiment: best unsigned_bal10000 recipe (or archive copy).
#
# Usage (from TIGER/):
#   bash code/scripts_eval/eval_ll37_apexgo_best.sh
#   EXP_DIR=checkpoints/ablation/02_pair_balance_bin1/unsigned_bal10000 GPU=0 \
#     bash code/scripts_eval/eval_ll37_apexgo_best.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts_train" && pwd)/_ablation_common.sh"

EXP_DIR="${EXP_DIR:-$ROOT/checkpoints/ablation/02_pair_balance_bin1/unsigned_bal10000}"
if [[ ! -d "$EXP_DIR" ]]; then
  EXP_DIR="${EXP_DIR_FALLBACK:-$ROOT/outputs/outputs_code_struct_s_binsize_grid/binsize_grid__unsigned_bin1p0_bal10000}"
fi

EXTRA=()
[[ "${SKIP_LL37:-0}" == "1" ]] && EXTRA+=(--skip-ll37)
[[ "${SKIP_APEXGO:-0}" == "1" ]] && EXTRA+=(--skip-apexgo)

export FOREGROUND="${FOREGROUND:-1}"
_eval_run "ll37_apexgo_best" \
  "$PY" scripts/eval_best_binsize_ll37_apexgo.py \
  --exp-dir "$EXP_DIR" \
  --gpu "$GPU" \
  "${EXTRA[@]}"
