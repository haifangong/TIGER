#!/usr/bin/env bash
# Evaluate a single checkpoint with code.main evaluate-apexgo
# (high-span APEX-GO families; log10MAE / RSE / PCC / KCC).
#
# Usage (from TIGER/):
#   bash code/scripts_eval/eval_apexgo_high_span.sh
#   CONFIG=checkpoints/ablation/02_pair_balance_bin1/unsigned_bal10000/config.json \
#   CKPT=checkpoints/ablation/02_pair_balance_bin1/unsigned_bal10000/checkpoints/fold1_best.pt \
#     bash code/scripts_eval/eval_apexgo_high_span.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts_train" && pwd)/_ablation_common.sh"

EXP_DIR="${EXP_DIR:-$ROOT/checkpoints/ablation/02_pair_balance_bin1/unsigned_bal10000}"
CONFIG="${CONFIG:-$EXP_DIR/config.json}"
CKPT="${CKPT:-$EXP_DIR/checkpoints/fold1_best.pt}"

export FOREGROUND="${FOREGROUND:-1}"
_eval_run "apexgo_high_span" \
  "$PY" -m code.main evaluate-apexgo \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --gpu "$GPU" \
  "$@"
