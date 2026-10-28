#!/usr/bin/env bash
# Score LL37/external queries with a checkpoint (neighbor pair-delta inference).
#
# Usage (from TIGER/):
#   bash code/scripts_eval/eval_infer_checkpoint.sh
#   CONFIG=.../config.json CKPT=.../fold1_best.pt bash code/scripts_eval/eval_infer_checkpoint.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts_train" && pwd)/_ablation_common.sh"

EXP_DIR="${EXP_DIR:-$ROOT/checkpoints/ablation/02_pair_balance_bin1/unsigned_bal10000}"
CONFIG="${CONFIG:-$EXP_DIR/config.json}"
CKPT="${CKPT:-$EXP_DIR/checkpoints/fold1_best.pt}"

export FOREGROUND="${FOREGROUND:-1}"
_eval_run "infer_checkpoint" \
  "$PY" -m code.main infer \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --gpu "$GPU" \
  "$@"
