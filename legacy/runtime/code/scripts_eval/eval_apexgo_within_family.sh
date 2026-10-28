#!/usr/bin/env bash
# APEX-GO within-family / template-centric pair evaluation for one experiment.
#
# Usage (from TIGER/):
#   bash code/scripts_eval/eval_apexgo_within_family.sh
#   EXP_DIR=checkpoints/ablation/02_pair_balance_bin1/unsigned_bal10000 \
#     bash code/scripts_eval/eval_apexgo_within_family.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts_train" && pwd)/_ablation_common.sh"

EXP_DIR="${EXP_DIR:-$ROOT/checkpoints/ablation/02_pair_balance_bin1/unsigned_bal10000}"
PAIR_CSV="${PAIR_CSV:-$ROOT/data/test_external/test_activity_apexgo/pairs/pairs_template_centric_alldelta_geo3.csv}"
if [[ ! -f "$PAIR_CSV" ]]; then
  echo "[eval] missing pair csv: $PAIR_CSV" >&2
  exit 1
fi

export FOREGROUND="${FOREGROUND:-1}"
_eval_run "apexgo_within_family" \
  "$PY" scripts/eval_apexgo_within_family_combos.py \
  --exp-dir "$EXP_DIR" \
  --pair-csv "$PAIR_CSV" \
  --gpu "$GPU" \
  "$@"
