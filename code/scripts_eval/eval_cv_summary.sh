#!/usr/bin/env bash
# Print saved 5-fold CV summary.json for a trained MIC experiment.
#
# Usage (from TIGER/):
#   bash code/scripts_eval/eval_cv_summary.sh
#   EXP_DIR=runs_ablation/01_modality_ablation/mod_gsh bash code/scripts_eval/eval_cv_summary.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts_train" && pwd)/_ablation_common.sh"

EXP_DIR="${EXP_DIR:-$ROOT/runs_ablation/02_pair_balance_bin1/unsigned_bal10000}"
CONFIG="${CONFIG:-$EXP_DIR/config.json}"

export FOREGROUND=1
"$PY" -m code.main evaluate --config "$CONFIG" --gpu "$GPU"
