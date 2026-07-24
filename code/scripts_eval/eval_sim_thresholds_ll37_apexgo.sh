#!/usr/bin/env bash
# Compare external LL37 + APEX-GO metrics across training similarity thresholds
# sim ∈ {0.3, 0.5, 0.7} (matches runs_ablation/04_similarity).
#
# Usage (from TIGER/):
#   bash code/scripts_eval/eval_sim_thresholds_ll37_apexgo.sh
#   GPU=0 bash code/scripts_eval/eval_sim_thresholds_ll37_apexgo.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts_train" && pwd)/_ablation_common.sh"

export FOREGROUND="${FOREGROUND:-1}"
_eval_run "sim03_05_07_ll37_apexgo" \
  "$PY" scripts/eval_sim03_05_07_ll37_apexgo.py \
  --gpu "$GPU" \
  "$@"
