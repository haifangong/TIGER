#!/usr/bin/env bash
# External hemolytic toxicity eval on internal_toxin_cohort active subset (n=88, HC50≤512).
# Rebuilds / refreshes predictions from checkpoints/ablation_toxin/both checkpoints
# via data/test_internal_toxin_cohort/build_and_predict_internal_toxin_cohort.py.
#
# Clean panel for reporting also lives at:
#   data/test_external/test_toxin_internal_toxin_cohort/
#
# Usage (from TIGER/):
#   bash code/scripts_eval/eval_toxin_internal_toxin_cohort.sh
#   PY=/home/ubuntu/anaconda3/envs/class/bin/python bash code/scripts_eval/eval_toxin_internal_toxin_cohort.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts_train" && pwd)/_ablation_common.sh"

if [[ -z "${PY_OVERRIDE:-}" && -x /home/ubuntu/anaconda3/envs/class/bin/python ]]; then
  if [[ "$PY" == *"/ccseg/"* ]]; then
    PY=/home/ubuntu/anaconda3/envs/class/bin/python
  fi
fi

BUILDER="$ROOT/data/test_internal_toxin_cohort/build_and_predict_internal_toxin_cohort.py"
if [[ ! -f "$BUILDER" ]]; then
  echo "[eval] missing $BUILDER" >&2
  exit 1
fi

export FOREGROUND="${FOREGROUND:-1}"
_eval_run "toxin_internal_toxin_cohort" "$PY" "$BUILDER" "$@"

echo "[eval] primary clean labels: data/test_external/test_toxin_internal_toxin_cohort/"
echo "[eval] detailed metrics (if present): data/test_internal_toxin_cohort/eval_active_micmin_le128/"
