#!/usr/bin/env bash
# External hemolytic toxicity eval on qlx227 active subset (n=88, HC50≤512).
# Rebuilds / refreshes predictions from runs_ablation_toxin/both checkpoints
# via data/test_qlx227/build_and_predict_qlx227.py.
#
# Clean panel for reporting also lives at:
#   data/test_external/test_toxin_qlx227/
#
# Usage (from TIGER/):
#   bash code/scripts_eval/eval_toxin_qlx227.sh
#   PY=/home/ubuntu/anaconda3/envs/class/bin/python bash code/scripts_eval/eval_toxin_qlx227.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts_train" && pwd)/_ablation_common.sh"

if [[ -z "${PY_OVERRIDE:-}" && -x /home/ubuntu/anaconda3/envs/class/bin/python ]]; then
  if [[ "$PY" == *"/ccseg/"* ]]; then
    PY=/home/ubuntu/anaconda3/envs/class/bin/python
  fi
fi

BUILDER="$ROOT/data/test_qlx227/build_and_predict_qlx227.py"
if [[ ! -f "$BUILDER" ]]; then
  echo "[eval] missing $BUILDER" >&2
  exit 1
fi

export FOREGROUND="${FOREGROUND:-1}"
_eval_run "toxin_qlx227" "$PY" "$BUILDER" "$@"

echo "[eval] primary clean labels: data/test_external/test_toxin_qlx227/"
echo "[eval] detailed metrics (if present): data/test_qlx227/eval_active_micmin_le128/"
