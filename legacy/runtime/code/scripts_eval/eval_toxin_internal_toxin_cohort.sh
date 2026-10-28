#!/usr/bin/env bash
# External hemolytic toxicity eval on the clean 88-peptide internal_toxin_cohort panel
# (HC50 ≤ 512 µg/mL; active filter mic_min ≤ 128).
#
# Labels:  data/test_external/test_toxin_internal_toxin_cohort/
# Models:  checkpoints/ablation_toxin/both/checkpoints/
# Runner:  code/toxin_filter/eval_internal_toxin_cohort.py
#
# Usage (from TIGER/):
#   bash code/scripts_eval/eval_toxin_internal_toxin_cohort.sh
#   FEATURE_MODE=both bash code/scripts_eval/eval_toxin_internal_toxin_cohort.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts_train" && pwd)/_ablation_common.sh"

if [[ -z "${PY_OVERRIDE:-}" && -x /home/ubuntu/anaconda3/envs/class/bin/python ]]; then
  if [[ "$PY" == *"/ccseg/"* ]]; then
    PY=/home/ubuntu/anaconda3/envs/class/bin/python
  fi
fi

LABELS="${LABELS:-$ROOT/data/test_external/test_toxin_internal_toxin_cohort/internal_toxin_cohort_toxicity_labeled.csv}"
PANEL="${PANEL:-$ROOT/data/test_external/test_toxin_internal_toxin_cohort/internal_toxin_cohort_hemolysis_active_micmin_le128.csv}"
CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/ablation_toxin/${FEATURE_MODE:-both}/checkpoints}"
OUT_DIR="${OUT_DIR:-$ROOT/data/test_external/test_toxin_internal_toxin_cohort}"
FEATURE_MODE="${FEATURE_MODE:-both}"

if [[ ! -f "$LABELS" ]]; then
  echo "[eval] missing labels: $LABELS" >&2
  exit 1
fi
if [[ ! -d "$CKPT_DIR" ]]; then
  echo "[eval] missing checkpoints: $CKPT_DIR" >&2
  exit 1
fi

export FOREGROUND="${FOREGROUND:-1}"
_eval_run "toxin_internal_toxin_cohort" \
  "$PY" "$ROOT/code/toxin_filter/eval_internal_toxin_cohort.py" \
  --labels-csv "$LABELS" \
  --panel-csv "$PANEL" \
  --ckpt-dir "$CKPT_DIR" \
  --feature-mode "$FEATURE_MODE" \
  --out-dir "$OUT_DIR" \
  "$@"

echo "[eval] labels + predictions: $OUT_DIR"
