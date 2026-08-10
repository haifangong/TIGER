#!/usr/bin/env bash
# Sequence-encoding ablation: integer vs embedding vs onehot under locked recipe.
# Archive panel: checkpoints/ablation/06_seq_encoding/
#
# Usage (from TIGER/):
#   bash code/scripts_train/run_ablation_06_seq_encoding.sh
#   SEED=1 GPUS="0 1" FOREGROUND=1 bash code/scripts_train/run_ablation_06_seq_encoding.sh
#   # only new categorical methods:
#   ONLY="seq_embedding seq_onehot" bash code/scripts_train/run_ablation_06_seq_encoding.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ablation_common.sh"
# Panel 06 released archive uses seed=1 (not 123).
SEED="${SEED:-1}"

OUT_ROOT="${OUT_ROOT:-$ROOT/checkpoints/ablation/06_seq_encoding}"
EXTRA=()
if [[ -n "${ONLY:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA+=(--only $ONLY)
fi
_ablation_launch "$OUT_ROOT" "scripts/run_struct_s_seq_encoding_grid.py" "${EXTRA[@]}"
