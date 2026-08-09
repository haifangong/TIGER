#!/usr/bin/env bash
# External eval: TIGER hard MoE (0.3+0.7) / TIGER 0.3 / TIGER 0.7 vs EvoGradient
# on LL37 neighbor pairs and APEXGO template-centric pairs.
#
# Metrics (per peptide group): PCC, KCC, MAE (=log2MAE), RSE
#
# Usage (from TIGER/):
#   bash code/scripts_eval/eval_tiger_vs_evo.sh
#   GPU=0 bash code/scripts_eval/eval_tiger_vs_evo.sh
#   DATASETS=LL37 bash code/scripts_eval/eval_tiger_vs_evo.sh
#   DATASETS=APEXGO bash code/scripts_eval/eval_tiger_vs_evo.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts_train" && pwd)/_ablation_common.sh"

export FOREGROUND="${FOREGROUND:-1}"
EXP_DIR_03="${EXP_DIR_03:-$ROOT/checkpoints/ablation/04_similarity/sim0p3_bal10000}"
EXP_DIR_07="${EXP_DIR_07:-$ROOT/checkpoints/ablation/04_similarity/sim0p7_bal10000}"
MOE_TAU="${MOE_TAU:-0.5}"
DATASETS="${DATASETS:-LL37,APEXGO}"
OUT_DIR="${OUT_DIR:-$ROOT/checkpoints/ablation/04_similarity/external_eval_tiger_vs_evo}"

_eval_run "tiger_vs_evo" \
  "$PY" code/compare_tiger_evo.py \
  --exp-dir-03 "$EXP_DIR_03" \
  --exp-dir-07 "$EXP_DIR_07" \
  --moe-tau "$MOE_TAU" \
  --datasets "$DATASETS" \
  --out-dir "$OUT_DIR" \
  --gpu "$GPU" \
  "$@"
