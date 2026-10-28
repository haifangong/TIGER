#!/usr/bin/env bash
# Re-implement checkpoints/ablation_toxin/{both,global,sequence}
# Classical ML only (matches archive), HC50 threshold=512, 5-fold CV, seed=1.
#
# Usage (from TIGER/):
#   bash code/scripts_train/run_ablation_toxin.sh
#   MODE=both bash code/scripts_train/run_ablation_toxin.sh          # one feature mode
#   SEED=1 SKIP_DL=1 bash code/scripts_train/run_ablation_toxin.sh
#   PY=/home/ubuntu/anaconda3/envs/class/bin/python bash code/scripts_train/run_ablation_toxin.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ablation_common.sh"

# Classical ML toxin ablations often use the lighter `class` env if present
if [[ -z "${PY_OVERRIDE:-}" && -x /home/ubuntu/anaconda3/envs/class/bin/python ]]; then
  # Keep caller PY if they already set it; otherwise prefer class for toxin ML
  if [[ "$PY" == *"/ccseg/"* ]]; then
    PY=/home/ubuntu/anaconda3/envs/class/bin/python
  fi
fi

THRESHOLD="${THRESHOLD:-512}"
SEED="${SEED:-1}"
SKIP_DL="${SKIP_DL:-1}"
MODE="${MODE:-all}"   # all | both | global | sequence
OUT_BASE="${OUT_BASE:-$ROOT/runs_ablation_toxin_repro}"
GPU="${GPU:-0}"

modes=()
if [[ "$MODE" == "all" ]]; then
  modes=(both global sequence)
else
  modes=("$MODE")
fi

echo "[toxin] ROOT=$ROOT PY=$PY seed=$SEED threshold=$THRESHOLD gpu=$GPU"
echo "[toxin] modes: ${modes[*]}"

for mode in "${modes[@]}"; do
  out="$OUT_BASE/$mode"
  mkdir -p "$out"
  cmd=(
    "$PY" -m code.toxin_filter.run
    --feature-mode "$mode"
    --threshold "$THRESHOLD"
    --seed "$SEED"
    --n-splits 5
    --gpu "$GPU"
    --out-dir "$out"
  )
  if [[ "$SKIP_DL" == "1" ]]; then
    cmd+=(--skip-dl)
  fi
  echo "[toxin] >>> ${cmd[*]}"
  if [[ "$FOREGROUND" == "0" && "$MODE" == "all" ]]; then
    # sequential foreground for toxin (fast classical ML); still log
    mkdir -p "$out"
    "${cmd[@]}" |& tee "$out/launcher.log"
  else
    "${cmd[@]}" |& tee "$out/launcher.log"
  fi
done

echo "[toxin] done → $OUT_BASE"
