#!/usr/bin/env bash
# Re-implement runs_ablation/05_seed_stability
# Sweep: sim ∈ {0.3, 0.7} × seed ∈ {1, 2, 3, 4, 5}  → 10 runs
# Folder names: sim0p{3|7}_bal10000_seed{1..5}  (no date-style seeds)
#
# Note: SEED env is ignored here — the Python grid owns the 1..5 sweep.
#
# Usage (from TIGER/):
#   bash code/scripts_train/run_ablation_05_seed_stability.sh
#   FOREGROUND=1 bash code/scripts_train/run_ablation_05_seed_stability.sh
#   # refresh tables only:
#   PYTHONPATH=. "$PY" scripts/run_struct_s_seed_stability_grid.py --leaderboard-only

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ablation_common.sh"

# Default writes alongside the archive under 05_seed_stability with seed{1..5} names.
# Override to isolate: OUT_ROOT=$ROOT/runs_ablation_repro/05_seed_stability
OUT_ROOT="${OUT_ROOT:-$ROOT/runs_ablation/05_seed_stability}"
mkdir -p "$OUT_ROOT/suite_logs"
log="$OUT_ROOT/suite_logs/launcher.log"

cmd=(
  "$PY" scripts/run_struct_s_seed_stability_grid.py
  --out-root "$OUT_ROOT"
  --gpus $GPUS
  --slots-per-gpu "$SLOTS_PER_GPU"
)

echo "[ablation] ROOT=$ROOT"
echo "[ablation] PY=$PY"
echo "[ablation] GPUS=($GPUS) slots=$SLOTS_PER_GPU"
echo "[ablation] out=$OUT_ROOT"
echo "[ablation] grid: sim={0.3,0.7} × seeds={1,2,3,4,5}"
echo "[ablation] cmd: ${cmd[*]}"

if [[ "$FOREGROUND" == "1" ]]; then
  "${cmd[@]}"
else
  nohup "${cmd[@]}" >"$log" 2>&1 &
  echo $! >"$OUT_ROOT/suite.pid"
  echo "[ablation] launched pid=$(cat "$OUT_ROOT/suite.pid") log=$log"
fi
