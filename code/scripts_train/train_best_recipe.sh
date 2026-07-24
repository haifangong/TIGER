#!/usr/bin/env bash
# Train the locked best MIC recipe (paper default):
#   gsh + attention/cross_qs + struct=s + sim=0.3 + unsigned bal=10000 + seed=1
#
# Usage (from TIGER/):
#   bash code/scripts_train/train_best_recipe.sh
#   OUT_DIR=runs_ablation_repro/best_unsigned_bal10000 GPU=0 bash code/scripts_train/train_best_recipe.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ablation_common.sh"

OUT_DIR="${OUT_DIR:-$ROOT/runs_ablation_repro/best_unsigned_bal10000}"
NAME="${NAME:-unsigned_bin1p0_bal10000}"
SEED="${SEED:-1}"

mkdir -p "$OUT_DIR"
cmd=(
  "$PY" -m code.main train
  --config code/configs/gsh_struct_s_base.json
  --out-dir "$OUT_DIR"
  --name "$NAME"
  --structure-features s
  --feature-modalities gsh
  --fusion attention
  --fusion-attn-mode cross_qs
  --no-include-node-coords
  --similarity-threshold 0.3
  --delta-bin-width 1.0
  --pair-balance-num 10000
  --no-use-signed-sampling
  --seed "$SEED"
  --gpu "$GPU"
)

echo "[train] ${cmd[*]}"
if [[ "$FOREGROUND" == "1" ]]; then
  "${cmd[@]}"
else
  log="$OUT_DIR/train.log"
  nohup "${cmd[@]}" >"$log" 2>&1 &
  echo $! >"$OUT_DIR/train.pid"
  echo "[train] launched pid=$(cat "$OUT_DIR/train.pid") log=$log"
fi
