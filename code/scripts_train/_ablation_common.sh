#!/usr/bin/env bash
# Shared env for launchers under code/scripts_train/ and code/scripts_eval/.
# Source:  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ablation_common.sh"
# Or from scripts_eval: source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts_train" && pwd)/_ablation_common.sh"

set -euo pipefail

# This file lives in code/scripts_train/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # TIGER/code
ROOT="$(cd "$CODE_DIR/.." && pwd)"                # TIGER/
TRAIN_SCRIPTS="$CODE_DIR/scripts_train"
EVAL_SCRIPTS="$CODE_DIR/scripts_eval"

cd "$ROOT"
export PYTHONPATH="$ROOT"

# Prefer ccseg (MIC / PyTorch+PyG); toxin classical-ML can override with PY=.../class/bin/python
PY="${PY:-/home/ubuntu/anaconda3/envs/ccseg/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

# Default seed for panels 01–04 (not date-style). Panel 05 sweeps 1..5.
SEED="${SEED:-1}"
GPUS="${GPUS:-0 1}"
GPU="${GPU:-0}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
FOREGROUND="${FOREGROUND:-0}"

_ablation_launch() {
  # Usage: _ablation_launch <out_root> <python_script> [extra args...]
  local out_root="$1"
  local py_script="$2"
  shift 2
  mkdir -p "$out_root/suite_logs"
  local log="$out_root/suite_logs/launcher.log"
  local cmd=(
    "$PY" "$py_script"
    --out-root "$out_root"
    --gpus $GPUS
    --slots-per-gpu "$SLOTS_PER_GPU"
    --seed "$SEED"
    "$@"
  )
  echo "[ablation] ROOT=$ROOT"
  echo "[ablation] PY=$PY"
  echo "[ablation] SEED=$SEED GPUS=($GPUS) slots=$SLOTS_PER_GPU"
  echo "[ablation] out=$out_root"
  echo "[ablation] cmd: ${cmd[*]}"
  if [[ "$FOREGROUND" == "1" ]]; then
    "${cmd[@]}"
  else
    nohup "${cmd[@]}" >"$log" 2>&1 &
    echo $! >"$out_root/suite.pid"
    echo "[ablation] launched pid=$(cat "$out_root/suite.pid") log=$log"
  fi
}

_eval_run() {
  # Usage: _eval_run <log_tag> -- python args...
  local tag="$1"
  shift
  local log_dir="${EVAL_LOG_DIR:-$ROOT/runs_eval_logs}"
  mkdir -p "$log_dir"
  local log="$log_dir/${tag}.log"
  echo "[eval] ROOT=$ROOT PY=$PY GPU=$GPU"
  echo "[eval] cmd: $*"
  if [[ "$FOREGROUND" == "1" ]]; then
    "$@"
  else
    nohup "$@" >"$log" 2>&1 &
    echo $! >"$log_dir/${tag}.pid"
    echo "[eval] launched pid=$(cat "$log_dir/${tag}.pid") log=$log"
  fi
}
