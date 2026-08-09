#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/outputs_code_struct_s_seq_encoding_grid}"
PY="${PY:-/home/ubuntu/anaconda3/envs/ccseg/bin/python}"
mkdir -p "$OUT_ROOT/suite_logs"
cd "$ROOT"
export PYTHONPATH="$ROOT"
nohup "$PY" scripts/run_struct_s_seq_encoding_grid.py \
  --out-root "$OUT_ROOT" \
  --gpus 0 1 \
  --slots-per-gpu 1 \
  > "$OUT_ROOT/suite_logs/launcher.log" 2>&1 &
echo $! > "$OUT_ROOT/suite.pid"
echo "launched pid=$(cat "$OUT_ROOT/suite.pid") log=$OUT_ROOT/suite_logs/launcher.log"
