#!/usr/bin/env bash
# Smoke-test CLI for 04_metric_ranking (does not require trained weights).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PY:-/home/ubuntu/anaconda3/envs/ccseg/bin/python}"
if [[ ! -x "$PY" ]]; then PY="$(command -v python3 || command -v python)"; fi
PYTHONPATH="$(cd "$HERE/../.." && pwd)" "$PY" rank_candidates.py --help >/tmp/rank_candidates_help.txt
head -n 40 /tmp/rank_candidates_help.txt
echo "[ok] rank_candidates.py --help"
