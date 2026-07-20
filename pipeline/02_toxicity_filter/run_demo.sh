#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python infer_toxin.py --csv examples/sample_input.csv --out-dir outputs/demo
