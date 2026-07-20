#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python infer_batch.py --csv_file examples/sample_sequences.csv --output_dir outputs/pdb/demo
# Optional:
# python relax.py --input_dir outputs/pdb/demo --output_dir outputs/relaxed/demo
