#!/usr/bin/env bash
# Train wetlab MIC models: full DBAASP × top-6 species × sim {0.3, 0.7} → 12 runs.
#
# Usage (from TIGER/):
#   bash code/scripts_train/run_wetlab_species_sim.sh
#   FOREGROUND=1 GPUS="0 1" bash code/scripts_train/run_wetlab_species_sim.sh
#   OUT_ROOT=checkpoints/wetlab bash code/scripts_train/run_wetlab_species_sim.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ablation_common.sh"

OUT_ROOT="${OUT_ROOT:-$ROOT/checkpoints/wetlab}"
DATA_DIR="${DATA_DIR:-$ROOT/data/wetlab}"
TOP_K="${TOP_K:-6}"

# Prepare tables first (idempotent), then launch the 12-run grid.
"$PY" scripts/prepare_wetlab_dbassp_tables.py --out-dir "$DATA_DIR" --top-k "$TOP_K"

_ablation_launch "$OUT_ROOT" scripts/run_wetlab_species_sim_grid.py \
  --data-dir "$DATA_DIR" \
  --top-k "$TOP_K"
