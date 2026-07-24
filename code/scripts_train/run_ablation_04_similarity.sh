#!/usr/bin/env bash
# Re-implement runs_ablation/04_similarity (sim ∈ {0.3, 0.5, 0.7} @ bal=10000).
#   - sim=0.3 comes from the binsize unsigned_bal10000 point
#   - sim=0.5 / 0.7 come from the sim05/07 fusion grid
#
# Usage (from TIGER/):
#   bash code/scripts_train/run_ablation_04_similarity.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ablation_common.sh"

# Reuse binsize grid for sim0.3 @ bal=10000 (full unsigned/signed grid is OK / resumable)
BIN_OUT="${BIN_OUT:-$ROOT/runs_ablation_repro/02_pair_balance_bin1}"
SIM_OUT="${SIM_OUT:-$ROOT/runs_ablation_repro/04_similarity}"

_ablation_launch "$BIN_OUT" "scripts/run_struct_s_binsize_grid.py"
_ablation_launch "$SIM_OUT" "scripts/run_struct_s_sim05_07_fusion_grid.py"
