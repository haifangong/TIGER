#!/usr/bin/env bash
# Re-implement runs_ablation/02_pair_balance_bin1 (unsigned/signed × bal sweep).
# Archive points: {unsigned,signed}_bal{1000..20000}  |  seed=1 by default
#
# Usage (from TIGER/):
#   bash code/scripts_train/run_ablation_02_pair_balance.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ablation_common.sh"

OUT_ROOT="${OUT_ROOT:-$ROOT/runs_ablation_repro/02_pair_balance_bin1}"
_ablation_launch "$OUT_ROOT" "scripts/run_struct_s_binsize_grid.py"
