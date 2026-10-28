#!/usr/bin/env bash
# Re-implement checkpoints/ablation/03_fusion_methods
#   - attention Q/KV modes (sim×bin×attn grid; paper panel uses sim0.3/bin1.0 @ bal=1000)
#   - concat gsh @ bal=10000
#
# Usage (from TIGER/):
#   bash code/scripts_train/run_ablation_03_fusion.sh
#   PART=attn|concat|all bash code/scripts_train/run_ablation_03_fusion.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ablation_common.sh"

PART="${PART:-all}"
ATTN_OUT="${ATTN_OUT:-$ROOT/runs_ablation_repro/03_fusion_methods/attn_grid}"
CONCAT_OUT="${CONCAT_OUT:-$ROOT/runs_ablation_repro/03_fusion_methods/concat_grid}"

if [[ "$PART" == "attn" || "$PART" == "all" ]]; then
  _ablation_launch "$ATTN_OUT" "scripts/run_struct_s_sim_bin_qkv_grid.py"
fi
if [[ "$PART" == "concat" || "$PART" == "all" ]]; then
  # Wait for attn launcher pid file only when both run in background sequentially here
  _ablation_launch "$CONCAT_OUT" "scripts/run_struct_s_concat_ablation.py"
fi
