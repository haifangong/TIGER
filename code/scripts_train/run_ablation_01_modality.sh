#!/usr/bin/env bash
# Re-implement checkpoints/ablation/01_modality_ablation (feature_modalities sweep).
# Archive points: mod_{g,s,h,gs,gh,sh,gsh}  |  seed=123 by default (matches released archive)
#
# Usage (from TIGER/):
#   bash code/scripts_train/run_ablation_01_modality.sh
#   SEED=1 GPUS="0 1" FOREGROUND=1 bash code/scripts_train/run_ablation_01_modality.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ablation_common.sh"

OUT_ROOT="${OUT_ROOT:-$ROOT/runs_ablation_repro/01_modality_ablation}"
_ablation_launch "$OUT_ROOT" "scripts/run_struct_s_modality_grid.py"
