#!/usr/bin/env bash
# Launch all MIC paper ablation panels (01–06) under runs_ablation_repro/ (05/06 → archive).
# Panels run **sequentially** (each blocks until its suite finishes) so they do not
# contend for the same GPUs. Default seed for 01–04 is 123; panel 06 uses 1; panel 05 sweeps 1–5.
#
# Usage (from TIGER/):
#   bash code/scripts_train/run_ablation_all_mic.sh
#   GPUS="0 1" bash code/scripts_train/run_ablation_all_mic.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ablation_common.sh"

# Force foreground so panels do not overlap on GPUs
export FOREGROUND=1

echo "[ablation] launching MIC panels 01→06 sequentially (FOREGROUND=1)"
bash "$TRAIN_SCRIPTS/run_ablation_01_modality.sh"
bash "$TRAIN_SCRIPTS/run_ablation_02_pair_balance.sh"
bash "$TRAIN_SCRIPTS/run_ablation_03_fusion.sh"
bash "$TRAIN_SCRIPTS/run_ablation_04_similarity.sh"
bash "$TRAIN_SCRIPTS/run_ablation_05_seed_stability.sh"
bash "$TRAIN_SCRIPTS/run_ablation_06_seq_encoding.sh"
echo "[ablation] all MIC panels finished"
