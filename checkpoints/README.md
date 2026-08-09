# checkpoints — final ablation / toxin weights

## MIC pair-delta ablations

Final **selected** MIC checkpoints from `runs_ablation/`: only `fold{1..5}_best.pt`
per experiment point (including the completed **sequence-encoding** panel).

```text
checkpoints/
├── 01_modality_ablation/
├── 02_pair_balance_bin1/
├── 03_fusion_methods/
├── 04_similarity/
├── 05_seed_stability/
├── 06_seq_encoding/          # integer / embedding / onehot (5 folds each)
├── toxin_classification/     # hemolytic HC50 classical-ML (+ MLP) joblibs
├── leaderboard.csv
└── CHECKPOINTS_MANIFEST.json
```

`fold*_last.pt`, training logs, prediction dumps, and wetlab production models
are **not** redistributed here.

Sequence-encoding **metrics tables** (without weights) also live under
[`data/ablation_results/06_seq_encoding/`](../data/ablation_results/06_seq_encoding/).

## Toxicity classification

`toxin_classification/{both,global,sequence}/checkpoints/*.joblib` — 5-fold
models for the HC50≤512 µg/mL filter ablation. Summaries:
`summary_metrics.csv`, `cv_fold_metrics.csv`.

## Usage

```bash
export PYTHONPATH=.
python -m code.main evaluate \
  --config checkpoints/06_seq_encoding/seq_encoding_grid__seq_integer/config.json \
  --checkpoint checkpoints/06_seq_encoding/seq_encoding_grid__seq_integer/checkpoints/fold1_best.pt \
  --gpu 0
```

Reproduce sequence-encoding panel:

```bash
bash code/scripts_train/run_ablation_06_seq_encoding.sh
```
