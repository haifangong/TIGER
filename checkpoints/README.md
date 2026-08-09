# checkpoints — final ablation weights

Final **selected** MIC pair-delta checkpoints from the paper ablation archive
(`runs_ablation/`): only `fold{1..5}_best.pt` per experiment point.

`fold*_last.pt`, training logs, prediction dumps, graph caches, wetlab models,
and other checkpoints are **not** redistributed here.

```text
checkpoints/
├── CHECKPOINTS_MANIFEST.json
├── leaderboard.csv
├── MANIFEST.json
├── 01_modality_ablation/<point>/checkpoints/fold*_best.pt
├── 02_pair_balance_bin1/...
├── 03_fusion_methods/...
├── 04_similarity/...
├── 05_seed_stability/...
└── 06_seq_encoding/...
```

Each point also includes `config.json` plus lightweight `results/summary.json`,
`results/calibrator.json`, and `results/fold_metrics.csv` when available.

## Locked / best recipe (reference)

Typical locked setting used across panels: modalities `gsh`, fusion `attention`
(`cross_qs`), unsigned `pair_balance_num=10000`, similarity `0.3`, seed `1`.

Example path:

```text
checkpoints/01_modality_ablation/mod_gsh/checkpoints/fold1_best.pt
```

Load / evaluate with `code/` (see `code/scripts_eval/`).

```bash
# from repository root
export PYTHONPATH=.
python -m code.main evaluate \
  --config checkpoints/01_modality_ablation/mod_gsh/config.json \
  --checkpoint checkpoints/01_modality_ablation/mod_gsh/checkpoints/fold1_best.pt \
  --gpu 0
```

CV metrics for all points: [`leaderboard.csv`](leaderboard.csv).
