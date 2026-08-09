# runs_ablation_toxin

Hemolytic HC50 toxicity-filter ablation checkpoints (threshold 512 µg/mL).

| Mode | Features | Path |
|------|----------|------|
| `both` | sequence + global | `both/checkpoints/*.joblib` |
| `global` | global only | `global/checkpoints/*.joblib` |
| `sequence` | sequence only | `sequence/checkpoints/*.joblib` |

Each mode includes 5-fold classical-ML (+ MLP) models and CV summary tables.
Labels: `data/trainval_dbassp/toxin/toxicity_labeled_dataset.csv`.

```bash
bash code/scripts_train/run_ablation_toxin.sh
```
