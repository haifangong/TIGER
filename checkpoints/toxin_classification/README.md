# toxin_classification — hemolytic HC50 filter checkpoints

Final 5-fold classical-ML (+ MLP) checkpoints from `runs_ablation_toxin/` for
feature modes `both` / `global` / `sequence` (threshold 512 µg/mL).

```text
toxin_classification/
├── both/checkpoints/*.joblib
├── global/checkpoints/*.joblib
├── sequence/checkpoints/*.joblib
└── README.md
```

Each mode also includes CV summary tables (`summary_metrics.csv`,
`cv_fold_metrics.csv`, `run_meta.json`). Labeled peptides used for training are
in `data/trainval_dbassp/toxin/toxicity_labeled_dataset.csv`.

Reproduce:

```bash
bash code/scripts_train/run_ablation_toxin.sh
```
