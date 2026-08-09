# Toxicity filter (HC50 binary classification + metric learning)

Sequence + global physicochemical features only (no structure / PDB / GNN).

## Label

Inequality-aware binary labels from DBAASP JSON (Human erythrocytes, HC50-like):

- exact `x`: toxin if `x <= T`
- `>c` / `≥c`: non-toxin only if `c >= T`; otherwise **drop** (ambiguous)
- `<c` / `≤c`: toxin only if `c <= T`; otherwise **drop**
- range `[a,b]`: toxin if `b <= T`; non-toxin if `a > T`; else **drop**

Default threshold `T = 512` µg/mL. Sequences with non-standard amino acids are removed.

## Models

Classical ML (tabular: length + 10 global props + 20 AA freqs, z-scored per fold),
matching the manuscript legend:

- CatBoost, LGBM, RF, SVM, GB, XGB, MLP, Adaboost, LR

Deep learning (sequence + z-scored 10-D global features):

- `fusion_seq_glob` — POAP/TIGER toxicity FusionPeptide mode=101 (BiGRU + global MLP)
- `tiger_seq_glob` — TIGER-style integer sequence encoder + global encoder + attention fusion
- `metric_learning` — same backbone with supervised contrastive loss + BCE head

## Run

From the `TIGER/` root:

```bash
PYTHONPATH=. /home/ubuntu/anaconda3/envs/class/bin/python -m code.toxin_filter.run --gpu 0

# classical ML only (legend methods)
PYTHONPATH=. /home/ubuntu/anaconda3/envs/class/bin/python -m code.toxin_filter.run --skip-dl
```

Outputs land in `outputs_toxin_filter/`:

- `cv_fold_metrics.csv` — per-fold Acc / P / R / F1 / MCC / AUC-ROC / AUC-PR
- `summary_metrics.csv` — CV mean and std
- `toxicity_labeled_dataset.csv` — filtered labeled peptides
