# Activity filter (MIC binary classification, deep learning)

Sequence + global physicochemical features only — **same feature stack as**
`toxin_filter` DL models (no structure / PDB / GNN).

## Label

Inequality-aware binary labels from DBAASP JSON `targetActivities` (MIC / MIC50 / MIC90):

| Assay form | Rule (threshold `T`, default **128 µg/mL**) |
|---|---|
| exact `x` | inactive (`1`) if `x >= T`, else active (`0`) |
| `>c` / `≥c` | inactive only if `c >= T`; otherwise **drop** |
| `<c` / `≤c` | active only if `c <= T`; otherwise **drop** |
| range `[a,b]` | inactive if `a >= T`; active if `b < T`; else **drop** |

Per peptide, exact MICs are aggregated by **minimum** (best potency).  
Positive class `label=1` = **inactive** (filter-out MIC ≥ 128).  
Sequences with non-standard amino acids or non-monomer complexity are removed.

## Models (5-fold stratified CV)

Deep learning only (reuses `toxin_filter` architectures):

- `fusion_seq_glob` — BiGRU sequence + global MLP fusion
- `tiger_seq_glob` — integer sequence encoder + global encoder + attention fusion
- `metric_learning` — same backbone with supervised contrastive loss + BCE head

## Run

From the `TIGER/` root:

```bash
# Full 5-fold CV on all DBAASP MIC peptides (JSON)
PYTHONPATH=. /home/ubuntu/anaconda3/envs/ccseg/bin/python -m code.activity_filter \
  --gpu 0 --threshold 128

# Subset of models / smoke test
PYTHONPATH=. /home/ubuntu/anaconda3/envs/ccseg/bin/python -m code.activity_filter \
  --gpu 0 --models tiger_seq_glob --epochs 5 --n-splits 2

# Numeric CSV fallback (inequalities already stripped)
PYTHONPATH=. /home/ubuntu/anaconda3/envs/ccseg/bin/python -m code.activity_filter \
  --source csv --csv newdata/dbaasp_amp_training_ug_per_mL.csv --gpu 0
```

## Outputs

Default directory: `outputs/outputs_activity_filter_mic128/`

| File | Content |
|------|---------|
| `activity_labeled_dataset.csv` | Filtered labeled peptides |
| `label_filter_stats.json` | Class balance + parse stats |
| `cv_fold_metrics.csv` | Per-fold Acc / P / R / F1 / MCC / AUC-ROC / AUC-PR |
| `summary_metrics.csv` / `summary_mean_std_pretty.csv` | CV mean±std |
| `checkpoints/{model}_fold{k}.pt` | Per-fold DL weights + z-score stats + decision threshold |
| `run_meta.json` | Run configuration |
