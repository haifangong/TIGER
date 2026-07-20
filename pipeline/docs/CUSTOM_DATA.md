# Using custom (user-defined) data

This guide explains how to adapt each pipeline stage to **your own sequences / labels**, instead of the bundled demo CSVs.

## Overview of data contracts

```text
Your template sequence(s)
        │
        ▼
[01] search_mutations.py
        │  positive CSV  (headerless: sequence,sequence)
        ▼
[02] infer_toxin.py
        │  non_toxins.csv  (header: sequence,tox_prob,...)
        ▼
[03] infer_batch.py / relax.py
        │  <SEQUENCE>.pdb  (and optional relaxed PDBs)
        ▼
Downstream ranking / wet-lab selection
```

For **model evaluation** (classification metrics), provide labeled CSVs:

| Task | Required columns | Example file |
|------|------------------|--------------|
| Activity classifier | `sequence`, `label` (0/1) | `01_mutation_search/examples/labeled_activity_demo.csv` |
| Toxicity classifier | `sequence`, `label` (0/1) | `02_toxicity_filter/examples/labeled_toxicity_demo.csv` |

`label=1` means the positive class (active AMP / toxic peptide).

---

## Step 1 — Mutational search on a custom template

No hardcoded template is required. Pass your sequence on the CLI:

```bash
cd 01_mutation_search
conda activate tiger-pipeline

python search_mutations.py \
  --sequence YOURPEPTIDESEQUENCE \
  --search_length 2 \
  --output_dir outputs/my_template \
  --model models/R_catboost_model.pkl
```

### What is hardcoded vs configurable?

| Item | Status | How to change |
|------|--------|---------------|
| Template sequence | CLI `--sequence` | pass your AA string |
| Mutation depth `k` | CLI `--search_length` | start with 1–2 |
| Activity model | CLI `--model` | point to another `.pkl` |
| Physicochemical filters | in `search_mutations.py` | edit thresholds if needed |
| Training CSV for the bundled CatBoost | external / historical | retrain via `evaluate_models.py --save-models` |

### Output for custom runs

- `outputs/my_template/<SEQ>_positive_<k>.csv`
- `outputs/my_template/<SEQ>_negative_<k>.csv`

Positive CSV format is **headerless** with two columns: `sequence,sequence` (second column is a duplicate for legacy compatibility).

### Evaluate / retrain classifiers on custom labeled data

```bash
python evaluate_models.py \
  --csv /path/to/your_labeled_activity.csv \
  --sequence-col sequence \
  --label-col label \
  --n-splits 5 \
  --test-size 0.2 \
  --out-dir outputs/metrics_custom \
  --save-models
```

This writes:

- `cv_fold_metrics.csv` — Acc/P/R/F1/MCC/AUC-ROC/AUC-PR for **each fold × model**
- `test_metrics.csv` — same suite on the **held-out test set**
- `summary_metrics.csv` — CV mean/std + test

If your CSV uses different column names:

```bash
python evaluate_models.py --csv my.csv --sequence-col Seq --label-col is_amp ...
```

---

## Step 2 — Toxicity filter on custom candidates

### A) Score unlabeled candidates (inference)

```bash
cd 02_toxicity_filter
python infer_toxin.py \
  --csv /path/to/candidates.csv \
  --out-dir outputs/my_run \
  --threshold 0.5 \
  --model-path checkpoints/model_1.pth
```

**Accepted input formats**

1. Headerless Step-1 output: `SEQ,SEQ`
2. Any CSV whose first column is the amino-acid sequence (the slim dataset reads column 0)

**Configurable**

| Flag | Meaning |
|------|---------|
| `--csv` | your candidate list |
| `--threshold` | toxin probability cutoff (default 0.5) |
| `--model-path` | your checkpoint |
| `--max-len` | skip longer peptides (default 30) |
| `--cpu` | force CPU |

### B) Evaluate on custom labeled toxicity data

```bash
python evaluate_models.py \
  --csv /path/to/your_labeled_toxicity.csv \
  --sequence-col sequence \
  --label-col label \
  --out-dir outputs/metrics_custom \
  --eval-pretrained   # also scores the bundled FusionPeptide checkpoint
```

---

## Step 3 — Structures for custom sequences

### From Step-2 non-toxin CSV

```bash
cd 03_structure_prediction
conda activate tiger-helix   # or helix
python infer_batch.py \
  --csv_file ../02_toxicity_filter/outputs/my_run/XXX_non_toxins.csv \
  --output_dir outputs/pdb/my_run \
  --init_model weights/helixfold-single.pdparams
```

### From an arbitrary user CSV

Create a CSV with a `sequence` column (recommended) or headerless first-column sequences:

```csv
sequence,description
ACDEFGHIKL,ACDEFGHIKL
KRIVQRIKDFLR,KR12
```

```bash
python infer_batch.py --csv_file /path/to/my_sequences.csv --output_dir outputs/pdb/custom
python infer_single.py --seq KRIVQRIKDFLR --output_dir outputs/pdb/custom
```

### Relax

```bash
conda activate tiger-rosetta   # or rosetta
python relax.py \
  --input_dir outputs/pdb/custom \
  --output_dir outputs/relaxed/custom \
  --workers 4
```

---

## Minimal custom end-to-end recipe

```bash
TEMPLATE=GRKKRRQRRRPPQ
K=1

# 1) mutate + activity filter
cd 01_mutation_search
python search_mutations.py -s $TEMPLATE -k $K -o outputs/custom

# 2) toxicity filter
cd ../02_toxicity_filter
python infer_toxin.py \
  --csv ../01_mutation_search/outputs/custom/${TEMPLATE}_positive_${K}.csv \
  --out-dir outputs/custom

# 3) structures (HelixFold env)
cd ../03_structure_prediction
python infer_batch.py \
  --csv_file ../02_toxicity_filter/outputs/custom/${TEMPLATE}_non_toxins.csv \
  --output_dir outputs/pdb/custom_${TEMPLATE}
```

See also: `notebooks/00_full_pipeline_demo.ipynb`.
