# Step 1 — Mutational Search & Activity Pre-filter

Enumerate physicochemical-filtered mutants of a template antimicrobial peptide (AMP) and classify them with a CatBoost activity model.

## Purpose

- Expand a template peptide into a local mutational neighborhood
- Discard mutants that violate simple physicochemical constraints
- Keep only candidates predicted as antimicrobial-active for downstream toxicity filtering

## Folder structure

```text
01_mutation_search/
├── README.md
├── demo.ipynb
├── requirements.txt
├── search_mutations.py      # main entry point
├── models/
│   └── R_catboost_model.pkl # pretrained CatBoost classifier
├── examples/
│   └── sample_positive.csv  # tiny example output
└── outputs/                 # default runtime outputs
```

## Environment

Recommended conda env (from original POAP notes): `ccseg`

```bash
conda activate ccseg
pip install -r requirements.txt
```

## Quick start

```bash
cd 01_mutation_search
python search_mutations.py -s KSMLKSMK -k 2 -o outputs
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-s / --sequence` | Template peptide sequence | `KSMLKSMPMTLK` |
| `-k / --search_length` | Number of mutated positions | `3` |
| `-o / --output_dir` | Output directory | `outputs` |
| `--model` | CatBoost model path | `models/R_catboost_model.pkl` |

## Outputs

For template `SEQ` and mutation depth `k`:

| File | Content |
|------|---------|
| `SEQ_positive_k.csv` | Predicted active mutants (headerless: `seq,seq`) |
| `SEQ_negative_k.csv` | Predicted inactive mutants |

The positive CSV is the input to **Step 2**.

## Notes

- Search space grows quickly with sequence length and `k`. Start with short templates / `k=1` or `k=2` for demos.
- Physicochemical filters applied during enumeration include GRAVY, aliphatic index, aromatic fraction, and DIWV-based stability heuristics.
- Interactive walkthrough: open `demo.ipynb`.

## Source

Cleaned from POAP `1_search_mutation_space/search_infer_fast.py`.


## Classification metrics (CV + test)

To report Accuracy, Precision, Recall, F1, MCC, AUC-ROC, and AUC-PR for all activity models:

```bash
python evaluate_models.py \
  --csv examples/labeled_activity_demo.csv \
  --sequence-col sequence \
  --label-col label \
  --n-splits 5 \
  --test-size 0.2 \
  --out-dir outputs/metrics_demo
```

For **custom labeled data**, keep the same two columns (or rename via flags).  
Outputs: `cv_fold_metrics.csv`, `test_metrics.csv`, `summary_metrics.csv`.

See also `../docs/CUSTOM_DATA.md`.
