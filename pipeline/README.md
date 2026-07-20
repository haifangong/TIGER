# TIGER Pipeline

Reproducible packaging of the early TIGER / POAP workflow:

1. **Mutational search + antimicrobial activity pre-filter**
2. **Hemolytic / host-toxicity filter**
3. **3D structure prediction (HelixFold-Single) + optional Rosetta FastRelax**

This directory is intentionally organized for community reuse: clear step folders, environment files, English READMEs, CLI entry points with explicit arguments, classification-metric reporting, and Jupyter demos for both per-step and full-pipeline runs.

---

## Repository layout

```text
pipeline/
├── README.md                      # this file
├── requirements.txt               # pip deps for Steps 1–2
├── environment.yml                # conda env for Steps 1–2 + notebooks
├── environment_helix.yml          # HelixFold scaffold env
├── environment_rosetta.yml        # Rosetta scaffold env
├── docs/
│   ├── SETUP.md                   # environment setup details
│   └── CUSTOM_DATA.md             # how to run on user-defined data
├── notebooks/
│   └── 00_full_pipeline_demo.ipynb
├── common/
│   └── metrics.py                 # Acc/P/R/F1/MCC/AUC-ROC/AUC-PR helpers
├── 01_mutation_search/
├── 02_toxicity_filter/
└── 03_structure_prediction/
```

Each numbered step folder contains:

| File / folder | Role |
|---------------|------|
| `README.md` | Exact CLI usage, I/O contracts, custom-data notes |
| `demo.ipynb` | Interactive minimal example |
| `requirements.txt` | Step-local dependency pin hints |
| `examples/` | Tiny sample inputs (and labeled demo CSVs where relevant) |
| `outputs/` | Runtime outputs (gitignored) |
| Core `*.py` | Runnable entry points (no hidden hard-coded paths required) |

---

## 1. Environment setup (required reading)

**Detailed instructions:** [`docs/SETUP.md`](docs/SETUP.md)

Quick start for Steps 1–2:

```bash
cd TIGER/pipeline
conda env create -f environment.yml
conda activate tiger-pipeline
python -c "import catboost, torch, Bio, sklearn; print('ok')"
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Step 3 uses **separate** environments (`tiger-helix` / `tiger-rosetta`) because HelixFold (PaddlePaddle) and PyRosetta have conflicting stacks. See `03_structure_prediction/README.md`.

---

## 2. What each script does (and what is / is not hardcoded)

### Step 1 — `01_mutation_search/search_mutations.py`

| Question | Answer |
|----------|--------|
| Do I need arguments? | **Yes.** Provide `--sequence`, optionally `-k`, `-o`, `--model`. |
| Is the data source hardcoded? | **No** for inference. The template comes from `--sequence`. |
| Where does the model come from? | Bundled `models/R_catboost_model.pkl` (override with `--model`). |
| Custom data? | Pass any AA template; for metrics use `evaluate_models.py --csv your_labels.csv`. |

```bash
cd 01_mutation_search
python search_mutations.py --sequence KSMLKSMK --search_length 1 --output_dir outputs/demo
```

### Step 2 — `02_toxicity_filter/infer_toxin.py`

| Question | Answer |
|----------|--------|
| Do I need arguments? | **Yes.** `--csv` is required. |
| Is the data source hardcoded? | **No.** Candidates come from `--csv` (Step-1 positives or your own list). |
| Where does the model come from? | `checkpoints/model_1.pth` (override with `--model-path`). |
| Custom data? | Point `--csv` at your sequences; for metrics use `evaluate_models.py`. |

```bash
cd 02_toxicity_filter
python infer_toxin.py --csv examples/sample_input.csv --out-dir outputs/demo
```

### Step 3 — `03_structure_prediction/infer_batch.py` (+ `relax.py`)

| Question | Answer |
|----------|--------|
| Do I need arguments? | **Yes.** `--csv_file` / `--output_dir` (and weights path). |
| Is the data source hardcoded? | **No.** Sequences come from your CSV. |
| Where do weights come from? | `weights/helixfold-single.pdparams` (symlink or download; ~4.5 GB). |
| Custom data? | Any CSV with a `sequence` column (or headerless first-column sequences). |

```bash
cd 03_structure_prediction
python infer_batch.py --csv_file examples/sample_sequences.csv --output_dir outputs/pdb/demo
python relax.py --input_dir outputs/pdb/demo --output_dir outputs/relaxed/demo
```

**Full custom-data walkthrough:** [`docs/CUSTOM_DATA.md`](docs/CUSTOM_DATA.md)

---

## 3. Classification metrics (CV folds + test set)

Reviewers / users should report the full suite for classifier comparisons:

**Accuracy, Precision, Recall, F1, MCC, AUC-ROC, AUC-PR**

### Activity models (Step 1)

```bash
cd 01_mutation_search
python evaluate_models.py \
  --csv examples/labeled_activity_demo.csv \
  --sequence-col sequence \
  --label-col label \
  --n-splits 5 \
  --test-size 0.2 \
  --out-dir outputs/metrics_demo
```

Outputs:

- `cv_fold_metrics.csv` — metrics for **each model × each CV fold**
- `test_metrics.csv` — metrics on the **held-out test set** for all models
- `summary_metrics.csv` — CV mean/std + test

### Toxicity models (Step 2)

```bash
cd 02_toxicity_filter
python evaluate_models.py \
  --csv examples/labeled_toxicity_demo.csv \
  --out-dir outputs/metrics_demo \
  --eval-pretrained
```

`--eval-pretrained` additionally scores the bundled FusionPeptide checkpoint with the same metric suite.

Shared metric implementation: `common/metrics.py`.

---

## 4. End-to-end demos

| Demo | Path |
|------|------|
| Full pipeline notebook | [`notebooks/00_full_pipeline_demo.ipynb`](notebooks/00_full_pipeline_demo.ipynb) |
| Step 1 notebook | `01_mutation_search/demo.ipynb` |
| Step 2 notebook | `02_toxicity_filter/demo.ipynb` |
| Step 3 notebook | `03_structure_prediction/demo.ipynb` |
| Shell smoke tests | each step’s `run_demo.sh` |

---

## 5. Recommended full command sequence

```bash
conda activate tiger-pipeline

# Step 1
cd 01_mutation_search
python search_mutations.py -s KSMLKSMK -k 1 -o outputs/e2e

# Step 2
cd ../02_toxicity_filter
python infer_toxin.py \
  --csv ../01_mutation_search/outputs/e2e/KSMLKSMK_positive_1.csv \
  --out-dir outputs/e2e

# Step 3a
cd ../03_structure_prediction
conda activate tiger-helix   # or helix
python infer_batch.py \
  --csv_file ../02_toxicity_filter/outputs/e2e/KSMLKSMK_non_toxins.csv \
  --output_dir outputs/pdb/e2e

# Step 3b
conda activate tiger-rosetta  # or rosetta
python relax.py \
  --input_dir outputs/pdb/e2e \
  --output_dir outputs/relaxed/e2e
```

---

## 6. Source mapping

Cleaned and reorganized from `/data4T/ubuntu/gonghaifan/POAP`:

| Pipeline step | Original POAP folder |
|---------------|----------------------|
| `01_mutation_search` | `1_search_mutation_space` |
| `02_toxicity_filter` | `2_hemo_docoupling` |
| `03_structure_prediction` | `3_helixfold` |

Training-only / experiment-dump scripts, huge intermediate scan outputs, and virtualenvs were intentionally excluded.
