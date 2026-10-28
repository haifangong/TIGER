# TIGER: Therapeutic-Index-Guided Exploration and Refinement for Antimicrobial Peptide Design

Reproducible release of the TIGER computational pipeline for antimicrobial peptide (AMP) design for the work: Deep learning decouples antimicrobial peptide efficacy from host toxicity. This includes mutational search → toxicity filtering → structure prediction, plus a CFU-aware MIC **pair-delta** model.

Repository: [https://github.com/haifangong/TIGER](https://github.com/haifangong/TIGER)

## What is included

```text
.
├── README.md                 # this file
├── code/                     # modular MIC pair-delta model (train / evaluate / infer)
├── metadata/                 # CFU-aware train/val + LL37 holdout CSVs used by configs
├── data/                     # curated sequences, splits, external tests, PDB zip
│   ├── trainval_dbassp/      # DBAASP train/val + toxin labels + Rosetta PDBs (LFS)
│   ├── test_external/        # APEX-GO / LL37 / internal_toxin_cohort eval packs + similarity audit
│   ├── wetlab/               # full-DBAASP per-species production tables
│   ├── ablation_results/     # sequence-encoding ablation CV metrics
│   └── metadata/features.txt # AA node features for the GNN
├── checkpoints/              # released fold*_best.pt / toxin joblibs
│   ├── ablation/             # MIC paper ablations (panels 01–06)
│   ├── ablation_toxin/       # HC50 toxicity classification (both/global/sequence)
│   └── wetlab/               # species×sim production MIC models
└── pipeline/                 # Steps 1–4 discovery + MIC ranking
    ├── 01_mutation_search/
    ├── 02_toxicity_filter/
    ├── 03_structure_prediction/
    ├── 04_metric_ranking/
    ├── common/
    ├── docs/
    └── notebooks/
```

## Clone (Git LFS required)

Train/val Rosetta structures are shipped as a ~104 MB zip via **Git LFS**.

```bash
git lfs install
git clone git@github.com:haifangong/TIGER.git
cd TIGER

# unzip PDBs for code/ training configs
mkdir -p data/3D_data_train_eva_Rosetta
unzip -q data/trainval_dbassp/pdb/trainval_and_excluded_pdbs.zip \
  -d data/3D_data_train_eva_Rosetta
```

Details: [`data/README.md`](data/README.md).

## Ablation checkpoints

Released weights live under [`checkpoints/`](checkpoints/): [`ablation/`](checkpoints/ablation/) (MIC panels 01–06; similarity **0.3 / 0.7**), [`ablation_toxin/`](checkpoints/ablation_toxin/) (HC50 classification), and [`wetlab/`](checkpoints/wetlab/) (species×sim production). Only `fold*_best.pt` / toxin `*.joblib` are shipped. The broken `cross_qs` sequence-only (`mod_s`) run and the sim=0.5 point are **not** redistributed.

```bash
export PYTHONPATH=.
python -m code.main evaluate \
  --config checkpoints/ablation/01_modality_ablation/mod_gsh/config.json \
  --checkpoint checkpoints/ablation/01_modality_ablation/mod_gsh/checkpoints/fold1_best.pt \
  --gpu 0
```

Weights live under [`checkpoints/ablation/`](checkpoints/ablation/), [`checkpoints/ablation_toxin/`](checkpoints/ablation_toxin/), and [`checkpoints/wetlab/`](checkpoints/wetlab/). Index: [`checkpoints/README.md`](checkpoints/README.md); MIC CV table: [`checkpoints/ablation/leaderboard.csv`](checkpoints/ablation/leaderboard.csv).

**CV protocol (MIC):** `fold*_best.pt` is selected on the same validation fold that contributes OOF predictions — report as **model-selection-aware OOF** (not a nested outer test). Primary CV is **raw OOF**; nested leave-one-fold calibration is secondary; the saved calibrator is fit on all OOF for external application only (see `code/train.py`).

**Seeds:** panels 01–04 use **seed=123**; panel 06 uses **seed=1**; panel 05 sweeps **1..5**. See `checkpoints/ablation/MANIFEST.json`.

Sequence-encoding CV tables: [`data/ablation_results/06_seq_encoding/`](data/ablation_results/06_seq_encoding/).  
Toxicity classification checkpoints: [`checkpoints/ablation_toxin/`](checkpoints/ablation_toxin/).

---

## Data, splits, and evaluation packs

Only the tables and structures **actually used** by training / evaluation are redistributed (not the full raw DBAASP dump or complete Rosetta stores). DBAASP collection snapshot: **2026-01-06**.

### 1. Train / val (`metadata/` + `data/trainval_dbassp/`)

CFU-aware MIC tables with an **LL37-family holdout**:

| File | Role |
|------|------|
| `metadata/train_val_by_cfu_group_ug_per_mL.csv` | Primary train/val CSV (~10.5k unique sequences) |
| `metadata/test_LL37_by_cfu_group_ug_per_mL.csv` | Default LL37 external test hook in configs |
| `metadata/removed_similar_to_LL37_*.csv` | **1,122** unique peptides / **1,402** assay rows removed (sim > 0.30 to LL37) |
| `metadata/removed_nonstandard_AA_*.csv` | **3739** non-standard-AA sequences removed |
| `metadata/similarity_filter_audit.csv` | Per-sequence max similarity + keep/remove flags |
| `metadata/LL37_v0.csv` / `test_seq_LL37_named.csv` | LL37 holdout family (**112** named sequences used for filtering) |

Holdout rule: length-normalized Needleman–Wunsch similarity `score / max(|a|,|b|) > 0.30` to the LL37 family, plus removal of non-standard amino acids.

Full archive (labels, excluded lists, toxin set, PDB zip + inventory): [`data/trainval_dbassp/`](data/trainval_dbassp/).

**Toxin ablation labels** (`data/trainval_dbassp/toxin/`): **1596** peptides (1152 toxin / 444 non-toxin), HC50 on human erythrocytes, threshold **512 µg/mL**.

**PDB archive:** `data/trainval_dbassp/pdb/trainval_and_excluded_pdbs.zip` — 15371 structures (Git LFS). Unzip to `data/3D_data_train_eva_Rosetta/` as shown above.

### 2. External / holdout evaluation packs (`data/test_external/`)

**LL37 and APEXGO are different protocols — do not lump them as a single “strict unseen-template” set.**

Always report **both sequence count and pair count** for these activity packs (do not cite pairs alone):

| Pack | Sequences | Eval pairs | Role |
|------|----------:|-----------:|------|
| LL37 (`test_activity_ll37/`) | **68** | **509** | similarity-filtered family holdout (DBAASP-derived) |
| APEXGO (`test_activity_apexgo/`) | **110** | **200** | sequence-disjoint external family panel |

#### LL37 — similarity-filtered family holdout (DBAASP-derived)

LL37 is **not** an independent external source. It is a **family holdout carved from the DBAASP-derived pool**:

- **Eval size:** **68 sequences** + **509 neighbor pairs** (*E. coli* MIC; `ll37_sequences_mic.csv` / `ll37_pairs_neighbor.csv`).
- Holdout family used for filtering: **112** named LL37-family sequences (`metadata/test_seq_LL37_named.csv`; also `metadata/LL37_v0.csv`).
- From train/val we removed every peptide with length-normalized NW similarity `> 0.30` to that family: **1,122 unique peptides** / **1,402 assay rows** (`metadata/removed_similar_to_LL37_*.csv`).
- Audit (`test_pair_similarity_compare/`): every LL37 eval sequence has `max_sim_to_train ≤ 0.30` (`frac_gt_0.3 = 0`).

#### APEXGO — sequence-disjoint external family panel (with reported train similarity)

APEXGO is the closer match to a true external family-structured evaluation:

- **Eval size:** **110 sequences** (10 families × WT + 10 variants) + **200 template↔variant pairs** (both directions; `apexgo_peptides.csv` / `pairs/pairs_template_centric_alldelta_geo3.csv`).
- Sequences are disjoint from the train table, **but not <30% similarity-disjoint**.
- Audit: **≈49.1%** of APEXGO sequences have `max_sim_to_train > 0.30` (max **0.4875**). Prefer the wording *sequence-disjoint external family panel with an explicitly reported train-similarity distribution*, not “strict template-disjoint”.

#### Other packs

| Pack | Task | Size |
|------|------|------|
| `test_toxin_internal_toxin_cohort` | Hemolytic HC50 (labels + predictions) | **88 sequences** |
| `test_pair_similarity_compare` | Test-vs-train similarity audit (LL37 & APEXGO) | — |

See [`data/test_external/README.md`](data/test_external/README.md). TIGER vs EvoGradient pair-delta comparison archive: [`checkpoints/ablation/04_similarity/external_eval_tiger_vs_evo/`](checkpoints/ablation/04_similarity/external_eval_tiger_vs_evo/).

### 3. Wet-lab / production species tables (`data/wetlab/` + `checkpoints/wetlab/`)

Full-DBAASP per-species tables (**no** LL37 holdout) for production-style training.
Matching **12** production MIC models (6 species × sim `{0.3, 0.7}`, five `fold*_best.pt` each) are shipped under [`checkpoints/wetlab/`](checkpoints/wetlab/) for wet-lab ranking ensembles.

- `train_Escherichia_coli.csv`
- `train_Pseudomonas_aeruginosa.csv`
- `train_Staphylococcus_aureus.csv`
- `train_Staphylococcus_epidermidis.csv`
- `train_Klebsiella_pneumoniae.csv`
- `train_Bacillus_subtilis.csv`

Each remaps that species’ MIC into the default `MIC_Escherichia_coli` column used by TIGER. See [`data/wetlab/README.md`](data/wetlab/README.md).

### Path mapping for `code/` configs

| Config key | Path in this repo |
|---|---|
| `train_csv` / `test_csv` | `metadata/*.csv` |
| `feature_path` | `data/metadata/features.txt` |
| `train_pdb_dir` / `test_pdb_dir` | `data/3D_data_train_eva_Rosetta/` (after unzip) |

---

## Quick start

### A) Discovery pipeline (Steps 1–3)

```bash
cd pipeline
conda env create -f environment.yml
conda activate tiger-pipeline

# Step 1 — mutational search
cd 01_mutation_search
python search_mutations.py --sequence KSMLKSMK --search_length 1 --output_dir outputs/demo

# Step 2 — toxicity filter
cd ../02_toxicity_filter
python infer_toxin.py --csv ../01_mutation_search/outputs/demo/KSMLKSMK_positive_1.csv --out-dir outputs/demo

# Step 3 — structure prediction (separate HelixFold env; see pipeline/docs/SETUP.md)
cd ../03_structure_prediction
python infer_batch.py --csv_file ../02_toxicity_filter/outputs/demo/KSMLKSMK_non_toxins.csv --output_dir outputs/pdb/demo
```

Full walkthrough:

- Environment setup: [`pipeline/docs/SETUP.md`](pipeline/docs/SETUP.md)
- Custom / user-defined data: [`pipeline/docs/CUSTOM_DATA.md`](pipeline/docs/CUSTOM_DATA.md)
- End-to-end notebook: [`pipeline/notebooks/00_full_pipeline_demo.ipynb`](pipeline/notebooks/00_full_pipeline_demo.ipynb)
- Pipeline overview: [`pipeline/README.md`](pipeline/README.md)

### B) Pair-delta MIC model (`code/`)

```bash
# from the repository root
export PYTHONPATH=.
python -m code.main train --config code/configs/default_fusion_attention.json --gpu 0
```

Paper-oriented launchers:

```bash
bash code/scripts_train/train_best_recipe.sh          # locked MIC recipe
bash code/scripts_train/run_ablation_all_mic.sh       # MIC ablation panels
bash code/scripts_train/run_ablation_toxin.sh         # toxin both/global/sequence

bash code/scripts_eval/eval_ll37_apexgo_best.sh       # LL37 68 seq/509 pairs + APEXGO 110 seq/200 pairs
bash code/scripts_eval/eval_toxin_internal_toxin_cohort.sh           # hemolytic internal_toxin_cohort (n=88)
bash code/scripts_eval/eval_cv_summary.sh             # print CV summary.json
```

See [`code/README.md`](code/README.md), [`code/scripts_train/README.md`](code/scripts_train/README.md), and [`code/scripts_eval/README.md`](code/scripts_eval/README.md).

## Classification metrics

For antimicrobial-activity and toxicity classifiers, report the full suite across CV folds and the held-out test set:

**Accuracy, Precision, Recall (toxic sensitivity), Specificity, F1, MCC, AUC-ROC, AUC-PR**,
plus **confusion matrix** and **false-safe rate** (= false-negative rate among true toxics).

The 88-peptide hemolytic panel writes these under
`data/test_external/test_toxin_internal_toxin_cohort/internal_toxin_cohort_prediction_summary.json`
and `internal_toxin_cohort_safety_metrics.csv` (includes sequence-overlap audit vs the toxin train table).

```bash
cd pipeline/01_mutation_search
python evaluate_models.py --csv examples/labeled_activity_demo.csv --out-dir outputs/metrics

cd ../02_toxicity_filter
python evaluate_models.py --csv examples/labeled_toxicity_demo.csv --out-dir outputs/metrics
```

Pair-delta MIC evaluation reports `log10MAE`, `RSE`, `PCC`, `KCC` (and related metrics) on neighbor / template-centric pairs.

## Notes

- HelixFold weights (`~4.5 GB`) are **not** vendored. Download or symlink into `pipeline/03_structure_prediction/weights/` (see pipeline Step-3 README).
- Steps 1–2 and Step 3 intentionally use separate conda environments because PaddlePaddle / PyRosetta stacks conflict with the PyTorch scientific stack.
- `trainval_and_excluded_pdbs.zip` is stored with **Git LFS**; clone with `git lfs install` first.
- Raw unfiltered DBAASP dumps and full Rosetta directories are not redistributed—only the filtered tables and structures used in the experiments.

## Citation

If you use this repository, please cite the corresponding TIGER study.
