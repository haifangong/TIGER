# TIGER: Therapeutic-Index-Guided Exploration and Refinement for Antimicrobial Peptide Design

Reproducible release of the TIGER computational pipeline for antimicrobial peptide (AMP) design: mutational search → toxicity filtering → structure prediction, plus a CFU-aware MIC **pair-delta** model.

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
├── checkpoints/              # MIC fold*_best.pt + toxin classification joblibs
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

Final selected MIC ablation weights are under [`checkpoints/`](checkpoints/) (**`fold{1..5}_best.pt`**, including completed **sequence-encoding** `integer` / `embedding` / `onehot` with 5 folds each). Toxicity HC50 classification joblibs are under `checkpoints/toxin_classification/`. `fold*_last`, wetlab production models, and other training artifacts are not uploaded.

```bash
export PYTHONPATH=.
python -m code.main evaluate \
  --config checkpoints/01_modality_ablation/mod_gsh/config.json \
  --checkpoint checkpoints/01_modality_ablation/mod_gsh/checkpoints/fold1_best.pt \
  --gpu 0
```

See [`checkpoints/README.md`](checkpoints/README.md) and [`checkpoints/leaderboard.csv`](checkpoints/leaderboard.csv).

Sequence-encoding CV tables: [`data/ablation_results/06_seq_encoding/`](data/ablation_results/06_seq_encoding/).  
Toxicity classification checkpoints: [`checkpoints/toxin_classification/`](checkpoints/toxin_classification/).

---

## Data, splits, and evaluation packs

Only the tables and structures **actually used** by training / evaluation are redistributed (not the full raw DBAASP dump or complete Rosetta stores). DBAASP collection snapshot: **2026-01-06**.

### 1. Train / val (`metadata/` + `data/trainval_dbassp/`)

CFU-aware MIC tables with an **LL37-family holdout**:

| File | Role |
|------|------|
| `metadata/train_val_by_cfu_group_ug_per_mL.csv` | Primary train/val CSV (~10.5k unique sequences) |
| `metadata/test_LL37_by_cfu_group_ug_per_mL.csv` | Default LL37 external test hook in configs |
| `metadata/removed_similar_to_LL37_*.csv` | **1122** sequences removed (sim > 0.30 to LL37) |
| `metadata/removed_nonstandard_AA_*.csv` | **3739** non-standard-AA sequences removed |
| `metadata/similarity_filter_audit.csv` | Per-sequence max similarity + keep/remove flags |
| `metadata/LL37_v0.csv` | LL37 holdout family (112 sequences) |

Holdout rule: length-normalized Needleman–Wunsch similarity `score / max(|a|,|b|) > 0.30` to the LL37 family, plus removal of non-standard amino acids.

Full archive (labels, excluded lists, toxin set, PDB zip + inventory): [`data/trainval_dbassp/`](data/trainval_dbassp/).

**Toxin ablation labels** (`data/trainval_dbassp/toxin/`): **1596** peptides (1152 toxin / 444 non-toxin), HC50 on human erythrocytes, threshold **512 µg/mL**.

**PDB archive:** `data/trainval_dbassp/pdb/trainval_and_excluded_pdbs.zip` — 15371 structures (Git LFS). Unzip to `data/3D_data_train_eva_Rosetta/` as shown above.

### 2. External test packs (`data/test_external/`)

| Pack | Task | Size | Primary files |
|------|------|------|---------------|
| `test_activity_apexgo` | *E. coli* MIC pairs | 110 seq / 200 pairs | `apexgo_peptides.csv`, `pairs/…`, `pdb/` |
| `test_activity_ll37` | *E. coli* MIC pairs | 68 seq / 509 pairs | `ll37_sequences_mic.csv`, `ll37_pairs_neighbor.csv`, `pdb/` |
| `test_toxin_internal_toxin_cohort` | Hemolytic HC50 | 88 seq | `internal_toxin_cohort_hemolysis_active_micmin_le128.csv` |
| `test_pair_similarity_compare` | Test-vs-train similarity audit | — | `*_similarity*.csv`, PDF figure |

See [`data/test_external/README.md`](data/test_external/README.md).

### 3. Wet-lab / production species tables (`data/wetlab/`)

Full-DBAASP per-species tables (**no** LL37 holdout) for production-style training:

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

bash code/scripts_eval/eval_ll37_apexgo_best.sh       # LL37-509 + APEX-GO-200
bash code/scripts_eval/eval_toxin_internal_toxin_cohort.sh           # hemolytic internal_toxin_cohort (n=88)
bash code/scripts_eval/eval_cv_summary.sh             # print CV summary.json
```

See [`code/README.md`](code/README.md), [`code/scripts_train/README.md`](code/scripts_train/README.md), and [`code/scripts_eval/README.md`](code/scripts_eval/README.md).

## Classification metrics

For antimicrobial-activity and toxicity classifiers, report the full suite across CV folds and the held-out test set:

**Accuracy, Precision, Recall, F1, MCC, AUC-ROC, AUC-PR**

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
