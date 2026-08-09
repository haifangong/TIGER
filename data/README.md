# Data

Curated sequences, train/val splits, external test packs, and related audit
tables actually used by the TIGER MIC / toxicity experiments.

```text
data/
├── metadata/
│   └── features.txt                 # AA node features for the GNN
├── trainval_dbassp/                 # DBAASP train/val + holdout + toxin labels + PDBs
│   ├── labels/                      # CFU-aware train/val + LL37 test hooks
│   ├── excluded/                    # LL37-similarity / nonstandard-AA removals
│   ├── toxin/                       # HC50-labeled set for toxin ablations
│   ├── provenance/                  # holdout builder + notes
│   └── pdb/                         # Rosetta PDB zip + inventory
├── test_external/                   # external holdouts for final pipeline eval
│   ├── test_activity_apexgo/        # E. coli MIC pairs (APEX-GO)
│   ├── test_activity_ll37/          # E. coli MIC pairs (LL37 family)
│   ├── test_toxin_qlx227/           # hemolytic HC50 labels
│   └── test_pair_similarity_compare/# test-vs-train similarity audit
└── wetlab/                          # full-DBAASP per-species production tables
```

## Path mapping for `code/` configs

Default configs expect:

| Config key | Path in this repo |
|---|---|
| `train_csv` / `test_csv` | `metadata/*.csv` (copies of `trainval_dbassp/labels/`) |
| `feature_path` | `data/metadata/features.txt` |
| `train_pdb_dir` / `test_pdb_dir` | unzip PDBs (see below) |

```bash
# from repository root
mkdir -p data/3D_data_train_eva_Rosetta
unzip -q data/trainval_dbassp/pdb/trainval_and_excluded_pdbs.zip \
  -d data/3D_data_train_eva_Rosetta
```

External activity PDBs for APEX-GO / LL37 already live under each
`test_external/test_activity_*/pdb/` folder.

## Notes

- Raw full Rosetta stores and unfiltered DBAASP dumps are **not** redistributed;
  only the filtered tables and PDB zip used by training / eval are included.
- `trainval_and_excluded_pdbs.zip` is stored with **Git LFS** (~104 MB).
- DBAASP collection snapshot used for these tables: **2026-01-06**.
