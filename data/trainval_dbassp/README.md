# trainval_dbassp

Packaged **DBAASP-derived train/val data** actually used by:

- `checkpoints/ablation/` (MIC pair model ablations)
- `checkpoints/ablation_toxin/` (HC50 toxin-filter ablations)

Copied from `TIGER/metadata/` (+ toxin outputs / PDB store) for self-contained archival under `TIGER/data/`.

> Folder name keeps the requested spelling `trainval_dbassp` (DBAASP source).

---

## Layout

```text
trainval_dbassp/
  labels/          # tables used in training / eval hooks
  excluded/        # LL37-holdout removals + audit
  toxin/           # labeled set used by toxin ablations
  provenance/      # how holdout tables were built
  pdb/             # Rosetta PDBs (zip) + inventory
  README.md
  MANIFEST.json
```

---

## 1. Labels used by `checkpoints/ablation`

| File | Role |
|------|------|
| `labels/train_val_by_cfu_group_ug_per_mL.csv` | **Primary train/val CSV** in all ablation `config.json` |
| `labels/train_val_ug_per_mL.csv` | Holdout-filtered assay table (pre–CFU expand) |
| `labels/train_val_seq.csv` | Unique sequences kept for train/val |
| `labels/test_LL37_by_cfu_group_ug_per_mL.csv` | Default LL37 test hook in configs |
| `labels/test_LL37_ug_per_mL.csv` | LL37 ug/mL assay rows |
| `labels/features.txt` | AA node feature table (`feature_path`) |

**Train PDBs (source):** `data/3D_data_train_eva_Rosetta/`  
**In this package:** `pdb/trainval_and_excluded_pdbs.zip`

Locked data settings (from `checkpoints/ablation/README.md`):

- Train/val: `metadata/train_val_by_cfu_group_ug_per_mL.csv`
- Holdout: remove train sequences with similarity **> 0.30** to LL37 family  
  (`score / max(|a|,|b|)`)
- Also remove non-standard AA sequences from train/val

---

## 2. Excluded lists (LL37 holdout)

| File | Content |
|------|---------|
| `excluded/removed_similar_to_LL37_seq.csv` | **1122** sequences removed by sim > 0.30 |
| `excluded/removed_similar_to_LL37_ug_per_mL.csv` | Matching assay rows |
| `excluded/removed_nonstandard_AA_seq.csv` | **3739** non-standard-AA sequences |
| `excluded/removed_nonstandard_AA_ug_per_mL.csv` | Matching assay rows |
| `excluded/similarity_filter_audit.csv` | Per-sequence max sim to LL37 + keep/remove flags |
| `excluded/LL37_v0.csv` | LL37 holdout family (112 sequences) |
| `excluded/test_seq.csv` / `test_seq_LL37_named.csv` | LL37 sequence lists |

Builder: `provenance/build_ll37_holdout.py`  
Docs: `provenance/metadata_README.md`

---

## 3. Toxin ablation (`checkpoints/ablation_toxin`)

Toxin runs do **not** use PDB/GNN. Labels come from DBAASP HC50 (human erythrocytes), inequality-aware filter, threshold **512 µg/mL**.

| File | Content |
|------|---------|
| `toxin/toxicity_labeled_dataset.csv` | **1596** peptides used (1152 toxin / 444 non-toxin) |
| `toxin/label_filter_stats.json` | Ambiguous / dropped assay counts |
| `toxin/run_meta_both.json` | Archive meta for `feature_mode=both` |
| `toxin/toxin_filter_README.md` | Method notes |

Upstream JSON source (not copied; large):  
`/data4T/ubuntu/wangyue/postdoc_2025/POAP/Data/dbaasp_jsons`

---

## 4. PDB structures

| File | Content |
|------|---------|
| `pdb/trainval_and_excluded_pdbs.zip` | One PDB per sequence (filename = sequence + `.pdb`) |
| `pdb/pdb_inventory.csv` | Sequence → categories + whether PDB was found |

**Included PDB categories (union):**

- train/val sequences actually used (`train_val_by_cfu_group…`)
- removed by LL37 similarity
- removed by non-standard AA (when a PDB exists)
- LL37 holdout family
- toxin-labeled peptides

**Counts (at packaging time):** 15387 unique sequences → **15371** PDBs zipped; **16** missing (mostly placeholder `X…` / nonstandard stubs).

Unzip example:

```bash
mkdir -p /tmp/trainval_pdbs
unzip -q pdb/trainval_and_excluded_pdbs.zip -d /tmp/trainval_pdbs
```

---

## Notes

- Original paths under `metadata/` and `data/3D_data_train_eva_Rosetta/` are unchanged; this folder is a curated copy.
- CFU-aware train table is the strict table consumed by `checkpoints/ablation` training configs.
