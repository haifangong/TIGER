# LL37-holdout metadata (ug/mL)

## Setting
- **Test family**: sequences from `POAP/4_siamese_mic/metadata/LL37_v0.csv` (112 unique)
- **Train/val source**: `newdata/dbaasp_amp_training_ug_per_mL.csv`
- **Filter**: remove any train/val sequence with length-normalized NW similarity **> 30%** to any LL37 family sequence
- **Standard AA filter (train/val only)**: keep sequences composed solely of the 20 canonical uppercase amino acids `ACDEFGHIKLMNPQRSTVWY` (exclude X/O, D-amino acids/lowercase, digits, etc.)
- **Similarity definition**: Biopython global `PairwiseAligner` (match=1, mismatch=-1, open=-0.5, extend=-0.1); similarity = `score(a,b) / max(len(a), len(b))`. (Replaces the older asymmetric `score/len(query)`, which over-filtered short peptides.)

## Counts
| Split | Rows | Unique sequences |
|---|---:|---:|
| train/val ug/mL | 15881 | 14543 |
| test ug/mL (LL37 rows with labels) | 148 | 112 |
| removed by LL37 similarity | 1402 | 1122 |
| removed by non-standard AA | 6005 | 3739 |
| LL37 family (sequence list) | - | 112 |

## Files
- `LL37_v0.csv` — copied test family table
- `test_seq.csv` — LL37 unique sequences (`sequence,description`)
- `test_seq_LL37_named.csv` — LL37 with id/name
- `test_LL37_ug_per_mL.csv` — ug/mL assay rows for LL37 sequences (from training table)
- `train_val_seq.csv` — filtered train/val sequences
- `train_val_ug_per_mL.csv` — filtered train/val ug/mL assays
- `similarity_filter_audit.csv` — per-sequence max similarity to LL37
- `removed_similar_to_LL37_seq.csv` — sequences removed by the similarity filter
- `removed_similar_to_LL37_ug_per_mL.csv` — assay rows removed by the similarity filter
- `removed_nonstandard_AA_seq.csv` — sequences removed for non-standard residues
- `removed_nonstandard_AA_ug_per_mL.csv` — assay rows removed for non-standard residues

Train/val is kept as one pool for GroupKFold CV (as in the TIGER pipeline).

## CFU files used by training scripts
- `train_val_by_cfu_group_ug_per_mL.csv` — CFU-aware train/val (sequence allowlist ∩ cfu table)
- `test_LL37_by_cfu_group_ug_per_mL.csv` — LL37 test with CFU when available
- Config: `../configs/ll37_holdout_cfu.json` (`eval_external: false`)
