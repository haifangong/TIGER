# metadata

Primary train/val and LL37-holdout tables expected by `code/` configs
(`train_csv`, `test_csv`). These files are copies of
`data/trainval_dbassp/labels/` and the corresponding `excluded/` lists.

| File | Role |
|------|------|
| `train_val_by_cfu_group_ug_per_mL.csv` | Primary CFU-aware train/val table |
| `test_LL37_by_cfu_group_ug_per_mL.csv` | Default LL37 external test hook |
| `removed_similar_to_LL37_*.csv` | Sequences/rows removed by sim > 0.30 |
| `removed_nonstandard_AA_*.csv` | Non-standard AA removals |
| `similarity_filter_audit.csv` | Per-sequence max similarity audit |
| `LL37_v0.csv` | LL37 holdout family |
| `features.txt` | Same AA feature table as `data/metadata/features.txt` |

Full packaging notes: [`data/trainval_dbassp/README.md`](../data/trainval_dbassp/README.md).
