# test_activity_ll37

External **E. coli activity** test set (LL37 family), packaged for TIGER neighbor-style pair evaluation.

> **Provenance:** this is a **similarity-filtered family holdout** from the DBAASP-derived pool (not an independent external database). Train/val removed **1,122 unique peptides / 1,402 assay rows** with similarity `> 0.30` to `metadata/LL37_v0.csv`. Every eval sequence here has `max_sim_to_train ≤ 0.30`.

| Item | Value |
|------|-------|
| **Sequences** | **68** (`ll37_sequences_mic.csv`) |
| **Eval pairs** | **509** (`ll37_pairs_neighbor.csv`; neighbor protocol) |
| Endpoint | E. coli MIC (µg/mL) |
| Primary CFU | `1E5 - 1E6` |

Report both numbers together: **68 sequences / 509 pairs**.

## Layout

```text
test_activity_ll37/
├── ll37_sequences_mic.csv
├── ll37_pairs_neighbor.csv
├── ll37_pairs_neighbor_meta.json
├── pdb/<SEQUENCE>.pdb
├── dataset_meta.json
└── README.md
```

## Pair protocol (`ll37_neighbor`)

1. Queries = LL37 panel (primary CFU `1E5–1E6`)
2. Anchors = train/val, same CFU, `|Δlen| ≤ 5`
3. Keep `sim ≥ 0.3`, top-50; else fallback top-20
4. Target: `Δlog2 = log2MIC(anchor) − log2MIC(query)`

Legacy explicit top-k pair tables from `test_ll37/` are **not** included.
