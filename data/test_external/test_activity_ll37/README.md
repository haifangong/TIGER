# test_activity_ll37

External **E. coli activity** test set (LL37 family), packaged for TIGER neighbor-style pair evaluation.

| Item | Value |
|------|-------|
| Sequences | **68** |
| Primary pairs | **509** (≈500; neighbor protocol) |
| Endpoint | E. coli MIC (µg/mL) |
| Primary CFU | `1E5 - 1E6` |

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
