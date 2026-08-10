# test_activity_apexgo

External **E. coli activity** test set (APEX-GO), packaged for TIGER pair-delta MIC evaluation.

> **Provenance:** a **sequence-disjoint external family panel** with an explicitly reported train-similarity distribution — **not** a strict `<30%` template-disjoint set. Audit: **≈49.1%** of sequences have `max_sim_to_train > 0.30` (max **0.4875**); see `../test_pair_similarity_compare/`.

| Item | Value |
|------|-------|
| Sequences | **110** (10 families × WT + 10 variants) |
| Primary pairs | **200** (template↔variant, both directions) |
| Endpoint | `MIC_Escherichia_coli` only |
| CFU | `1E5 - 1E6` |

## Layout

```text
test_activity_apexgo/
├── apexgo_peptides.csv
├── labels/apexgo_peptides_<LABEL>.csv
├── pairs/pairs_template_centric_alldelta_<LABEL>.csv   # 200 pairs
├── pairs/pairs_template_centric_absdelta_ge1_geo3.csv  # optional
├── pdb/   # symlink → Rosetta PDBs
├── dataset_meta.json
└── README.md
```

## Labels

| Label | Meaning |
|-------|---------|
| `geo3` (default in `apexgo_peptides.csv`) | Geometric mean of ATCC11775, AIC221, AIC222 |
| `ATCC11775` / `AIC221` / `AIC222` | Single E. coli strain |

## Pair protocol (`template_centric_alldelta`)

1. Within family: WT template ↔ each of 10 variants
2. Both directions → `10 × 10 × 2 = 200`
3. Similarity ≥ 0.30 (same as TIGER training)
4. Target: `Δ = log2(MIC_anchor) − log2(MIC_query)`

## Recommended eval

```text
peptides = labels/apexgo_peptides_AIC222.csv   # or geo3
pairs    = pairs/pairs_template_centric_alldelta_AIC222.csv
pdb      = pdb/
```

This pack excludes unrelated toxicity/CC50 eval folders from `test_apexgo/`.
