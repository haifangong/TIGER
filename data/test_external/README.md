# test_external

Clean evaluation packs for the TIGER MIC / toxin pipeline.

**Do not describe LL37 and APEXGO as one shared “strict unseen-template” set** — they differ in provenance and train-similarity.

```text
test_external/
├── README.md
├── MANIFEST.json
├── test_activity_ll37/                 # DBAASP-derived LL37 family holdout
├── test_activity_apexgo/               # external APEX-GO family panel
├── test_toxin_internal_toxin_cohort/   # 88-peptide hemolytic panel (+ predictions)
└── test_pair_similarity_compare/       # test-vs-train similarity audit
```

## LL37 (`test_activity_ll37/`) — similarity-filtered family holdout

- **Provenance:** carved from the DBAASP-derived pool, not an independent database dump.
- **Holdout construction:** remove every train/val peptide with NW similarity `> 0.30` to the LL37 family (`metadata/LL37_v0.csv`, 112 sequences).
- **Removed:** **1,122 unique peptides** / **1,402 assay rows** (`metadata/removed_similar_to_LL37_{seq,ug_per_mL}.csv`).
- **Eval size:** 68 sequences / 509 neighbor pairs (*E. coli* MIC).
- **Train similarity:** `frac_gt_0.3 = 0`, `max ≤ 0.30` (see `test_pair_similarity_compare/`).

## APEXGO (`test_activity_apexgo/`) — sequence-disjoint external family panel

- **Provenance:** APEX-GO family templates + variants (independent of the LL37 holdout recipe).
- **Eval size:** 110 sequences / 200 template↔variant pairs.
- **Train similarity:** sequence-disjoint from the train table, but **not** `<30%` similarity-disjoint.
  Audit (`test_vs_train_similarity_summary.csv`): **≈49.1%** of sequences have `max_sim_to_train > 0.30`, max **0.4875**.
- Preferred wording: *sequence-disjoint external family panel with an explicitly reported train-similarity distribution* (not “strict template-disjoint”).

## Toxicity (`test_toxin_internal_toxin_cohort/`)

88 active peptides (`mic_min ≤ 128`) with HC50 labels (threshold 512 µg/mL).  
Labels + archived predictions ship in-pack; regenerate with:

```bash
bash code/scripts_eval/eval_toxin_internal_toxin_cohort.sh
```

## Similarity audit (`test_pair_similarity_compare/`)

Per-sequence max similarity of LL37 / APEXGO panels vs the CFU-aware train/val table.
