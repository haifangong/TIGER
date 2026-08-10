# test_external

Clean evaluation packs for the TIGER MIC / toxin pipeline.

**Do not describe LL37 and APEXGO as one shared “strict unseen-template” set** — they differ in provenance and train-similarity.

**Always cite both sequence count and pair count** (not pairs alone):

| Pack | Sequences | Eval pairs |
|------|----------:|-----------:|
| `test_activity_ll37/` | **68** | **509** |
| `test_activity_apexgo/` | **110** | **200** |
| `test_toxin_internal_toxin_cohort/` | **88** | — (sequence-level HC50) |

```text
test_external/
├── README.md
├── MANIFEST.json
├── test_activity_ll37/                 # 68 sequences / 509 pairs
├── test_activity_apexgo/               # 110 sequences / 200 pairs
├── test_toxin_internal_toxin_cohort/   # 88-peptide hemolytic panel (+ predictions)
└── test_pair_similarity_compare/       # test-vs-train similarity audit
```

## LL37 (`test_activity_ll37/`) — similarity-filtered family holdout

- **Eval size:** **68 sequences** + **509 neighbor pairs** (*E. coli* MIC).
- **Provenance:** carved from the DBAASP-derived pool, not an independent database dump.
- **Holdout construction:** remove every train/val peptide with NW similarity `> 0.30` to the LL37 family (**112** named sequences in `metadata/test_seq_LL37_named.csv`).
- **Removed from train/val:** **1,122 unique peptides** / **1,402 assay rows** (`metadata/removed_similar_to_LL37_{seq,ug_per_mL}.csv`).
- **Train similarity:** `frac_gt_0.3 = 0`, `max ≤ 0.30` (see `test_pair_similarity_compare/`).

## APEXGO (`test_activity_apexgo/`) — sequence-disjoint external family panel

- **Eval size:** **110 sequences** (10 families × WT + 10 variants) + **200 template↔variant pairs**.
- **Provenance:** APEX-GO family templates + variants (independent of the LL37 holdout recipe).
- **Train similarity:** sequence-disjoint from the train table, but **not** `<30%` similarity-disjoint.
  Audit (`test_vs_train_similarity_summary.csv`): **≈49.1%** of sequences have `max_sim_to_train > 0.30`, max **0.4875**.
- Preferred wording: *sequence-disjoint external family panel with an explicitly reported train-similarity distribution* (not “strict template-disjoint”).

## Toxicity (`test_toxin_internal_toxin_cohort/`)

**88 sequences** (active peptides with `mic_min ≤ 128`) with HC50 labels (threshold 512 µg/mL).  
Labels + archived predictions ship in-pack; regenerate with:

```bash
bash code/scripts_eval/eval_toxin_internal_toxin_cohort.sh
```

## Similarity audit (`test_pair_similarity_compare/`)

Per-sequence max similarity of the LL37 (**68**) / APEXGO (**110**) panels vs the CFU-aware train/val table.
