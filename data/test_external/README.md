# test_external

Clean external test packs for the **final TIGER pipeline**.

Only evaluation-relevant endpoints are kept:

- **Activity** → *Escherichia coli* MIC (pair-delta)
- **Toxicity** → hemolytic HC50 (human erythrocytes), threshold 512 µg/mL

```text
test_external/
├── README.md
├── MANIFEST.json
├── test_activity_apexgo/            # 110 sequences, 200 pairs
├── test_activity_ll37/              # 68 sequences, ~500 (509) pairs
├── test_toxin_qlx227/               # 88 sequences (HC50)
└── test_pair_similarity_compare/    # test-vs-train similarity audit
```

| Pack | Task | Size | Primary files |
|------|------|------|---------------|
| `test_activity_apexgo` | E. coli activity | 110 seq / 200 pairs | `apexgo_peptides.csv`, `pairs/pairs_template_centric_alldelta_*.csv` |
| `test_activity_ll37` | E. coli activity | 68 seq / 509 pairs | `ll37_sequences_mic.csv`, `ll37_pairs_neighbor.csv` |
| `test_toxin_qlx227` | Hemolytic toxicity | 88 seq | `qlx227_hemolysis_active_micmin_le128.csv` |
| `test_pair_similarity_compare` | Similarity audit | — | `pair_similarity_apexgo_vs_ll37.pdf`, `*_similarity*.csv` |

Sources: `data/test_apexgo`, `data/test_ll37`, `data/test_qlx227`  
(unrelated toxicity/CC50 eval outputs, model predictions, and legacy pair tables are excluded).
