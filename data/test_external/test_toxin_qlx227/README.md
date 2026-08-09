# test_toxin_qlx227

External **hemolytic toxicity** test set (qlx227), packaged for TIGER toxin-filter evaluation.

| Item | Value |
|------|-------|
| Sequences | **88** |
| Label | HC50 ≤ **512** µg/mL (toxin=1) |
| Class balance | toxin=12, non-toxin=76 |
| Selection | active peptides with `mic_min ≤ 128` |

## Layout

```text
test_toxin_qlx227/
├── qlx227_hemolysis_active_micmin_le128.csv  # primary panel
├── qlx227_toxicity_labeled.csv               # minimal: sequence, label, hc50
├── dataset_meta.json
└── README.md
```

## Labeling

Same contract as `trainval_dbassp/toxin/` / `code/toxin_filter`:

- Endpoint: **human erythrocyte HC50** (hemolysis), not CC50
- Threshold **T = 512 µg/mL**, inequality-aware

`mic_min` is retained only to document the active-peptide filter used by the
final external protocol; bacterial MIC columns and model predictions are omitted.

Source: `data/test_qlx227/qlx227_active_micmin_le128_subset.csv`.
