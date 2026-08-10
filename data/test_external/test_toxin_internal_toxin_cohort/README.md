# test_toxin_internal_toxin_cohort

External **hemolytic toxicity** panel (internal_toxin_cohort) for TIGER toxin-filter evaluation.

| Item | Value |
|------|-------|
| Sequences | **88** |
| Label | HC50 ≤ **512** µg/mL (toxin=1) |
| Class balance | toxin=12, non-toxin=76 |
| Selection | active peptides with `mic_min ≤ 128` |

## Layout

```text
test_toxin_internal_toxin_cohort/
├── internal_toxin_cohort_hemolysis_active_micmin_le128.csv  # primary panel
├── internal_toxin_cohort_toxicity_labeled.csv               # sequence, label, hc50
├── internal_toxin_cohort_hemolysis_predictions_both.csv     # archived model predictions
├── internal_toxin_cohort_hemolysis_predictions_both_full.csv
├── internal_toxin_cohort_prediction_summary.json
├── dataset_meta.json
└── README.md
```

## Reproduce (one command)

```bash
bash code/scripts_eval/eval_toxin_internal_toxin_cohort.sh
# or:
PYTHONPATH=. python code/toxin_filter/eval_internal_toxin_cohort.py
```

Uses `checkpoints/ablation_toxin/both/checkpoints/*_fold{1..5}.joblib` by default.

## Labeling

Same contract as `trainval_dbassp/toxin/` / `code/toxin_filter`:

- Endpoint: **human erythrocyte HC50** (hemolysis), not CC50
- Threshold **T = 512 µg/mL**, inequality-aware
