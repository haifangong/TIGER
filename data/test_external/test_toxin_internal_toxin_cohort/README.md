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
├── internal_toxin_cohort_hemolysis_active_micmin_le128.csv
├── internal_toxin_cohort_toxicity_labeled.csv
├── internal_toxin_cohort_hemolysis_predictions_both.csv
├── internal_toxin_cohort_hemolysis_predictions_both_full.csv
├── internal_toxin_cohort_prediction_summary.json   # full metrics + overlap audit
├── internal_toxin_cohort_safety_metrics.csv        # precision/recall/AUC-PR/false-safe
├── dataset_meta.json
└── README.md
```

## Safety metrics (required)

Beyond Accuracy / F1 / MCC / AUC-ROC, the evaluator reports:

- Precision, **Recall / sensitivity** (toxic class)
- **False-safe rate** = FN / n_toxic (= 1 − recall)
- Specificity, **AUC-PR**
- Confusion matrix (TN/FP/FN/TP)

## Sequence overlap audit

`prediction_summary.json` → `sequence_overlap_audit` compares panel sequences to
`data/trainval_dbassp/toxin/toxicity_labeled_dataset.csv` (exact sequence match).

## Reproduce

```bash
bash code/scripts_eval/eval_toxin_internal_toxin_cohort.sh
# or:
PYTHONPATH=. python code/toxin_filter/eval_internal_toxin_cohort.py
```

Uses `checkpoints/ablation_toxin/both/checkpoints/*_fold{1..5}.joblib` by default.
