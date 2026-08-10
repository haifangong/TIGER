# external_eval_tiger_vs_evo

Archived external comparison of **TIGER** (hard MoE / sim 0.3 / sim 0.7) vs
**EvoGradient** on the LL37 and APEXGO activity panels.

## Method note

EvoGradient predicts **absolute** log10(MIC); this archive converts those scores
into **pair ΔMIC** (`anchor − query` in log2 space) so both methods are scored
on the same pair-delta task that TIGER trains.

## Layout

```text
external_eval_tiger_vs_evo/
├── leaderboard.csv / leaderboard_metrics.csv / summary.json
├── ll37/
│   ├── pair_predictions_aligned.csv
│   ├── tiger_{moe_hard,0p3,0p7}_predictions.csv
│   └── evo/evo_pair_predictions.csv
└── apexgo/
    └── (same)
```

## Reproduce

```bash
bash code/scripts_eval/eval_tiger_vs_evo.sh
```

Requires EvoGradient under `../baseline/AMP-potency-prediction-EvoGradient`
(relative to the TIGER repo) when regenerating; the CSVs here are sufficient to
inspect the published comparison without re-running.
