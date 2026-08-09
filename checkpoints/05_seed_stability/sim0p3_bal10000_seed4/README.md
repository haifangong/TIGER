# `sim0p3_bal10000_seed123`

Training run in the paper ablation archive.

## Settings

| Key | Value |
|-----|-------|
| `name` | `sim0p3_bal10000_seed123` |
| `feature_modalities` | `gsh` |
| `fusion` | `attention` |
| `fusion_attn_mode` | `cross_qs` |
| `similarity_threshold` | `0.3` |
| `pair_balance_num` | `10000` |
| `use_signed_sampling` | `False` |
| `delta_bin_width` | `1.0` |
| `seed` | `123` |
| `structure_features` | `s` |

## 5-fold CV (ensemble)

- **PCC:** 0.6977
- **log10MAE:** 0.4341
- **RSE:** 0.5132
- **selection_score** (lower better): 0.7687

Full metrics: `results/summary.json`. Panel context: [`../README.md`](../README.md).

## Contents

```text
config.json
checkpoints/          # fold{1..5}_best.pt, fold{1..5}_last.pt
results/summary.json
intermediate/         # fold-wise z-score stats, etc.
logs/                 # training logs (if present)
external_eval/        # optional LL37 / APEX-GO eval outputs
```

## Note

Legacy seed naming may appear in the folder name (e.g. `seed123`). New runs use plain seeds `1..5` (`…_seed1` … `…_seed5`).
