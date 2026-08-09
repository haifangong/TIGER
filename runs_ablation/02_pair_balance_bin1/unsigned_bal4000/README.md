# `unsigned_bal4000`

Pair-balance ablation point (samples per Δ-bin).

## Settings

| Key | Value |
|-----|-------|
| `name` | `unsigned_bin1p0_bal4000` |
| `feature_modalities` | `gsh` |
| `fusion` | `attention` |
| `fusion_attn_mode` | `cross_qs` |
| `similarity_threshold` | `0.3` |
| `pair_balance_num` | `4000` |
| `use_signed_sampling` | `False` |
| `delta_bin_width` | `1.0` |
| `seed` | `123` |
| `structure_features` | `s` |

## 5-fold CV (ensemble)

- **PCC:** 0.7005
- **log10MAE:** 0.4318
- **RSE:** 0.5092
- **selection_score** (lower better): 0.7494

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

## Provenance

Copied from `outputs_code_struct_s_binsize_grid/binsize_grid__unsigned_bin1p0_bal4000`.
