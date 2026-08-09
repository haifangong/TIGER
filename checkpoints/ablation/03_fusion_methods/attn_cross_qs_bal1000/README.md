# `attn_cross_qs_bal1000`

Fusion-method ablation point.

## Settings

| Key | Value |
|-----|-------|
| `name` | `sim0p3_bin1p0_cross_qs` |
| `feature_modalities` | `gsh` |
| `fusion` | `attention` |
| `fusion_attn_mode` | `cross_qs` |
| `similarity_threshold` | `0.3` |
| `pair_balance_num` | `1000` |
| `use_signed_sampling` | `False` |
| `delta_bin_width` | `1.0` |
| `seed` | `123` |
| `structure_features` | `s` |

## 5-fold CV (ensemble)

- **PCC:** 0.6940
- **log10MAE:** 0.4365
- **RSE:** 0.5184
- **selection_score** (lower better): 0.7894

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

Copied from `outputs_code_struct_s_sim_bin_qkv/struct_s_grid__sim0p3_bin1p0_cross_qs`.
