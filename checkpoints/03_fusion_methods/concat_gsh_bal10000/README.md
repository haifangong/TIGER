# `concat_gsh_bal10000`

Fusion-method ablation point.

## Settings

| Key | Value |
|-----|-------|
| `name` | `concat_mod_gsh` |
| `feature_modalities` | `gsh` |
| `fusion` | `concat` |
| `fusion_attn_mode` | `self_gsh` |
| `similarity_threshold` | `0.3` |
| `pair_balance_num` | `10000` |
| `use_signed_sampling` | `False` |
| `delta_bin_width` | `1.0` |
| `seed` | `123` |
| `structure_features` | `s` |

## 5-fold CV (ensemble)

- **PCC:** 0.6883
- **log10MAE:** 0.4396
- **RSE:** 0.5263
- **selection_score** (lower better): 0.8194

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

Copied from `outputs_code_struct_s_concat_ablation/concat_ablation__mod_gsh`.
