# 02_pair_balance_bin1

Ablation of **how many training pairs are kept per Δ-bin** (`pair_balance_num`),
with bin width fixed at `delta_bin_width=1.0`. Compares **unsigned** vs **signed**
binning under the locked `gsh` + `cross_qs` + sim=0.3 recipe.

## Question

How does the per-bin sample cap affect 5-fold CV pair-delta accuracy? Is unsigned
or signed Δ binning better?

## Locked settings

| Setting | Value |
|---------|-------|
| `feature_modalities` | `gsh` |
| `fusion` / `fusion_attn_mode` | `attention` / `cross_qs` |
| `similarity_threshold` | 0.3 |
| `delta_bin_width` | **1.0** (not swept) |
| `structure_features` | `s` |
| `include_node_coords` | false |
| `seed` | **123** (from released `config.json` / `summary.json`) |

## Sweep

1. `use_signed_sampling ∈ {false, true}`
2. `pair_balance_num ∈ {1000, 2000, 4000, 8000, 10000, 20000}`

```text
unsigned_bal{1000,2000,4000,8000,10000,20000}/
signed_bal{1000,2000,4000,8000,10000,20000}/
```

## Binning semantics

| Mode | Bin key | Role of `delta_bin_width=1.0` |
|------|---------|------------------------------|
| **unsigned** (`use_signed_sampling=false`) | `round(\|Δ\| × 100)` | unused for the key (kept for bookkeeping) |
| **signed** (`use_signed_sampling=true`) | `floor(Δ / 1.0)` | **active** — defines signed bins |

`pair_balance_num` caps how many pairs are sampled **inside each bin**.

## Best point (paper default recipe)

```text
unsigned_bal10000/
  CV: PCC ≈ 0.705, log10MAE ≈ 0.429, RSE ≈ 0.503, score ≈ 0.729
```

This run is also aliased conceptually as:

- `03_fusion_methods/attn_cross_qs_bal10000/`
- `04_similarity/sim0p3_bal10000/`

and often carries `external_eval/` (LL37 / APEX-GO) under the same directory.

## CV results (from `../leaderboard.csv`)

**Unsigned (primary panel):**

| point | PCC | log10MAE | RSE | score ↓ |
|-------|----:|---------:|----:|--------:|
| unsigned_bal10000 | **0.705** | **0.429** | **0.503** | **0.729** |
| unsigned_bal2000 | 0.705 | 0.431 | 0.502 | 0.733 |
| unsigned_bal4000 | 0.701 | 0.432 | 0.509 | 0.749 |
| unsigned_bal8000 | 0.697 | 0.432 | 0.515 | 0.764 |
| unsigned_bal1000 | 0.695 | 0.436 | 0.517 | 0.783 |
| unsigned_bal20000 | 0.694 | 0.437 | 0.518 | 0.789 |

**Signed (secondary):**

| point | PCC | log10MAE | RSE | score ↓ |
|-------|----:|---------:|----:|--------:|
| signed_bal20000 | 0.699 | 0.435 | 0.511 | 0.766 |
| signed_bal10000 | 0.683 | 0.444 | 0.534 | 0.845 |
| signed_bal8000 | 0.665 | 0.457 | 0.557 | 0.948 |
| signed_bal4000 | 0.643 | 0.471 | 0.586 | 1.067 |
| signed_bal2000 | 0.603 | 0.491 | 0.636 | 1.258 |
| signed_bal1000 | 0.548 | 0.517 | 0.700 | 1.510 |

## Reproduce

```bash
cd TIGER
bash code/scripts_train/run_ablation_02_pair_balance.sh
```

**Provenance:** `outputs_code_struct_s_binsize_grid/binsize_grid__{unsigned,signed}_bin1p0_bal*`.  
Parent: [`../README.md`](../README.md).
