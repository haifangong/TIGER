# 03_fusion_methods

Ablation of **how the three modality tokens are fused** (attention Q/KV roles and
token order, plus concat vs attention), under locked `gsh` + sim=0.3 + unsigned
binning.

## Question

Which multimodal fusion operator works best: attention with different Query
roles / token orders, or simple concatenation?

## Locked settings (all points)

| Setting | Value |
|---------|-------|
| `feature_modalities` | `gsh` |
| `similarity_threshold` | 0.3 |
| `use_signed_sampling` | false |
| `delta_bin_width` | 1.0 |
| `structure_features` | `s` |
| `include_node_coords` | false |

## Two matched panels (do not mix `bal` when comparing)

### Panel C1 — Attention modes (`pair_balance_num=1000`)

| Folder | Mode | Meaning |
|--------|------|---------|
| `attn_cross_qs_bal1000/` | `cross_qs` | Q=sequence; K/V=graph+global |
| `attn_cross_qh_bal1000/` | `cross_qh` | Q=graph; K/V=seq+global |
| `attn_cross_qg_bal1000/` | `cross_qg` | Q=global; K/V=seq+graph |
| `attn_self_gsh_bal1000/` | `self_gsh` | self-attn, token order g→s→h |
| `attn_self_sgh_bal1000/` | `self_sgh` | self-attn, token order s→g→h |

### Panel C2 — Concat vs attention (`pair_balance_num=10000`)

| Folder | Fusion | Notes |
|--------|--------|-------|
| `attn_cross_qs_bal10000/` | attention / `cross_qs` | same recipe as best binsize run |
| `concat_gsh_bal10000/` | concat | no attention; flat fusion of g+s+h |

## CV results (from `../leaderboard.csv`)

**C1 (bal=1000):**

| point | PCC | log10MAE | RSE | score ↓ |
|-------|----:|---------:|----:|--------:|
| attn_cross_qs_bal1000 | **0.694** | 0.437 | 0.518 | **0.789** |
| attn_self_sgh_bal1000 | 0.691 | 0.436 | 0.523 | 0.794 |
| attn_self_gsh_bal1000 | 0.690 | 0.442 | 0.524 | 0.824 |
| attn_cross_qh_bal1000 | 0.683 | 0.442 | 0.533 | 0.843 |
| attn_cross_qg_bal1000 | 0.667 | 0.452 | 0.555 | 0.922 |

**C2 (bal=10000):**

| point | PCC | log10MAE | RSE | score ↓ |
|-------|----:|---------:|----:|--------:|
| attn_cross_qs_bal10000 | **0.705** | **0.429** | **0.503** | **0.729** |
| concat_gsh_bal10000 | 0.688 | 0.440 | 0.526 | 0.819 |

**Takeaway:** `cross_qs` wins among attention modes; attention beats concat at
matched bal=10000 (ΔPCC ≈ +0.017).

## Reproduce

```bash
cd TIGER
bash code/scripts_train/run_ablation_03_fusion.sh
# PART=attn|concat|all
```

**Provenance:**

| Archive folder | Source |
|----------------|--------|
| `attn_*_bal1000` | `outputs_code_struct_s_sim_bin_qkv/struct_s_grid__sim0p3_bin1p0_*` |
| `concat_gsh_bal10000` | `outputs_code_struct_s_concat_ablation/concat_ablation__mod_gsh` |
| `attn_cross_qs_bal10000` | `outputs_code_struct_s_binsize_grid/...unsigned_bin1p0_bal10000` |

Parent: [`../README.md`](../README.md).
