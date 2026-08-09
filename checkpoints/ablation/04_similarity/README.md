# 04_similarity

Ablation of the **minimum sequence similarity** used when building training pairs
(`similarity_threshold`), at fixed `pair_balance_num=10000` and the locked
`gsh` + `cross_qs` recipe.

Also contains a post-hoc **external evaluation** folder comparing LL37 / APEX-GO
performance across the released thresholds.

## Question

Does requiring more similar training pairs (0.7 vs 0.3) help or hurt CV
(and external) pair-delta prediction?

## Locked settings

| Setting | Value |
|---------|-------|
| `feature_modalities` | `gsh` |
| `fusion` / `fusion_attn_mode` | `attention` / `cross_qs` |
| `pair_balance_num` | **10000** |
| `use_signed_sampling` | false |
| `delta_bin_width` | 1.0 |
| `structure_features` | `s` |
| `include_node_coords` | false |

## Sweep

`similarity_threshold ∈ {0.3, 0.7}` (Needleman–Wunsch score / query length,
same contract as training). The sim=0.5 point is not redistributed.

```text
sim0p3_bal10000/   # = best unsigned_bal10000 recipe
sim0p7_bal10000/
```

## CV results (from `../leaderboard.csv`)

| point | sim | PCC | log10MAE | RSE | score ↓ |
|-------|----:|----:|---------:|----:|--------:|
| sim0p3_bal10000 | 0.3 | **0.705** | **0.429** | **0.503** | **0.729** |
| sim0p7_bal10000 | 0.7 | 0.470 | 0.537 | 0.779 | 1.775 |

Higher thresholds shrink the pair pool and **hurt** CV correlation under this
matched bal=10000 setting; **0.3** is best.

## `external_eval_ll37_apexgo/`

Post-training evaluation package (not an independent training sweep). Holds
per-threshold external summaries for:

- LL37 neighbor-style pairs
- APEX-GO template-centric pairs

plus optional MoE/similarity-gate notes under `moe_similarity_gate/`.
See [`external_eval_ll37_apexgo/README.md`](external_eval_ll37_apexgo/README.md).

Primary external eval launcher:

```bash
bash code/scripts_eval/eval_sim_thresholds_ll37_apexgo.sh
# or for the best recipe only:
bash code/scripts_eval/eval_ll37_apexgo_best.sh
```

## Reproduce training points

```bash
cd TIGER
bash code/scripts_train/run_ablation_04_similarity.sh
```

**Provenance:**

| Folder | Source |
|--------|--------|
| `sim0p3_bal10000` | `outputs_code_struct_s_binsize_grid/...unsigned_bin1p0_bal10000` |
| `sim0p7_bal10000` | `outputs_code_struct_s_sim05_07_fusion/...cross_qs_sim0p7_bal10000` |

Parent: [`../README.md`](../README.md).
