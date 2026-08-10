# 01_modality_ablation

Ablation of **which input modalities** the MIC pair-delta encoder uses, under the
locked paper recipe (sim=0.3, unsigned bal=10000, attention `cross_qs`, struct=`s`).

## Question

Do global (`g`), sequence (`s`), and graph/structure (`h`) features all help, and
which combinations are essential?

## Locked settings

| Setting | Value |
|---------|-------|
| `similarity_threshold` | 0.3 |
| `pair_balance_num` | 10000 |
| `use_signed_sampling` | false (unsigned Δ bins) |
| `delta_bin_width` | 1.0 |
| `fusion` / `fusion_attn_mode` | `attention` / `cross_qs` |
| `structure_features` | `s` |
| `include_node_coords` | false |
| `seed` | **123** (from released `config.json` / `summary.json`) |

## Sweep

`feature_modalities` letter codes:

| Letter | Meaning |
|--------|---------|
| `g` | global physicochemical features |
| `s` | sequence embedding |
| `h` | graph / structure GNN |

## Points in this folder

```text
mod_gsh/   # full model (best in this panel)
mod_gs/
mod_sh/
mod_gh/
mod_g/
mod_h/
```

Each point directory ships `config.json`, slim `checkpoints/fold*_best.pt`, and
`results/{summary,fold_metrics,calibrator}.json|.csv`.

## CV results (from `../leaderboard.csv`)

| point | PCC | log10MAE | RSE | selection_score ↓ |
|-------|----:|---------:|----:|------------------:|
| mod_gsh | 0.696 | 0.437 | 0.516 | **0.788** |
| mod_gs | 0.679 | 0.444 | 0.539 | 0.858 |
| mod_sh | 0.673 | 0.447 | 0.548 | 0.891 |
| mod_g | 0.568 | 0.501 | 0.678 | 1.399 |
| mod_gh | 0.562 | 0.504 | 0.684 | 1.420 |
| mod_h | 0.544 | 0.514 | 0.704 | 1.506 |

`selection_score = log2MAE + RSE − PCC − KCC` (lower better).

## Note

The invalid `cross_qs` sequence-only run (`mod_s`) is **not** included in this
release. Prefer `gsh/gs/sh/...` for the main narrative. Fairer single-modality
baselines use concat fusion (see `03_fusion_methods/concat_gsh_bal10000`).

## Reproduce

```bash
cd TIGER
bash code/scripts_train/run_ablation_01_modality.sh
# or: bash scripts/launch_struct_s_modality_grid.sh
```

**Provenance:** copied from `outputs_code_struct_s_modality_grid/modality_grid__mod_*`.  
Parent index: [`../README.md`](../README.md), [`../leaderboard.csv`](../leaderboard.csv).
