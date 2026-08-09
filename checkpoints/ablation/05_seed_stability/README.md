# 05_seed_stability

Ablation of **random seed** under the locked best recipe, at two training-pair
similarity thresholds, to quantify 5-fold CV **stability** (mean ± std across seeds).

## Question

How much do CV metrics move when only the seed changes? Is the sim=0.3 recipe
stable, and what happens at the harder sim=0.7 setting?

## Locked settings

| Setting | Value |
|---------|-------|
| `feature_modalities` | `gsh` |
| `fusion` / `fusion_attn_mode` | `attention` / `cross_qs` |
| `structure_features` | `s` |
| `include_node_coords` | false |
| `pair_balance_num` | 10000 |
| `delta_bin_width` | 1.0 |
| `use_signed_sampling` | false (unsigned) |

## Sweep (current convention)

| Factor | Values |
|--------|--------|
| `similarity_threshold` | `{0.3, 0.7}` |
| `seed` | `{1, 2, 3, 4, 5}` |

→ **10 runs**, folders named `sim0p{3|7}_bal10000_seed{1..5}`.

### How the five seeds were filled

| Seed id | Source |
|--------:|--------|
| **1** | Copied from paper originals `04_similarity/sim0p{3\|7}_bal10000` |
| **2** | Relabeled from former `seed7` stability run |
| **3** | Relabeled from former `seed42` stability run |
| **4** | Relabeled from former `seed123` stability run |
| **5** | Fresh train with `--seed 5` |

Each run’s `config.json` may include `stability_provenance` describing the source.
Weights for seed1–4 are unchanged; only directory / seed **ids** were normalized to 1–5.

Refresh tables:

```bash
bash code/scripts_train/run_ablation_05_seed_stability.sh
PYTHONPATH=. python scripts/run_struct_s_seed_stability_grid.py --leaderboard-only
```

## Index files in this folder

| File | Role |
|------|------|
| `experiment_manifest.json` | locked settings + planned grid |
| `cv_leaderboard.json` | per-run CV metrics as jobs finish |
| `stability_summary.json` | mean ± std of CV metrics across seeds (per sim) |
| `suite_logs/` | launcher / per-run logs |
| `suite_done.json` | written when the grid completes |

## Metrics of interest

Primary: CV ensemble `PCC`, `RSE`, `log10MAE`, `selection_score` across seeds  
(mean / std in `stability_summary.json`). Secondary: per-fold metrics inside each
run’s `results/summary.json`.

## Layout of each run directory

Same as other ablation points:

```text
sim0p3_bal10000_seed1/
  config.json
  checkpoints/fold{1..5}_{best,last}.pt
  results/summary.json
  intermediate/          # fold z-score stats, etc.
  logs/
```

Parent: [`../README.md`](../README.md).  
Train scripts: `code/scripts_train/run_ablation_05_seed_stability.sh`.
