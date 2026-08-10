# checkpoints/wetlab — production MIC models for wetlab ranking

Trained on the **full DBAASP** per-species tables in [`data/wetlab/`](../../data/wetlab/)
(**without** the LL37 holdout used in `checkpoints/ablation/`). These are the
models used for prospective / synthesis screening via a **five-fold ensemble**.

## Grid (12 runs — uploaded)

| Factor | Values |
|--------|--------|
| Species (top-6 by MIC coverage) | E. coli, S. aureus, P. aeruginosa, B. subtilis, K. pneumoniae, S. epidermidis |
| `similarity_threshold` | `{0.3, 0.7}` |

Each run ships `checkpoints/fold{1..5}_best.pt` + `config.json` + CV summaries.

Locked recipe: `gsh` + `cross_qs` + `struct=s` + unsigned `bal=10000` + `seed=1`.

## Launch / ranking

```bash
cd TIGER
bash code/scripts_train/run_wetlab_species_sim.sh
```

Inference / ranking: `pipeline/04_metric_ranking/`
