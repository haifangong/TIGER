# checkpoints/wetlab — production MIC models for wetlab ranking

Trained on the **full DBAASP** table (`data/dbaasp_amp_training_by_cfu_group_ug_per_mL _123.csv`),
**without** the LL37 holdout used in `checkpoints/ablation/`.

## Grid (12 runs)

| Factor | Values |
|--------|--------|
| Species (top-6 by MIC coverage) | E. coli, S. aureus, P. aeruginosa, B. subtilis, K. pneumoniae, S. epidermidis |
| `similarity_threshold` | `{0.3, 0.7}` |

Locked recipe: `gsh` + `cross_qs` + `struct=s` + unsigned `bal=10000` + `seed=1`.

## Launch

```bash
cd TIGER
bash code/scripts_train/run_wetlab_species_sim.sh
```

Per-species tables: `data/wetlab/train_<Species>.csv`  
Inference / ranking: `pipeline/04_metric_ranking/`
