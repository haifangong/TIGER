# checkpoints — index of released run weights

Ablation and production MIC/toxin weights are published under the `runs_*`
directories (same layout as local training archives). This folder keeps the
aggregate index files for convenience.

| Local / script path | Released path |
|---------------------|---------------|
| `runs_ablation/` | [`../runs_ablation/`](../runs_ablation/) |
| `runs_ablation_toxin/` | [`../runs_ablation_toxin/`](../runs_ablation_toxin/) |
| `runs_wetlab/` | [`../runs_wetlab/`](../runs_wetlab/) |

Only **final selected** weights are shipped (`fold*_best.pt` for MIC; `*.joblib`
for toxin). `fold*_last.pt` and training dumps are omitted.

Aggregate MIC CV table: [`../runs_ablation/leaderboard.csv`](../runs_ablation/leaderboard.csv)
(also mirrored here as [`leaderboard.csv`](leaderboard.csv)).
