# Sequence-encoding ablation results

CV metrics for the sequence-branch encoding panel under the locked MIC pair-delta
recipe (`gsh` + `cross_qs` + sim=0.3 + unsigned bal=10000 + struct=`s`).

| point | `seq_encoding` | Meaning |
|-------|----------------|---------|
| `seq_integer` | `integer` | `Linear(max_len)` over AA codes 1..20 (paper default) |
| `seq_embedding` | `embedding` | `nn.Embedding(21, d)` + positional embedding + masked mean |
| `seq_onehot` | `onehot` | one-hot(21) → `Linear(21, d)` + positional embedding + masked mean |

## Files

```text
06_seq_encoding/
├── leaderboard.csv              # CV summary for all 3 encodings
├── cv_leaderboard.json
├── best_config.json
├── seq_integer/{summary.json,fold_metrics.csv,calibrator.json,config.json}
├── seq_embedding/...
└── seq_onehot/...
```

Final weights: `runs_ablation/06_seq_encoding/seq_encoding_grid__seq_{integer,embedding,onehot}/checkpoints/fold*_best.pt` (5 folds each).

Reproduce:

```bash
bash code/scripts_train/run_ablation_06_seq_encoding.sh
```
