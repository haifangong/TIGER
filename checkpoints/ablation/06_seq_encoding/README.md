# 06_seq_encoding

Ablation of the **sequence branch encoding** under the locked paper MIC pair-delta
recipe (`gsh` + `cross_qs` + sim=0.3 + unsigned bal=10000 + struct=`s`).

## Question

Does replacing the legacy integer AA-code linear map with a categorical
one-hot / embedding sequence encoder improve (or hurt) pair-delta MIC prediction?

## Locked settings

| Setting | Value |
|---------|-------|
| `feature_modalities` | `gsh` |
| `fusion` / `fusion_attn_mode` | `attention` / `cross_qs` |
| `similarity_threshold` | 0.3 |
| `pair_balance_num` | 10000 |
| `use_signed_sampling` | false |
| `delta_bin_width` | 1.0 |
| `structure_features` | `s` |
| `include_node_coords` | false |
| `seed` | 1 |

## Sweep

| point | `seq_encoding` | Meaning |
|-------|----------------|---------|
| `seq_integer` | `integer` | `Linear(max_len)` over AA codes 1..20 (paper default) |
| `seq_embedding` | `embedding` | `nn.Embedding(21, d)` + positional embedding + masked mean |
| `seq_onehot` | `onehot` | one-hot(21) → `Linear(21, d)` + positional embedding + masked mean |

Implementation: `code/models.py` → `PeptideEncoder._encode_sequence`.

## Layout

Training writes under experiment ids:

```text
seq_encoding_grid__seq_integer/
seq_encoding_grid__seq_embedding/
seq_encoding_grid__seq_onehot/
```

Convenient short names (symlinks) may also be present:

```text
seq_integer/ -> seq_encoding_grid__seq_integer/
seq_embedding/ -> seq_encoding_grid__seq_embedding/
seq_onehot/ -> seq_encoding_grid__seq_onehot/
```

## Reproduce

```bash
cd /path/to/TIGER
bash code/scripts_train/run_ablation_06_seq_encoding.sh

# only categorical methods:
ONLY="seq_embedding seq_onehot" bash code/scripts_train/run_ablation_06_seq_encoding.sh
```

Or:

```bash
export PYTHONPATH=.
python -m code.main train \
  --config code/configs/gsh_struct_s_base.json \
  --out-dir checkpoints/ablation/06_seq_encoding/seq_encoding_grid__seq_onehot \
  --name seq_onehot \
  --structure-features s --feature-modalities gsh \
  --fusion attention --fusion-attn-mode cross_qs \
  --seq-encoding onehot \
  --no-include-node-coords \
  --similarity-threshold 0.3 --delta-bin-width 1.0 \
  --pair-balance-num 10000 --no-use-signed-sampling \
  --seed 1 --gpu 0
```

## Notes

- Prior `outputs/outputs_ll37_seq_encoding_fix/` used an **older** recipe
  (`bal=2500`, signed sampling, `structure_features=sep`) and is **not** comparable
  to this panel.
- After completion, refresh `cv_leaderboard.json` with:
  `PYTHONPATH=. python scripts/run_struct_s_seq_encoding_grid.py --out-root checkpoints/ablation/06_seq_encoding --leaderboard-only`
