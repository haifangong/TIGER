# TIGER `code/` — modular core

Normalized package for the CFU-aware MIC **pair-delta** model. The legacy
monolithic pipelines under [`src/poap_gpt/`](../src/poap_gpt/) are **kept unchanged**.

## Layout

```text
code/
├── main.py              # CLI: train | evaluate | infer
├── train.py             # GroupKFold + final retrain
├── evaluation.py        # Neighbor pair-delta eval
├── infer.py             # Checkpoint scoring
├── dataloader.py        # Preprocess, graphs, pairs, datasets
├── models.py            # GNN / PeptideEncoder / Pair|Single heads
├── configs/
│   └── default_fusion_attention.json
└── utils/
    ├── config.py        # Config dataclass + JSON load/save
    ├── constants.py     # AA codes 1..20, CFU maps
    ├── features.py      # Physicochemical / tabular features
    ├── metrics.py       # RMSE / Pearson / calibrator
    ├── scaling.py       # Per-fold global_f z-score
    └── seed.py
```

## Amino-acid encoding (1–20)

Alphabetical standard residues, **padding = 0**:

| AA | A | C | D | E | F | G | H | I | K | L | M | N | P | Q | R | S | T | V | W | Y |
|----|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ID | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |10 |11 |12 |13 |14 |15 |16 |17 |18 |19 |20 |

### Default (strongest empirical recipe)

From LL37-holdout ablations, the best CV came from **`global_feature_scaling=zscore`** with the
legacy-style position-slot sequence branch. Defaults therefore are:

- `seq_encoding=integer` — `Linear(max_len)` over AA codes **1..20** (pad 0)
- `global_feature_scaling=zscore` — per-fold train mean/std on `global_f`
- `fusion=attention`, AdamW `lr=1.5e-3`, `weight_decay=5e-4`, `lr_scheduler=none`

For a categorically cleaner sequence branch, switch to
`--seq-encoding embedding` (Embedding + positional encoding + masked mean).

## Usage

From the `TIGER/` root:

```bash
export PYTHONPATH=.
python -m code.main train --config code/configs/default_fusion_attention.json --gpu 0

python -m code.main evaluate --config outputs/tiger_code_run/config.json

python -m code.main infer \
  --config outputs/tiger_code_run/config.json \
  --checkpoint outputs/tiger_code_run/checkpoints/final.pt \
  --gpu 0
```

Smoke test (few epochs / fewer pairs):

```bash
python -m code.main train --config code/configs/default_fusion_attention.json --smoke --gpu 0
```
