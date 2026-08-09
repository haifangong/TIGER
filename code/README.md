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
├── configs/             # JSON training templates
├── toxin_filter/        # HC50 classical-ML (+ optional DL) package
├── scripts_train/       # ablation / best-recipe train launchers (+ README)
├── scripts_eval/        # external / CV eval launchers (+ README)
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
```

### Paper train / eval launchers

- Training & ablations: [`scripts_train/README.md`](scripts_train/README.md)
- External / CV evaluation: [`scripts_eval/README.md`](scripts_eval/README.md)

```bash
bash code/scripts_train/train_best_recipe.sh          # locked MIC recipe (seed=1)
bash code/scripts_train/run_ablation_01_modality.sh   # modality panel
bash code/scripts_train/run_ablation_all_mic.sh       # MIC panels 01→05
bash code/scripts_train/run_ablation_toxin.sh         # toxin both/global/sequence

bash code/scripts_eval/eval_cv_summary.sh             # print CV summary.json
bash code/scripts_eval/eval_ll37_apexgo_best.sh       # LL37-509 + APEX-GO-200
bash code/scripts_eval/eval_toxin_internal_toxin_cohort.sh           # hemolytic internal_toxin_cohort (n=88)
```

Default training seed is **`1`**. Seed-stability panel uses **`1..5`**.

```bash
python -m code.main evaluate --config outputs/tiger_code_run/config.json

python -m code.main infer \
  --config outputs/tiger_code_run/config.json \
  --checkpoint outputs/tiger_code_run/checkpoints/final.pt \
  --gpu 0

# APEXGO high-span families (Mylodonin-2/3, Equusin-4, Mammuthusin-3, Hesperelin-3)
python -m code.main evaluate-apexgo \
  --config outputs/tiger_code_run/config.json \
  --checkpoint outputs/tiger_code_run/checkpoints/fold1_best.pt \
  --gpu 0
```

Smoke test (few epochs / fewer pairs):

```bash
python -m code.main train --config code/configs/default_fusion_attention.json --smoke --gpu 0
```

### APEXGO high-span eval

Selected templates are the 5 APEXGO families with the largest within-family
`log10(MIC)` span (`metadata/test_apexgo_high_span_*.csv`). The eval builds
all directed within-family pairs (450) and reports overall + per-family
`log10MAE`, `RSE`, `PCC`, `KCC` (plus `log2MAE`). PDBs: `data/3D_data_apexgo_Rosetta`.
