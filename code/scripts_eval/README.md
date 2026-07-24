# `scripts_eval/` — evaluation & inference launchers

Shell entrypoints that **evaluate** trained TIGER / toxin models on CV summaries
and external packs under `data/test_external/` (and legacy `data/test_*`).

Run from the `TIGER/` root:

```bash
cd /path/to/TIGER
conda activate ccseg
# toxin classical ML may use:  PY=/home/ubuntu/anaconda3/envs/class/bin/python
```

Shared helpers are sourced from `../scripts_train/_ablation_common.sh`
(`ROOT`, `PY`, `GPU`, `FOREGROUND`, `_eval_run`).

---

## Layout

```text
scripts_eval/
├── README.md
├── eval_cv_summary.sh                 # print results/summary.json
├── eval_ll37_apexgo_best.sh           # primary external MIC eval
├── eval_sim_thresholds_ll37_apexgo.sh # sim 0.3/0.5/0.7 external compare
├── eval_apexgo_within_family.sh       # APEX-GO template-centric pairs
├── eval_apexgo_high_span.sh           # code.main evaluate-apexgo
├── eval_tiger_vs_evo.sh               # LL37 + APEXGO TIGER MoE/0.3/0.7 vs EvoGradient
├── eval_infer_checkpoint.sh           # code.main infer
└── eval_toxin_qlx227.sh               # hemolytic HC50 external (qlx227 n=88)
```

Underlying Python tools mainly live in `TIGER/scripts/eval_*.py` and
`python -m code.main {evaluate,evaluate-apexgo,infer}`.

---

## External test data (clean packs)

Prefer these paths (see `data/test_external/README.md`):

| Pack | Task | Size |
|------|------|------|
| `data/test_external/test_activity_apexgo/` | E. coli MIC pairs | 110 seq / 200 pairs |
| `data/test_external/test_activity_ll37/` | E. coli MIC pairs | 68 seq / 509 pairs |
| `data/test_external/test_toxin_qlx227/` | Hemolytic HC50 | 88 seq |

Legacy sources (`data/test_apexgo`, `test_ll37`, `test_qlx227`) still work for
older eval Python scripts.

---

## Scripts

### `eval_cv_summary.sh`

**Function:** print the saved 5-fold CV `results/summary.json` for one MIC run  
**Backend:** `python -m code.main evaluate`  
**Default exp:** `runs_ablation/02_pair_balance_bin1/unsigned_bal10000`

```bash
bash code/scripts_eval/eval_cv_summary.sh
EXP_DIR=runs_ablation/01_modality_ablation/mod_gsh bash code/scripts_eval/eval_cv_summary.sh
```

### `eval_ll37_apexgo_best.sh`

**Function:** primary paper external MIC evaluation  
- LL37 **neighbor-509**  
- APEX-GO **geo3 template-centric 200**  

**Backend:** `scripts/eval_best_binsize_ll37_apexgo.py`  
**Default exp:** archive `unsigned_bal10000` (falls back to `outputs_code_struct_s_binsize_grid/...`)  
**Outputs:** `<exp>/external_eval/ll37_neighbor/` and `…/apexgo_geo3_template_centric/`

```bash
bash code/scripts_eval/eval_ll37_apexgo_best.sh
EXP_DIR=runs_ablation/02_pair_balance_bin1/unsigned_bal10000 \
  GPU=0 bash code/scripts_eval/eval_ll37_apexgo_best.sh
SKIP_LL37=1 bash code/scripts_eval/eval_ll37_apexgo_best.sh   # APEX-GO only
SKIP_APEXGO=1 bash code/scripts_eval/eval_ll37_apexgo_best.sh # LL37 only
```

### `eval_sim_thresholds_ll37_apexgo.sh`

**Function:** compare external LL37 + APEX-GO across training sim **0.3 / 0.5 / 0.7**  
(matches `runs_ablation/04_similarity`)  
**Backend:** `scripts/eval_sim03_05_07_ll37_apexgo.py`

```bash
bash code/scripts_eval/eval_sim_thresholds_ll37_apexgo.sh
```

### `eval_apexgo_within_family.sh`

**Function:** APEX-GO within-family / template-centric pair scoring for one experiment  
**Backend:** `scripts/eval_apexgo_within_family_combos.py`  
**Default pairs:** `data/test_external/test_activity_apexgo/pairs/…_geo3.csv`  
(fallback: `data/test_apexgo/pairs/…`)

```bash
bash code/scripts_eval/eval_apexgo_within_family.sh
EXP_DIR=… PAIR_CSV=data/test_external/test_activity_apexgo/pairs/pairs_template_centric_alldelta_AIC222.csv \
  bash code/scripts_eval/eval_apexgo_within_family.sh
```

### `eval_apexgo_high_span.sh`

**Function:** high-span APEX-GO family eval via the modular CLI  
**Backend:** `python -m code.main evaluate-apexgo`  
**Metrics:** log10MAE / RSE / PCC / KCC (overall + per-family)

```bash
bash code/scripts_eval/eval_apexgo_high_span.sh
CONFIG=…/config.json CKPT=…/checkpoints/fold1_best.pt \
  bash code/scripts_eval/eval_apexgo_high_span.sh
```

### `eval_tiger_vs_evo.sh`

**Function:** LL37 + APEXGO comparison of TIGER MoE / 0.3 / 0.7 vs EvoGradient (hard MoE only)  
**Backend:** `code/compare_tiger_evo.py`  
**Metrics:** PCC / KCC / MAE / RSE per peptide group  
**Typical out:** `runs_ablation/04_similarity/external_eval_tiger_vs_evo/`

```bash
bash code/scripts_eval/eval_tiger_vs_evo.sh
```

### `eval_infer_checkpoint.sh`

**Function:** neighbor-style inference / scoring with one checkpoint  
**Backend:** `python -m code.main infer`

```bash
bash code/scripts_eval/eval_infer_checkpoint.sh
CONFIG=…/config.json CKPT=…/fold1_best.pt bash code/scripts_eval/eval_infer_checkpoint.sh
```

### `eval_toxin_qlx227.sh`

**Function:** external **hemolytic** toxicity evaluation on qlx227 active subset  
(`mic_min ≤ 128` → **n=88**, label HC50 ≤ **512**)  
**Backend:** `data/test_qlx227/build_and_predict_qlx227.py`  
(uses `runs_ablation_toxin/both/checkpoints` by default inside that script)

Clean label table for reporting:

```text
data/test_external/test_toxin_qlx227/qlx227_hemolysis_active_micmin_le128.csv
```

Detailed metric folders (if regenerated):

```text
data/test_qlx227/eval_active_micmin_le128/
```

```bash
bash code/scripts_eval/eval_toxin_qlx227.sh
PY=/home/ubuntu/anaconda3/envs/class/bin/python bash code/scripts_eval/eval_toxin_qlx227.sh
```

---

## Common environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `PY` | ccseg python | Interpreter |
| `GPU` | `0` | CUDA device index |
| `FOREGROUND` | `1` for most evals | `0` = background via `nohup` |
| `EXP_DIR` | best unsigned bal10000 | Experiment directory with `config.json` + `checkpoints/` |
| `CONFIG` / `CKPT` | derived from `EXP_DIR` | Explicit config / checkpoint paths |
| `EVAL_LOG_DIR` | `runs_eval_logs/` | Background log directory |
| `SKIP_LL37` / `SKIP_APEXGO` | `0` | Skip one arm of the joint external eval |

---

## Suggested order

1. Train (or use archive): `bash code/scripts_train/train_best_recipe.sh`  
2. Check CV: `bash code/scripts_eval/eval_cv_summary.sh`  
3. External MIC: `bash code/scripts_eval/eval_ll37_apexgo_best.sh`  
4. Toxin external: `bash code/scripts_eval/eval_toxin_qlx227.sh`

Related training launchers: [`../scripts_train/`](../scripts_train/README.md).
