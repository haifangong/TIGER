# `scripts_train/` — training & ablation launchers

Shell entrypoints that **train** TIGER models (MIC pair-delta ablations and
toxin classical-ML feature ablations). Run all commands from the `TIGER/` root.

```bash
cd /path/to/TIGER
conda activate ccseg          # MIC (PyTorch + PyG)
# toxin classical ML may use:  PY=/home/ubuntu/anaconda3/envs/class/bin/python
```

---

## Layout

```text
scripts_train/
├── README.md
├── _ablation_common.sh              # shared ROOT / PY / SEED / GPU helpers
├── train_best_recipe.sh             # single best MIC recipe
├── run_ablation_01_modality.sh      # → runs_ablation/01_*
├── run_ablation_02_pair_balance.sh  # → runs_ablation/02_*
├── run_ablation_03_fusion.sh        # → runs_ablation/03_*
├── run_ablation_04_similarity.sh    # → runs_ablation/04_*
├── run_ablation_05_seed_stability.sh# → runs_ablation/05_*
├── run_ablation_all_mic.sh          # panels 01→05 sequential
├── run_ablation_toxin.sh            # → runs_ablation_toxin/{both,global,sequence}
└── run_wetlab_species_sim.sh        # → runs_wetlab/ (top-6 species × sim 0.3/0.7)
```

Python grid workers live under `TIGER/scripts/run_struct_s_*.py` and
`python -m code.main train` / `python -m code.toxin_filter.run`. These `.sh`
files are the stable user-facing wrappers.

---

## Seeds

| Context | Seed values | Folder naming |
|---------|-------------|---------------|
| Default MIC train / panels 01–04 | **`1`** | not in path |
| Seed stability (panel 05) | **`1, 2, 3, 4, 5`** | `…_seed1` … `…_seed5` |
| Toxin feature ablation | **`1`** | not in path |

Date-style seeds (e.g. `20260714`) are **not** used.

---

## Scripts

### `_ablation_common.sh`

Shared bootstrap (sourced by other scripts):

- Resolves `ROOT` (= `TIGER/`), sets `PYTHONPATH`
- Defaults: `PY` (ccseg), `SEED=1`, `GPUS="0 1"`, `GPU=0`, `FOREGROUND=0`
- Helpers: `_ablation_launch` (grid runners), `_eval_run` (used by `scripts_eval/`)

### `train_best_recipe.sh`

Trains the locked paper MIC recipe in one shot:

| Setting | Value |
|---------|-------|
| modalities | `gsh` |
| fusion | `attention` / `cross_qs` |
| structure | `s`, no node coords |
| similarity | `0.3` |
| pairing | unsigned, `delta_bin_width=1.0`, `pair_balance_num=10000` |
| seed | `1` |

Default out: `runs_ablation_repro/best_unsigned_bal10000/`.

```bash
bash code/scripts_train/train_best_recipe.sh
FOREGROUND=1 GPU=0 bash code/scripts_train/train_best_recipe.sh
```

### `run_ablation_01_modality.sh`

**Purpose:** modality combination ablation (`g,s,h,gs,gh,sh,gsh`).  
**Archive:** `runs_ablation/01_modality_ablation/`  
**Default out:** `runs_ablation_repro/01_modality_ablation/`  
**Backend:** `scripts/run_struct_s_modality_grid.py`

```bash
bash code/scripts_train/run_ablation_01_modality.sh
```

### `run_ablation_02_pair_balance.sh`

**Purpose:** per-bin sample cap × signed/unsigned (`bal ∈ {1k…20k}`).  
**Archive:** `runs_ablation/02_pair_balance_bin1/`  
**Default out:** `runs_ablation_repro/02_pair_balance_bin1/`  
**Backend:** `scripts/run_struct_s_binsize_grid.py`

### `run_ablation_03_fusion.sh`

**Purpose:** attention Q/KV modes + concat vs attention.  
**Archive:** `runs_ablation/03_fusion_methods/`  
**Default out:** `runs_ablation_repro/03_fusion_methods/{attn_grid,concat_grid}/`  
**Env:** `PART=attn|concat|all` (default `all`)  
**Backends:** `run_struct_s_sim_bin_qkv_grid.py`, `run_struct_s_concat_ablation.py`

### `run_ablation_04_similarity.sh`

**Purpose:** training-pair similarity threshold `{0.3, 0.5, 0.7}` @ bal=10000.  
**Archive:** `runs_ablation/04_similarity/`  
**Default out:** binsize repro + `runs_ablation_repro/04_similarity/`  
**Backends:** binsize grid + `run_struct_s_sim05_07_fusion_grid.py`

### `run_ablation_05_seed_stability.sh`

**Purpose:** CV stability across seeds at sim 0.3 and 0.7.  
**Archive / default out:** `runs_ablation/05_seed_stability/`  
**Grid:** `sim ∈ {0.3,0.7} × seed ∈ {1..5}` → 10 runs  
**Backend:** `scripts/run_struct_s_seed_stability_grid.py`

```bash
bash code/scripts_train/run_ablation_05_seed_stability.sh
# after runs finish:
PYTHONPATH=. python scripts/run_struct_s_seed_stability_grid.py --leaderboard-only
```

### `run_ablation_all_mic.sh`

Runs panels **01 → 05 sequentially** with `FOREGROUND=1` so jobs do not fight
over the same GPUs.

```bash
bash code/scripts_train/run_ablation_all_mic.sh
```

### `run_ablation_toxin.sh`

**Purpose:** toxin classical-ML feature-mode ablation (`both` / `global` / `sequence`).  
**Archive:** `runs_ablation_toxin/`  
**Default out:** `runs_ablation_toxin_repro/`  
**Settings:** HC50 threshold **512**, 5-fold CV, `--skip-dl`, seed **1**  
**Backend:** `python -m code.toxin_filter.run`

```bash
bash code/scripts_train/run_ablation_toxin.sh
MODE=both bash code/scripts_train/run_ablation_toxin.sh
```

### `run_wetlab_species_sim.sh`

**Purpose:** wetlab production MIC models on **full DBAASP** (no LL37 holdout).  
**Out:** `runs_wetlab/`  
**Grid:** top-6 species × `similarity ∈ {0.3, 0.7}` → **12 runs**  
**Locked:** `gsh` / `cross_qs` / `struct=s` / unsigned `bal=10000` / seed **1**  
**Backends:** `scripts/prepare_wetlab_dbassp_tables.py`, `scripts/run_wetlab_species_sim_grid.py`  
**Inference:** `pipeline/04_metric_ranking/rank_candidates.py`

```bash
bash code/scripts_train/run_wetlab_species_sim.sh
# refresh CV table after completion:
PYTHONPATH=. python scripts/run_wetlab_species_sim_grid.py --leaderboard-only
```

Species tables are written to `data/wetlab/train_<Species>.csv`.

---

## Common environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `PY` | `…/envs/ccseg/bin/python` | Python interpreter |
| `SEED` | `1` | RNG seed (panels 01–04 / toxin / best recipe) |
| `GPUS` | `0 1` | GPU ids for grid launchers |
| `GPU` | `0` | Single-GPU train (`train_best_recipe`) |
| `SLOTS_PER_GPU` | `1` | Concurrent jobs per GPU |
| `FOREGROUND` | `0` | `1` = block; `0` = `nohup` background |
| `OUT_ROOT` / `OUT_DIR` / `OUT_BASE` | (per script) | Override output directory |

---

## Related evaluation

After training, use wrappers in [`../scripts_eval/`](../scripts_eval/README.md)
(LL37 neighbor, APEX-GO pairs, toxin qlx227, CV summary).
