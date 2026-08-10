# checkpoints/ablation — Paper ablation archive

This folder packages the **finished ablation runs** used for the manuscript (slim: `fold*_best.pt` + config/metrics only). Most panels lock training similarity at **0.3**; panel `04_similarity` compares **0.3 / 0.7** at fixed `bal=10000` (sim=0.5 omitted). Panel `05_seed_stability` re-trains the locked recipe at sim **0.3 / 0.7** across **5 seeds**. The invalid `cross_qs` sequence-only (`mod_s`) run is excluded.

| Ablation | Path | #points |
|----------|------|--------:|
| Modality combination | `01_modality_ablation/` | 6 |
| Per-bin sample cap (`pair_balance_num`), bin width fixed at 1.0 | `02_pair_balance_bin1/` | 12 |
| Multimodal fusion methods | `03_fusion_methods/` | 7 |
| Training-pair similarity threshold | `04_similarity/` | 2 |
| Seed stability (sim 0.3/0.7 × 5 seeds) | `05_seed_stability/` | 10 |
| Sequence encoding (`integer` / `embedding` / `onehot`) | `06_seq_encoding/` | 3 |

Index files:

- `MANIFEST.json` — machine-readable settings + per-run metadata
- `leaderboard.csv` — CV metrics for primary points (**30** rows, with per-run `seed`)
- `leaderboard_all_points.csv` — all **40** released dirs (includes seed-stability)
- `MANIFEST.json` — release inventory (excludes `mod_s`, `sim0p5`; seeds from configs)

Each panel folder (`01_`…`05_`) and each experiment point has its own `README.md`
describing the sweep, locked settings, and CV metrics for that directory.

---



## CV reporting protocol (checkpoint selection)

Within each GroupKFold split, `fold{k}_best.pt` is selected by validation
`selection_score` on fold `k`, then that checkpoint’s predictions on the **same**
validation fold enter the OOF CV aggregate.

Interpret published CV as **model-selection-aware OOF** (possible mild within-fold
optimistic bias), **not** as a fully nested outer test. Primary metric tables use
**raw OOF**; nested calibration is secondary. Seed-stability (`05_seed_stability/`)
probes variance across seeds `{1..5}`.

## 1. Common locked settings

All runs below share:

| Setting | Value |
|---------|-------|
| `similarity_threshold` | **0.3** |
| `structure_features` | `s` (AA node descriptors; no energy channel) |
| `include_node_coords` | `false` |
| `fusion` | `attention` |
| `fusion_attn_mode` | `cross_qs` (Q = sequence; K/V = graph + global) |
| `pair_interaction` | `diff` |
| `seq_encoding` | `integer` (A=1…Y=20) |
| `global_feature_scaling` / `node_feature_scaling` | `zscore` |
| `seed` | **panels 01–04: `123`** (from each `config.json`); **panel 05:** `{1,2,3,4,5}`; **panel 06:** `1`. Do not assume global seed=1. |
| `folds` | 5 |
| `lr` / `weight_decay` / `lr_scheduler` | `1e-3` / `0` / `cosine` |
| Base config | `code/configs/gsh_struct_s_base.json` |

**Data**

- Train/val CSV: `metadata/train_val_by_cfu_group_ug_per_mL.csv`
- Default test CSV (for neighbor eval hooks): `metadata/test_LL37_by_cfu_group_ug_per_mL.csv`
- Train PDBs: `data/3D_data_train_eva_Rosetta`
- Graph feature table: `data/metadata/features.txt`
- Shared graph cache: `artifacts/cache_ll37_holdout_cfu`

**Environment (as used)**

- Conda env: `ccseg` (Python 3.10, PyTorch + PyG)
- Entry point: `python -m code.main train ...` with `PYTHONPATH=<TIGER_ROOT>`

**Selection score (lower better)**

```text
selection_score = log2MAE + RSE - PCC - KCC
```

---

## 2. Ablation A — Modality combinations

**Directory:** `01_modality_ablation/`  
**Source:** `outputs_code_struct_s_modality_grid/`  
**Scripts:** `scripts/run_struct_s_modality_grid.py`, `scripts/launch_struct_s_modality_grid.sh`

### Locked for this ablation

- `pair_balance_num = 10000`
- `delta_bin_width = 1.0`
- `use_signed_sampling = false` (unsigned Δ bins)

### Sweep

`feature_modalities` letters in the encoder:

| Letter | Representation |
|--------|----------------|
| `g` | global features |
| `s` | sequence embedding |
| `h` | graph / structure GNN |

Points (each folder is a symlink):

```text
01_modality_ablation/mod_gsh/   # full model
01_modality_ablation/mod_gs/
01_modality_ablation/mod_sh/
01_modality_ablation/mod_gh/
01_modality_ablation/mod_g/
01_modality_ablation/mod_h/
```

### CV summary (from `leaderboard.csv`)

| point | PCC | log10MAE | RSE | score |
|-------|----:|---------:|----:|------:|
| mod_gsh | 0.696 | 0.437 | 0.516 | 0.788 |
| mod_gs | 0.679 | 0.444 | 0.539 | 0.858 |
| mod_sh | 0.673 | 0.447 | 0.548 | 0.891 |
| mod_g | 0.568 | 0.501 | 0.678 | 1.399 |
| mod_gh | 0.562 | 0.504 | 0.684 | 1.420 |
| mod_h | 0.544 | 0.514 | 0.704 | 1.506 |

Full precision: `leaderboard.csv` / each run’s `results/summary.json`.

### Note on sequence-only

The invalid `cross_qs` sequence-only run (`mod_s`; Q from sequence with zeroed K/V) is **excluded** from this release. Prefer multi-modality rows (`gsh/gs/sh/...`). Fairer single-modality comparisons use concat fusion (see `03_fusion_methods/concat_gsh_bal10000` and local `outputs_code_struct_s_concat_ablation/`).

### Reproduce

```bash
cd /path/to/TIGER
conda activate ccseg
export PYTHONPATH=.

# full grid (2 GPUs by default in launch script)
bash scripts/launch_struct_s_modality_grid.sh

# or one point, e.g. gsh:
python -m code.main train \
  --config code/configs/gsh_struct_s_base.json \
  --out-dir checkpoints/ablation/01_modality_ablation/mod_gsh_repro \
  --name mod_gsh \
  --structure-features s \
  --feature-modalities gsh \
  --fusion attention \
  --fusion-attn-mode cross_qs \
  --no-include-node-coords \
  --similarity-threshold 0.3 \
  --delta-bin-width 1.0 \
  --pair-balance-num 10000 \
  --gpu 0
```

---

## 3. Ablation B — Per-bin sample count (bin width = 1.0)

**Directory:** `02_pair_balance_bin1/`  
**Source:** `outputs_code_struct_s_binsize_grid/`  
**Scripts:** `scripts/run_struct_s_binsize_grid.py`, `scripts/launch_struct_s_binsize_grid.sh`

### Locked for this ablation

- `feature_modalities = gsh`
- `delta_bin_width = 1.0` (not swept)

### Sweep

1. `use_signed_sampling ∈ {false, true}`
2. `pair_balance_num ∈ {1000, 2000, 4000, 8000, 10000, 20000}`

```text
02_pair_balance_bin1/unsigned_bal{1000,2000,4000,8000,10000,20000}/
02_pair_balance_bin1/signed_bal{1000,2000,4000,8000,10000,20000}/
```

### Binning semantics (read carefully)

Training pairs are capped **per Δ-bin** by `pair_balance_num`:

| Mode | Bin key | Role of `delta_bin_width=1.0` |
|------|---------|------------------------------|
| **unsigned** (`use_signed_sampling=false`) | `round(\|Δ\| × 100)` | **Unused** for the key; width is kept at 1.0 in config for bookkeeping only |
| **signed** (`use_signed_sampling=true`) | `floor(Δ / 1.0)` | **Active** — width 1.0 defines signed bins |

Manuscript “bin size / samples per bin” curves should state which mode is plotted.

**Unsigned arm (primary):**

| point | PCC | log10MAE | RSE | score |
|-------|----:|---------:|----:|------:|
| unsigned_bal1000 | 0.695 | 0.436 | 0.517 | 0.783 |
| unsigned_bal2000 | 0.705 | 0.431 | 0.502 | 0.733 |
| unsigned_bal4000 | 0.701 | 0.432 | 0.509 | 0.749 |
| unsigned_bal8000 | 0.697 | 0.432 | 0.515 | 0.764 |
| unsigned_bal10000 | **0.705** | **0.429** | **0.503** | **0.729** |
| unsigned_bal20000 | 0.694 | 0.437 | 0.518 | 0.789 |

Best CV point in this archive:

```text
02_pair_balance_bin1/unsigned_bal10000/
  CV: PCC ≈ 0.705, selection_score ≈ 0.729
```

### Reproduce

```bash
cd /path/to/TIGER
conda activate ccseg
export PYTHONPATH=.

bash scripts/launch_struct_s_binsize_grid.sh

# example: unsigned, bal=10000
python -m code.main train \
  --config code/configs/gsh_struct_s_base.json \
  --out-dir checkpoints/ablation/02_pair_balance_bin1/unsigned_bal10000_repro \
  --name unsigned_bin1p0_bal10000 \
  --structure-features s \
  --feature-modalities gsh \
  --fusion attention \
  --fusion-attn-mode cross_qs \
  --no-include-node-coords \
  --similarity-threshold 0.3 \
  --delta-bin-width 1.0 \
  --pair-balance-num 10000 \
  --gpu 0
# add --use-signed-sampling for the signed arm
```

---

## 4. Ablation C — Multimodal fusion methods

**Directory:** `03_fusion_methods/`  
**Sources:** `outputs_code_struct_s_sim_bin_qkv/`, `outputs_code_struct_s_concat_ablation/`, `outputs_code_struct_s_binsize_grid/`  
**Scripts:** `scripts/run_struct_s_sim_bin_qkv_grid.py`, `scripts/run_struct_s_concat_ablation.py`

Locked for this ablation (all points): `sim=0.3`, `modalities=gsh`, `struct=s`, `coords=false`, unsigned binning, `delta_bin_width=1.0`.

This folder has **two matched panels**. Do not mix `bal` across panels when comparing.

### Panel C1 — Attention fusion modes (`pair_balance_num=1000`)

Compares how the three modality tokens are fused under `fusion=attention`:

| Letter in mode | Role |
|----------------|------|
| `g` | global |
| `s` | sequence |
| `h` | graph / structure |

| Mode | Meaning |
|------|---------|
| `cross_q*` | Query from one modality; Key/Value from the other two (`cross_qs` / `cross_qh` / `cross_qg`) |
| `self_*` | Self-attention over stacked tokens; suffix = token order (`self_gsh`, `self_sgh`) |

| point | mode | PCC | log10MAE | RSE | score |
|-------|------|----:|---------:|----:|------:|
| attn_cross_qs_bal1000 | cross_qs | **0.694** | 0.437 | 0.520 | **0.789** |
| attn_self_sgh_bal1000 | self_sgh | 0.691 | 0.438 | 0.524 | 0.794 |
| attn_self_gsh_bal1000 | self_gsh | 0.690 | 0.441 | 0.529 | 0.824 |
| attn_cross_qh_bal1000 | cross_qh | 0.683 | 0.443 | 0.535 | 0.843 |
| attn_cross_qg_bal1000 | cross_qg | 0.667 | 0.455 | 0.560 | 0.922 |

**Winner in this panel:** `cross_qs` (Q=sequence, KV=graph+global).

### Panel C2 — Concat vs attention (`pair_balance_num=10000`)

Same pairing recipe as the best binsize run; only the fusion operator changes:

| point | fusion | attn mode | PCC | log10MAE | RSE | score |
|-------|--------|-----------|----:|---------:|----:|------:|
| attn_cross_qs_bal10000 | attention | cross_qs | **0.705** | **0.429** | **0.503** | **0.729** |
| concat_gsh_bal10000 | concat | (n/a) | 0.688 | 0.440 | 0.528 | 0.819 |

Attention (`cross_qs`) outperforms simple concatenation under the matched bal=10000 setting (ΔPCC ≈ +0.017).

### Reproduce

```bash
cd /path/to/TIGER
conda activate ccseg
export PYTHONPATH=.

# attention mode grid (includes sim×bin×mode; use sim0.3 bin1.0 names)
bash scripts/launch_struct_s_sim_bin_qkv_grid.sh

# concat gsh point
bash scripts/launch_struct_s_concat_ablation.sh

# single attention cross_qs example (bal=1000)
python -m code.main train \
  --config code/configs/gsh_struct_s_base.json \
  --out-dir checkpoints/ablation/03_fusion_methods/attn_cross_qs_bal1000_repro \
  --name sim0p3_bin1p0_cross_qs \
  --structure-features s --feature-modalities gsh \
  --fusion attention --fusion-attn-mode cross_qs \
  --no-include-node-coords \
  --similarity-threshold 0.3 --delta-bin-width 1.0 \
  --pair-balance-num 1000 --gpu 0
```

---

## 4b. Ablation D — Similarity threshold (`bal=10000`)

**Directory:** `04_similarity/`  
**Sources:** `outputs_code_struct_s_binsize_grid/`, `outputs_code_struct_s_sim05_07_fusion/`  
**Scripts:** `scripts/run_struct_s_binsize_grid.py`, `scripts/run_struct_s_sim05_07_fusion_grid.py`

Locked: `gsh`, `cross_qs`, `struct=s`, `coords=false`, unsigned, `delta_bin_width=1.0`, **`pair_balance_num=10000`**.

Sweep: `similarity_threshold ∈ {0.3, 0.7}` (minimum NW similarity for training pairs). The sim=0.5 point is not redistributed.

| point | sim | PCC | log10MAE | RSE | score |
|-------|----:|----:|---------:|----:|------:|
| sim0p3_bal10000 | 0.3 | **0.705** | **0.429** | **0.503** | **0.729** |
| sim0p7_bal10000 | 0.7 | 0.470 | 0.537 | 0.779 | 1.775 |

Higher training similarity thresholds shrink the pair pool and hurt CV correlation; **0.3** is best under this matched bal=10000 setting.

---

## 4c. Ablation E — Seed stability (sim 0.3 / 0.7)

**Directory:** `05_seed_stability/`  
**Scripts:** `code/scripts_train/run_ablation_05_seed_stability.sh`, `scripts/run_struct_s_seed_stability_grid.py`

Locked: `gsh`, `cross_qs`, `struct=s`, `coords=false`, unsigned, `delta_bin_width=1.0`, **`pair_balance_num=10000`**.

Sweep: `similarity_threshold ∈ {0.3, 0.7}` × `seed ∈ {1, 2, 3, 4, 5}` → **10 runs** on GPUs 0 & 1.

Folder names use plain integers (`…_seed1` … `…_seed5`), not date-style seeds.

Primary deliverable: `stability_summary.json` (mean ± std of 5-fold CV ensemble metrics across seeds). See `05_seed_stability/README.md`.

```bash
cd TIGER
bash code/scripts_train/run_ablation_05_seed_stability.sh
# refresh tables after completion:
PYTHONPATH=. python scripts/run_struct_s_seed_stability_grid.py --leaderboard-only
```

---

## 5. What’s inside each point folder

Each subdirectory is a **full local copy** of the experiment and contains:

```text
config.json              # full hyper-parameters
checkpoints/             # fold{1..5}_best.pt only (slim release)
results/summary.json     # CV metrics (+ calibrator, etc.)
results/calibrator.json  # if present
```

There are **40** experiment points under `checkpoints/ablation/` in this slim release (each with five `fold*_best.pt`; `leaderboard.csv` lists the primary CV rows). `fold*_last.pt`, intermediates, and logs are omitted. Excluded: `mod_s`, `sim0p5_bal10000`.

Verify a run:

```bash
ls checkpoints/ablation/01_modality_ablation/mod_gsh/checkpoints/
python -c "import json; print(json.load(open('checkpoints/ablation/01_modality_ablation/mod_gsh/results/summary.json'))['cv'])"
```

Rebuild the index after adding runs:

```bash
python - <<'PY'
# optional: regenerate MANIFEST/leaderboard from symlinks if needed
print('see MANIFEST.json / leaderboard.csv')
PY
```

---

## 6. Recommended paper mapping

| Figure / table topic | Use these folders |
|-------------------|-------------------|
| Modality ablation | `01_modality_ablation/mod_*` |
| Effect of samples-per-bin | `02_pair_balance_bin1/unsigned_bal*` (primary); `signed_bal*` as optional signed-binning panel |
| Attention fusion modes (Q/KV & token order) | `03_fusion_methods/attn_*_bal1000` |
| Concat vs attention | `03_fusion_methods/{concat_gsh_bal10000,attn_cross_qs_bal10000}` |
| Similarity threshold (0.3 / 0.7) | `04_similarity/sim0p{3,7}_bal10000` |
| Seed / CV stability (sim 0.3 & 0.7) | `05_seed_stability/sim0p*_bal10000_seed*` |
| Sequence encoding (integer vs embedding vs onehot) | `06_seq_encoding/seq_*` |
| Default / best recipe cited elsewhere | `02_pair_balance_bin1/unsigned_bal10000` (= `03_fusion_methods/attn_cross_qs_bal10000` = `04_similarity/sim0p3_bal10000`) |

TIGER vs EvoGradient external comparison (LL37 + APEXGO pair-delta, including
EvoGradient absolute-MIC → pair-Δ conversion) is archived at:

```text
checkpoints/ablation/04_similarity/external_eval_tiger_vs_evo/
```

Reproduce: `bash code/scripts_eval/eval_tiger_vs_evo.sh`.

**Calibration note:** historical `results/summary.json` `cv` fields in this archive
may still reflect the legacy in-sample OOF calibrator. New training runs report
**raw OOF** as primary (`code/train.py`).

---

## 7. Provenance

| Archive path | Original path |
|--------------|---------------|
| `01_modality_ablation/mod_<X>` | `outputs_code_struct_s_modality_grid/modality_grid__mod_<X>` |
| `02_pair_balance_bin1/unsigned_bal<N>` | `outputs_code_struct_s_binsize_grid/binsize_grid__unsigned_bin1p0_bal<N>` |
| `02_pair_balance_bin1/signed_bal<N>` | `outputs_code_struct_s_binsize_grid/binsize_grid__signed_bin1p0_bal<N>` |
| `03_fusion_methods/attn_<mode>_bal1000` | `outputs_code_struct_s_sim_bin_qkv/struct_s_grid__sim0p3_bin1p0_<mode>` |
| `03_fusion_methods/concat_gsh_bal10000` | `outputs_code_struct_s_concat_ablation/concat_ablation__mod_gsh` |
| `03_fusion_methods/attn_cross_qs_bal10000` | `outputs_code_struct_s_binsize_grid/binsize_grid__unsigned_bin1p0_bal10000` |
| `04_similarity/sim0p3_bal10000` | `outputs_code_struct_s_binsize_grid/binsize_grid__unsigned_bin1p0_bal10000` |
| `04_similarity/sim0p7_bal10000` | `outputs_code_struct_s_sim05_07_fusion/sim_bal_fusion__cross_qs_sim0p7_bal10000` |

`MANIFEST.json` still records `source_path` (original `outputs_*` location) for provenance. The weights under `checkpoints/ablation/**/checkpoints/` are independent copies and are sufficient for inference/evaluation without the original trees.
