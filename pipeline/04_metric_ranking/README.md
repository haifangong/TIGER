# Step 4 — MIC metric ranking (wetlab TIGER models)

Rank candidate AMPs by **predicted MIC** using species-specific TIGER pair-delta
models trained on the **full DBAASP** set (`checkpoints/wetlab/`).

## Purpose

- Take non-toxin candidates (Step 2) that already have 3D structures (Step 3)
- Score them with a wetlab model for a chosen species × similarity setting
- Rank by predicted potency (lower MIC first)

## Folder structure

```text
04_metric_ranking/
├── README.md
├── requirements.txt
├── rank_candidates.py      # main entry point
├── examples/
│   └── sample_candidates.csv
└── outputs/                # default runtime outputs
```

## Prerequisites

1. Train wetlab models (12 runs: top-6 species × sim `{0.3, 0.7}`):

```bash
cd /path/to/TIGER
conda activate ccseg
bash code/scripts_train/run_wetlab_species_sim.sh
```

Outputs land in `TIGER/checkpoints/wetlab/sim0p3_<Species>/` and `sim0p7_<Species>/`.

2. PDBs for the template and every candidate named `{SEQUENCE}.pdb`
   (HelixFold / Rosetta outputs from Step 3, or the train PDB store).

## Environment

```bash
conda activate ccseg   # same as TIGER MIC training (PyTorch + PyG)
pip install -r requirements.txt
```

## Quick start (template mode)

Template mode is the recommended wetlab path: you know the WT/template MIC and
want mutants ranked by predicted improvement.

```bash
cd 04_metric_ranking
PYTHONPATH=../.. python rank_candidates.py \
  --csv examples/sample_candidates.csv \
  --species Escherichia_coli \
  --similarity 0.3 \
  --pdb-dir /path/to/pdbs \
  --template GIGKFLHSAKKFGKAFVGEIMNS \
  --template-mic 8.0 \
  --out-dir outputs/demo \
  --gpu 0
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--csv` | Candidate sequences | required |
| `--species` | Species slug matching `checkpoints/wetlab` | required |
| `--similarity` | `0.3` or `0.7` | `0.3` |
| `--pdb-dir` | Directory of `{SEQ}.pdb` | required |
| `--mode` | `template` \| `neighbor` | `template` |
| `--template` | WT / template sequence | required in template mode |
| `--template-mic` | Template MIC (µg/mL) | required in template mode |
| `--runs-root` | Wetlab run root | `TIGER/checkpoints/wetlab` |
| `--top-k` | Keep only top-K rows | `0` (all) |
| `--gpu` / `--cpu` | Device | GPU 0 |

### Neighbor mode

When the template MIC is unknown, estimate absolute MIC from DBAASP neighbor
anchors in the wetlab training pool:

```bash
PYTHONPATH=../.. python rank_candidates.py \
  --csv examples/sample_candidates.csv \
  --species Staphylococcus_aureus \
  --similarity 0.7 \
  --pdb-dir /path/to/pdbs \
  --mode neighbor \
  --out-dir outputs/demo_neighbor
```

## Outputs

| File | Content |
|------|---------|
| `ranked_<species>_sim0p3.csv` | Ranked table (lower `pred_MIC_ug_mL` = more potent) |
| `run_meta.json` | Species / sim / mode metadata |

Key columns (template mode):

- `pred_delta_log2_template_minus_query`
- `pred_MIC_ug_mL`
- `pred_MIC_improvement_vs_template` (`>1` ⇒ more potent than template)

Ensemble: mean over `fold{1..5}_best.pt` (+ optional calibrator).

## Species available after training

Top-6 by DBAASP MIC coverage (full table, no LL37 holdout):

1. `Escherichia_coli`
2. `Staphylococcus_aureus`
3. `Pseudomonas_aeruginosa`
4. `Bacillus_subtilis`
5. `Klebsiella_pneumoniae`
6. `Staphylococcus_epidermidis`

See `TIGER/data/wetlab/species_manifest.json`.
