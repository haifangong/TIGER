# Step 3 — Structure Prediction & Relaxation

Predict 3D peptide structures with **HelixFold-Single**, then optionally refine them with **PyRosetta FastRelax**.

## Purpose

- Convert Step-2 non-toxic sequences into PDB structures
- Optionally relax predicted PDBs for downstream scoring (e.g., Step 4 Siamese MIC models)

## Folder structure

```text
03_structure_prediction/
├── README.md
├── demo.ipynb
├── requirements.txt
├── infer_batch.py           # CSV -> many PDBs
├── infer_single.py          # one sequence -> one PDB
├── relax.py                 # PyRosetta FastRelax
├── alphafold_paddle/        # HelixFold model code
├── tape/                    # protein language-model components
├── utils/                   # model wrappers
├── model_configs/           # JSON configs
├── weights/
│   └── helixfold-single.pdparams   # symlink or downloaded weights (~4.5 GB)
├── examples/
│   └── sample_sequences.csv
└── outputs/
    ├── pdb/
    └── relaxed/
```

## Environment

This step typically uses **two** conda environments:

| Sub-step | Env | Notes |
|----------|-----|-------|
| HelixFold inference | `helix` or `helix_test` | PaddlePaddle + CUDA matching your GPU |
| Rosetta relaxation | `rosetta` | Requires `pyrosetta` |

```bash
# HelixFold
conda activate helix
# If CUDA library conflicts appear:
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Rosetta
conda activate rosetta
```

Install Python deps (HelixFold side):

```bash
pip install -r requirements.txt
# Plus a compatible PaddlePaddle build for your CUDA / Python version.
```

### Model weights

Weights are large and are **not vendored** here.

Current default path:

```text
weights/helixfold-single.pdparams
```

On this machine it is a symlink to the POAP checkpoint. If missing, download:

```bash
cd weights
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/HelixFold-Single/helixfold-single.pdparams
```

## Quick start

### A) Batch prediction from Step-2 CSV

```bash
cd 03_structure_prediction
conda activate helix
python infer_batch.py \
  --csv_file examples/sample_sequences.csv \
  --output_dir outputs/pdb/demo \
  --init_model weights/helixfold-single.pdparams
```

### B) Single template sequence

```bash
python infer_single.py \
  --seq KRIVQRIKDFLR \
  --output_dir outputs/pdb/demo
```

### C) Relax predicted PDBs

```bash
conda activate rosetta
python relax.py \
  --input_dir outputs/pdb/demo \
  --output_dir outputs/relaxed/demo
```

## Inputs / outputs

| Stage | Input | Output |
|-------|-------|--------|
| `infer_batch.py` | CSV with `sequence` column (or headerless seq) | `<SEQ>.pdb` files |
| `infer_single.py` | `--seq` | one PDB |
| `relax.py` | directory of PDBs | relaxed PDBs (same filenames) |

## Notes

- Prefer `--skip_exist` (default) for long batch jobs so restarts resume cleanly.
- For large CSVs, predict on GPU and keep Rosetta parallelization moderate (`--workers`).
- Interactive walkthrough: open `demo.ipynb`.

## Source

Cleaned from POAP `3_helixfold` (`infer.py`, `infer_single.py`, `relax.py`, HelixFold libraries).


## Custom user sequences

No sequences are hardcoded. Provide either:

1. a CSV with a `sequence` column, or
2. a headerless CSV whose first column is the sequence (e.g. Step-2 outputs).

```bash
python infer_batch.py --csv_file /path/to/my_sequences.csv --output_dir outputs/pdb/custom
python infer_single.py --seq MYSEQUENCE --output_dir outputs/pdb/custom
```

Environment setup details: `../docs/SETUP.md`. Custom-data guide: `../docs/CUSTOM_DATA.md`.
