# Environment setup

This document explains how a new user can create the environments needed to run the TIGER pipeline on their own machine.

## Recommended environments

| Environment file | Used for | Create command |
|------------------|----------|----------------|
| `environment.yml` | Steps 1–2 + metrics + notebooks | `conda env create -f environment.yml` |
| `environment_helix.yml` | Step 3a HelixFold inference | `conda env create -f environment_helix.yml` |
| `environment_rosetta.yml` | Step 3b FastRelax scaffold | `conda env create -f environment_rosetta.yml` |

Alternatively for Steps 1–2 only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step-by-step (Steps 1–2)

```bash
cd TIGER/pipeline
conda env create -f environment.yml
conda activate tiger-pipeline

# optional: register Jupyter kernel
python -m ipykernel install --user --name tiger-pipeline --display-name "Python (tiger-pipeline)"
```

Verify:

```bash
python -c "import catboost, torch, Bio, sklearn; print('ok')"
```

## Step 3a (HelixFold)

```bash
conda env create -f environment_helix.yml
conda activate tiger-helix
# Install a CUDA-compatible PaddlePaddle build for your GPU/driver.
# Example (adjust version to your CUDA):
#   python -m pip install paddlepaddle-gpu==2.4.2

cd 03_structure_prediction/weights
# Either symlink an existing checkpoint, or download:
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/HelixFold-Single/helixfold-single.pdparams
```

If you see CUDA library conflicts:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## Step 3b (PyRosetta)

Install PyRosetta according to [pyrosetta.org](https://www.pyrosetta.org/downloads) into `tiger-rosetta` (or any env named `rosetta` locally).

```bash
conda activate tiger-rosetta
python -c "import pyrosetta; print('ok')"
```

## Hardware notes

- Steps 1–2 run on CPU; GPU optional for toxicity inference.
- HelixFold benefits strongly from GPU.
- FastRelax is CPU-bound and can use multiple workers.

## Offline / institutional clusters

1. Create the conda envs on a login node with internet.
2. Pack with `conda-pack` if compute nodes are offline.
3. Place HelixFold weights under `03_structure_prediction/weights/`.
4. Keep absolute paths out of scripts; use the CLI flags documented in each step README.
