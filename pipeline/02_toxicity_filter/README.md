# Step 2 — Hemolytic / Toxicity Filter

Score peptide candidates for hemolytic / host toxicity and split them into toxin vs non-toxin subsets.

## Purpose

- Consume Step-1 positive mutants
- Predict toxicity probability with a pretrained FusionPeptide model (`mode=101`: sequence + physicochemical features)
- Keep non-toxic candidates for structure prediction (Step 3)

## Folder structure

```text
02_toxicity_filter/
├── README.md
├── demo.ipynb
├── requirements.txt
├── infer_toxin.py           # main entry point
├── dataset.py               # slim inference dataset
├── network.py               # slim FusionPeptide (GRU + MLP)
├── checkpoints/
│   ├── model_1.pth          # pretrained weights
│   └── config.json          # training/inference config
├── examples/
│   └── sample_input.csv     # tiny Step-1 style input
└── outputs/
```

## Environment

Recommended conda env: `ccseg`

```bash
conda activate ccseg
pip install -r requirements.txt
```

## Quick start

```bash
cd 02_toxicity_filter
python infer_toxin.py \
  --csv ../01_mutation_search/examples/sample_input.csv \
  --out-dir outputs \
  --threshold 0.5
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--csv` | Input CSV (headerless `seq[,...]` from Step 1) | required |
| `--out-dir` | Output directory | `outputs` |
| `--model-path` | Checkpoint | `checkpoints/model_1.pth` |
| `--threshold` | Toxicity probability cutoff | `0.5` |
| `--max-len` | Max sequence length | `30` |
| `--batch-size` | Inference batch size | `32` |
| `--mode` | Modality bits (`seq,voxel,globf`) | `101` |
| `--q-encoder` | Sequence encoder | `gru` |
| `--cpu` | Force CPU | off |

## Outputs

Given an input stem such as `KSMLKSMK_positive_2`:

| File | Content |
|------|---------|
| `KSMLKSMK_toxins.csv` | Predicted toxins |
| `KSMLKSMK_non_toxins.csv` | Predicted non-toxins (**input to Step 3**) |
| `KSMLKSMK_all.csv` | All scored sequences |

CSV columns include `sequence`, `tox_prob`, and physicochemical descriptors.

## Notes

- This slim package supports **mode=101 only** (no 3D voxel encoder / no `mamba_ssm` dependency).
- Sequences shorter than 6 or longer than `max-len` residues are skipped.
- Interactive walkthrough: open `demo.ipynb`.

## Source

Cleaned from POAP `2_hemo_docoupling` (`infer_toxin.py`, `dataset.py`, `network.py`).


## Classification metrics (CV + test)

```bash
python evaluate_models.py \
  --csv examples/labeled_toxicity_demo.csv \
  --sequence-col sequence \
  --label-col label \
  --n-splits 5 \
  --test-size 0.2 \
  --out-dir outputs/metrics_demo \
  --eval-pretrained
```

This writes fold-level and held-out test metrics for classical baselines, and optionally for the bundled FusionPeptide checkpoint.

Custom labeled CSVs only need `sequence` + binary `label` columns.
