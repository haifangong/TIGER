# TIGER: Therapeutic-Index-Guided Exploration and Refinement for Antimicrobial Peptide Design

This repository provides a **reproducible, community-oriented** release of the TIGER computational pipeline.

## What is included

```text
.
├── README.md                 # this file
├── code/                     # modular MIC pair-delta model (train / evaluate / infer)
└── pipeline/                 # Steps 1–3 early-stage discovery pipeline
    ├── 01_mutation_search/   # mutational search + activity pre-filter
    ├── 02_toxicity_filter/   # hemolytic / toxicity filtering
    ├── 03_structure_prediction/  # HelixFold-Single + optional Rosetta relax
    ├── common/               # shared classification metrics
    ├── docs/                 # setup + custom-data guides
    └── notebooks/            # full end-to-end demo notebook
```

## Quick start

### A) Discovery pipeline (Steps 1–3)

```bash
cd pipeline
conda env create -f environment.yml
conda activate tiger-pipeline

# Step 1 — mutational search
cd 01_mutation_search
python search_mutations.py --sequence KSMLKSMK --search_length 1 --output_dir outputs/demo

# Step 2 — toxicity filter
cd ../02_toxicity_filter
python infer_toxin.py --csv ../01_mutation_search/outputs/demo/KSMLKSMK_positive_1.csv --out-dir outputs/demo

# Step 3 — structure prediction (separate HelixFold env; see pipeline/docs/SETUP.md)
cd ../03_structure_prediction
python infer_batch.py --csv_file ../02_toxicity_filter/outputs/demo/KSMLKSMK_non_toxins.csv --output_dir outputs/pdb/demo
```

Full walkthrough:

- Environment setup: [`pipeline/docs/SETUP.md`](pipeline/docs/SETUP.md)
- Custom / user-defined data: [`pipeline/docs/CUSTOM_DATA.md`](pipeline/docs/CUSTOM_DATA.md)
- End-to-end notebook: [`pipeline/notebooks/00_full_pipeline_demo.ipynb`](pipeline/notebooks/00_full_pipeline_demo.ipynb)
- Pipeline overview: [`pipeline/README.md`](pipeline/README.md)

### B) Pair-delta MIC model (`code/`)

```bash
cd code
# from the repository root with PYTHONPATH=.
export PYTHONPATH=..
python -m code.main train --config code/configs/default_fusion_attention.json --gpu 0
```

See [`code/README.md`](code/README.md) for evaluate / infer usage.

## Classification metrics

For antimicrobial-activity and toxicity classifiers, report the full suite across CV folds and the held-out test set:

**Accuracy, Precision, Recall, F1, MCC, AUC-ROC, AUC-PR**

```bash
cd pipeline/01_mutation_search
python evaluate_models.py --csv examples/labeled_activity_demo.csv --out-dir outputs/metrics

cd ../02_toxicity_filter
python evaluate_models.py --csv examples/labeled_toxicity_demo.csv --out-dir outputs/metrics
```

## Notes

- HelixFold weights (`~4.5 GB`) are **not** vendored. Download or symlink into `pipeline/03_structure_prediction/weights/` (see pipeline Step-3 README).
- Steps 1–2 and Step 3 intentionally use separate conda environments because PaddlePaddle / PyRosetta stacks conflict with the PyTorch scientific stack.

## Citation

If you use this repository, please cite the corresponding TIGER study.
