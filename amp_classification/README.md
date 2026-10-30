# E. coli AMP classification: 15-method benchmark

[![Verify AMP classification](https://github.com/haifangong/TIGER/actions/workflows/amp_classification.yml/badge.svg)](https://github.com/haifangong/TIGER/actions/workflows/amp_classification.yml)

This module replaces the original AMP activity classifier and demonstration benchmark in TIGER. The endpoint is **E. coli MIC <128 µg/mL**, with five final training seeds (**120–124**) and the same fixed **200-peptide holdout** for every method. It is separate from TIGER-pair's ΔMIC regression task.

The release contains the requested **15 methods**, all **75 test prediction files**, validation predictions and thresholds, **75 final-training recipes**, archived training-only CV recipes, fixed data/feature matrices, **seed123 checkpoints for all 15 methods**, numerical replay, CPU timing code, and publication figures. The original full study had 21 methods; DeepAMP, CatBoost-Recall, Das, CatBoost, APEX and Logistic are excluded from this release. No models were retrained to produce the published table below.

## Test results

Methods are ordered by test AUROC. Values are **mean ± sample SD over five training seeds**. Precision, recall and F1 use the threshold maximizing MCC on Vop, fixed before testing. AUPRC is **average precision**, not trapezoidal PR area. `NA` preserves undefined precision when a model predicts no positives. Dummy is an **uninformative class-prior baseline**.

| Method | AUROC | AUPRC | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ExtraTrees | 0.9608 ± 0.0006 | 0.9568 ± 0.0012 | 0.8534 ± 0.0165 | 0.9640 ± 0.0089 | 0.9052 ± 0.0073 |
| Cascade | 0.9605 ± 0.0018 | 0.9566 ± 0.0034 | 0.8207 ± 0.0055 | 0.9700 ± 0.0071 | 0.8891 ± 0.0038 |
| CatBoost-Tuned | 0.9575 ± 0.0018 | 0.9501 ± 0.0039 | 0.8560 ± 0.0164 | 0.9480 ± 0.0179 | 0.8994 ± 0.0073 |
| XGBoost | 0.9539 ± 0.0024 | 0.9480 ± 0.0040 | 0.8417 ± 0.0130 | 0.9440 ± 0.0270 | 0.8896 ± 0.0084 |
| HistGradientBoosting | 0.9516 ± 0.0000 | 0.9321 ± 0.0000 | 0.8393 ± 0.0000 | 0.9400 ± 0.0000 | 0.8868 ± 0.0000 |
| TinyCNN | 0.9509 ± 0.0076 | 0.9300 ± 0.0117 | 0.8648 ± 0.0283 | 0.9280 ± 0.0192 | 0.8950 ± 0.0187 |
| RBFSVM | 0.9465 ± 0.0000 | 0.9388 ± 0.0000 | 0.8205 ± 0.0000 | 0.9600 ± 0.0000 | 0.8848 ± 0.0000 |
| LightGBM | 0.9430 ± 0.0000 | 0.9225 ± 0.0000 | 0.8067 ± 0.0000 | 0.9600 ± 0.0000 | 0.8767 ± 0.0000 |
| RF | 0.9425 ± 0.0023 | 0.9316 ± 0.0039 | 0.8326 ± 0.0089 | 0.9340 ± 0.0114 | 0.8803 ± 0.0053 |
| AdaBoost | 0.9321 ± 0.0001 | 0.8973 ± 0.0002 | 0.7985 ± 0.0008 | 0.8720 ± 0.0045 | 0.8336 ± 0.0025 |
| LinearSVM | 0.9320 ± 0.0000 | 0.8917 ± 0.0000 | 0.8378 ± 0.0000 | 0.9300 ± 0.0000 | 0.8815 ± 0.0000 |
| GBDT | 0.9293 ± 0.0014 | 0.9037 ± 0.0064 | 0.8019 ± 0.0134 | 0.9620 ± 0.0084 | 0.8746 ± 0.0055 |
| EBM | 0.9263 ± 0.0019 | 0.8953 ± 0.0042 | 0.8172 ± 0.0255 | 0.9220 ± 0.0311 | 0.8658 ± 0.0062 |
| SimpleLogistic | 0.7793 ± 0.0001 | 0.7509 ± 0.0001 | 0.6207 ± 0.0000 | 0.9000 ± 0.0000 | 0.7347 ± 0.0000 |
| Dummy | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | NA | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

AMP and non-AMP results at all five operating points (`balanced`, `default`, `recall90`, `recall95`, `recall98`) are in [`results/tables/test_classwise_summary.csv`](results/tables/test_classwise_summary.csv). Per-seed values and confusion counts are in [`test_classwise_per_seed.csv`](results/tables/test_classwise_per_seed.csv). The class-prior baseline has AUROC and AUPRC of 0.5; its maximum-MCC tie-breaking rule selects an all-negative decision, rather than random labels.

## Dataset and validation

The cohort contains **10,962 unique canonical 6–50-residue sequences**, balanced at **5,481 AMP and 5,481 non-AMP labels**. It combines the original measured activity dataset, the historical external panel, and sampled background negatives. Measured sequences use their median log2 MIC label; the threshold is strictly `<7`, equivalent to MIC `<128 µg/mL`. Background negatives inherit their source-pool label; their numeric MIC remains missing. Measured records take precedence over exact negative-pool conflicts.

- **T:** 8,610 peptides for fitting and five-fold, training-only model selection.
- **Vcal:** 1,076 peptides for sigmoid probability calibration.
- **Vop:** 1,076 peptides for decision-threshold selection.
- **Test200:** 200 peptides, comprising 100 measured AMPs, 22 measured non-AMPs and 78 background negatives.

Test200 was sampled without replacement using a fixed random seed; its 100:100 balance occurred in that draw and was not imposed by stratified test sampling. Each training fold has a separate inner early-stopping split. Final training uses only T, calibration only Vcal, and threshold selection only Vop. The `recall95` and `recall98` thresholds maximize validation specificity subject to the target validation recall; the targets do not guarantee equivalent test or deployment recall.

This is a **retrospective random holdout from the merged pool**, not an untouched external cohort or a homology-disjoint split. Exact duplicates were removed; 113/200 test peptides have ≥80% identity to at least one training peptide. See [`data/test_similarity_audit.csv`](data/test_similarity_audit.csv). Seed SD describes training variation on this fixed holdout. The five-fold CV metrics are model-selection diagnostics, not nested-CV estimates of final generalization.

## Method implementations

| Method | Inputs and model |
| :--- | :--- |
| ExtraTrees | 40 F1 sequence descriptors; 500 extremely randomized trees, depth 16, minimum leaf size 2, feature fraction 0.5; fit-partition class weights. |
| Cascade | 553 AMPActiPred descriptors; Deep Forest cascade, up to four layers, two estimator pairs per layer, 100 trees, depth 12, minimum leaf size 3. |
| CatBoost-Tuned | 40 F1 descriptors; plain symmetric boosting, depth 10, learning rate 0.02, L2 1.0; 974 final trees. |
| XGBoost | 440 F2 descriptors; histogram boosting, depth 6, learning rate 0.05, subsample 0.7, L1 1, L2 10; 485 trees. |
| HistGradientBoosting | F2; up to 31 leaves, minimum leaf size 20, L2 1, learning rate 0.05; 340 iterations. |
| TinyCNN | Full-length padded sequence tokens; frozen one-hot embedding; convolutions of widths 3/5/7, 48 channels each; masked maximum/mean pooling; 16 epochs. |
| RBFSVM | Standardized F1 descriptors; radial-basis SVM, C=10, gamma=scale. |
| LightGBM | F1; 31 leaves, minimum child count 20, learning rate 0.02; 778 trees. |
| RF | F2; 200 random-forest trees, depth 12, minimum leaf size 3, square-root feature sampling. |
| AdaBoost | F2; discrete SAMME boosting of depth-3 trees, learning rate 1.0; 291 estimators. |
| LinearSVM | Imputed and standardized F2; linear SVM, C=0.1. |
| GBDT | F1; depth-3 gradient boosting, minimum leaf size 5, learning rate 0.03, subsample 0.7; 978 estimators. |
| EBM | F1; Explainable Boosting Machine with 10 interactions, 128 bins and eight outer bags; learning rate 0.03; internal validation/early stopping and a 5,000-round ceiling. |
| SimpleLogistic | Sequence length and charge proxy only; standardized logistic regression, C=1. |
| Dummy | Training class-prior probability; no sequence discrimination. |

F1 comprises 20 amino-acid composition values and 20 fixed sequence descriptors. F2 adds 400 dipeptide frequencies. No measured molecular weight, terminal-chemistry annotation, CFU, source identifier or experimental label enters these feature vectors. Each model's raw score is sigmoid-calibrated on Vcal. Frozen configs are the source of truth for all parameters; `core/v2` and `core/v3` preserve the original fitting functions, with filesystem/device adapters documented in [`provenance/source_adaptations.json`](provenance/source_adaptations.json). Shared historical helpers contain unused architectures; the public registry and training CLI accept only the 15 released methods.

## Reproduce tables without model dependencies

From the repository root, using Python 3.10:

```bash
python -m venv .venv-amp
source .venv-amp/bin/activate
python -m pip install -r amp_classification/requirements-replay.txt
python -m amp_classification verify
python -m amp_classification replay
```

Replay aligns IDs, sequences and labels, checks all 375 validation thresholds, verifies stored hard decisions, and independently recomputes 9,000 metric/count values across the seed-level and summary tables. Outputs go to `outputs/amp_replay/`; archived results are protected from overwrite. `provenance/manifest.json` records SHA256 checksums for the release payload.

## Predict new sequences

```bash
python -m pip install -r amp_classification/requirements.txt
python -m amp_classification predict --model CatBoost-Tuned \
  --input amp_classification/examples/peptides.csv \
  --output outputs/amp_predictions.csv

# End-to-end value check against all 200 archived seed123 predictions.
python -m amp_classification predict --model ExtraTrees \
  --input amp_classification/data/Test.csv \
  --output outputs/ExtraTrees_Test200.csv --check-reference
```

Input is a CSV with a `sequence` column or a FASTA file. Sequences must contain 6–50 uppercase canonical amino acids; the classifier does not truncate or silently remove symbols. Output includes the calibrated probability and all five fixed-threshold decisions. Probabilities describe this constructed balanced cohort, not the prevalence of AMP activity in arbitrary peptide libraries.

The convenience default and Step 1 example use **CatBoost-Tuned seed123** to replace the old CatBoost interface. This is not a newly selected winner. The full 21-method study's pre-test deployment rule selected CatBoost-Recall, which is excluded from this subset by request; that historical selection is not rewritten. All 15 seed123 checkpoints are available via `--model`. Seeds 120–124 have predictions and recipes; only seed123 weights are bundled to keep the release compact.

TinyCNN additionally requires the recorded PyTorch version:

```bash
python -m pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
```

Cascade uses a **separate Python 3.9 environment**, matching its actual training/inference environment:

```bash
conda create -n tiger-amp-cascade python=3.9 -y
conda activate tiger-amp-cascade
python -m pip install -r amp_classification/requirements-cascade-py39.txt
python -m amp_classification predict --model Cascade \
  --input amp_classification/data/Test.csv \
  --output outputs/Cascade_Test200.csv --check-reference
```

The released local acceptance covers all 15 raw-sequence seed123 predictions. Inference comparisons allow absolute probability error ≤2×10⁻⁶ across backends; archived CSVs define the paper tables. Model files use Python serialization and should be loaded from this trusted release or your own training outputs.

## Retrain an archived recipe

```bash
python -m amp_classification train \
  --config amp_classification/configs/final/CatBoost-Tuned/seed123.json \
  --out-dir outputs/amp_training/CatBoost-Tuned_seed123 --threads 4

python -m amp_classification train \
  --config amp_classification/configs/final/TinyCNN/seed123.json \
  --out-dir outputs/amp_training/TinyCNN_seed123 --device cuda
```

Use Cascade's Python 3.9 environment for Cascade recipes; the other methods use Python 3.10. TinyCNN originally trained on CUDA and calibrated on CPU. Final recipes retain the original CV-selected iteration counts; CV recipes under `configs/cv/` include the original fold and inner early-stopping identities. The same `train --config ...` command runs either type. Version-2 classic workers use four CPU threads, neural workers two; version-3 workers accept `--threads`. Output directories must be new/empty and outside the frozen package. Fresh stochastic training may differ numerically; it does not overwrite published predictions.

## CPU workload and figures

```bash
python -m amp_classification timing --model CatBoost-Tuned --threads 1 \
  --batches 1 200 1000 10000 --output outputs/amp_timing_CatBoost-Tuned.json
```

The full archived timing table includes one- and four-thread settings. The cost figure uses **single-thread, seed123, batch=200**, with median time over 30 repeats after three warm-ups. Timing includes raw-sequence feature computation, prediction and calibration, excluding model loading and file I/O. Workloads use fixed Vop sequences, separate from Test200. Hardware: AMD Ryzen 9 9950X; serial measurements on a shared server. New measurements are stored separately from the archived costs.

- [Sensitivity and false-positive burden (PDF)](figures/Sensitivity_and_false_positive_burden.pdf)
- [AMP AUPRC and CPU deployment cost (PDF)](figures/Accuracy_and_CPU_deployment_cost_AUPRC.pdf)
- [Both figures (PDF)](figures/AMP_classification_selected15_figures.pdf)
- [Executed, editable notebook](figures/Selected15_classification_figures.ipynb)

Both figures are 200 × 200 mm vector PDFs with embedded Arial, 1.5-pt axes/ticks and 6.5-pt major ticks. Install the figure requirements and an Arial font available on your system before regeneration:

```bash
python -m pip install -r amp_classification/requirements-figures.txt
python amp_classification/figures/plot_selected15.py
```

The notebook has one execution cell per figure and editable fonts, marker sizes and label positions. The figure source tables are restricted to the same 15 methods. Third-party attribution is in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
