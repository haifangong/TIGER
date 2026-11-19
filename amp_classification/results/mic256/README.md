# E. coli activity classification at MIC <256 µg/mL

Fifteen classifiers were fitted and evaluated with the MIC256 endpoint using seeds 120–124. This release contains the numerical results underlying both panels of Figure S7. CatBoost-Tuned uses the original Figure S7 recipe (974 trees, depth 10, learning rate 0.02 and L2 regularization 1). All model weights, probability calibrators and operating thresholds were fitted using the MIC256 labels.

## Data and evaluation

The benchmark uses the same unique peptide sequences and partitions as the [MIC128 benchmark](../../README.md). A measured sequence is positive when its median log₂ MIC is below 8, with MIC expressed in µg/mL. A sequence at exactly 256 µg/mL is negative. Background negatives inherit their source label and have no measured MIC.

| Partition | Peptides | AMP | Non-AMP | Use |
| --- | ---: | ---: | ---: | --- |
| T | 8,610 | 5,022 | 3,588 | Model fitting |
| Vcal | 1,076 | 633 | 443 | Sigmoid calibration |
| Vop | 1,076 | 627 | 449 | Decision threshold selection |
| Test200 | 200 | 112 | 88 | Fixed random holdout |

Test200 includes 112 measured active peptides, 10 measured inactive peptides and 78 background negatives. All five seeds use the same holdout. In the ranking, AUROC and AMP average precision use continuous probabilities. Precision, recall and F1 use the seed-specific threshold that maximized MCC on Vop. The high-recall operating points maximize specificity on Vop subject to the indicated minimum recall. Standard deviations describe variation across training seeds.

## Results

TinyCNN ranked first by mean Test200 AUROC (0.9852 ± 0.0033), with AMP average precision of 0.9862 ± 0.0040.

- [All 15 methods ranked by Test200 AUROC](ranked_test200.csv)
- [AMP and non-AMP results at each operating point](test_classwise_summary.csv) and [the 122-peptide measured-MIC subset](measured_classwise_summary.csv)
- [Per-seed metrics and confusion counts](metrics_per_seed.csv), [75 per-peptide prediction files](predictions/) and [validation-selected thresholds](frozen_thresholds.json)
- [Figure S7A sensitivity and false-positive rates](sensitivity_source_data.csv) and [Figure S7B AMP average precision and CPU time](accuracy_cost_source_data.csv)
- [Single-thread CPU timing summaries](cpu_timing.csv) and [individual measurements](cpu_timing_repeats.csv). Seed 123 was timed on a batch of 200 peptides from Vop. Each measurement includes sequence feature extraction, inference and calibration. Times are in milliseconds, with three warm-ups and 30 timed repeats.
- [Partition counts by source and label](partition_counts.csv) and [model configurations and protocol](protocol.json)

The prediction tables contain the MIC256 label (`y`), calibrated AMP probability and binary decisions for all five frozen operating points. The parent module's packaged checkpoints and default prediction interface use MIC128.
