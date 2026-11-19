# Human erythrocyte hemolysis at HC50 ≤ 1,024 µg/mL

This numerical release evaluates 17 internal classifiers for human erythrocyte hemolysis. A measured positive has HC50 ≤ 1,024 µg/mL, including the boundary value. A measured negative has HC50 > 1,024 µg/mL. Censored measurements contribute a label only when the bound determines the outcome. The comparison uses five training seeds (42, 1, 2, 3, 4) and five outer folds per seed.

## Cohorts

| Cohort | Measured positive | Measured negative | Presumed negative | Total |
|---|---:|---:|---:|---:|
| Measured training set | 1,048 | 133 | 0 | 1,181 |
| Augmented training set | 1,048 | 133 | 915 | 2,096 |
| Measured external panel | 159 | 37 | 0 | 196 |
| Balanced external panel | 159 | 37 | 122 | 318 |

The balanced panel contains the complete measured panel. The additional controls are 6–50-residue entries from UniProtKB/Swiss-Prot release 2026_04, matched in five-residue length bins. Their negative labels are presumptive. The HC50 endpoint comes from the assay data. No external peptide entered model fitting, selection or calibration. Measured training cases kept their original fold roles under augmentation. Within each outer training pool, the fit, selection and calibration roles received approximately 60%, 20% and 20% of cases. [Cohort audit](benchmark/cohort_audit.json)

## External performance

For each seed, the five fold models' scores were averaged per external case. AUROC and average precision (AP) were calculated from those scores and summarized across seeds using the sample standard deviation. The CSV tables also report trapezoidal PR area in a separate column.

| Augmented model | Balanced AUROC | Balanced AUPRC (AP) | Measured AUROC | Measured AUPRC (AP) |
|---|---:|---:|---:|---:|
| Ensemble | 0.921 ± 0.002 | 0.916 ± 0.003 | 0.750 ± 0.008 | 0.931 ± 0.004 |
| LightGBM | 0.919 ± 0.006 | 0.919 ± 0.006 | 0.765 ± 0.016 | 0.936 ± 0.006 |

The ensemble had the highest mean AUROC on the balanced panel. LightGBM had the highest mean AP on that panel and the highest mean AUROC and AP among augmented models on the measured panel. On the same 196 measured cases, augmentation changed LightGBM's AUROC by +0.022 (paired 95% interval −0.044 to +0.088). Interval estimates use 2,000 paired case resamples stratified by label and evidence source, conditional on the five trained seeds, without adjustment for multiple comparisons. [All methods and panels](benchmark/summary.csv) · [Paired training differences](benchmark/training_differences.csv)

Background source affects interpretation. A diagnostic rule that gives all measured peptides the same high score and all UniProt controls the same low score achieves AUROC 0.884 on the balanced panel. Positive prevalence, the AP reference level, is 0.500 on the balanced panel and 0.811 on the measured panel. The measured panel has only 37 negatives, of which 34 came from one source.

## Numerical replay

Run from the repository root using Python 3.10.

```bash
python -m pip install -r hemolysis_classification/hc50_1024/requirements.txt
python hemolysis_classification/hc50_1024/replay.py
```

The CPU replay checks input hashes and computes all 680 seed-level benchmark rows from the frozen scores. It then verifies 136 summary rows, 68 paired mean differences and both sets of ablation summary means and standard deviations. The interval endpoints are archived estimates from the original analysis.

- [External scores](scores/external.csv) contain one row per anonymous case, training arm and seed. [Outer-fold validation scores](scores/validation.csv) contain one held-out score per training case, method and seed under augmentation. Each method has its own probability column. The two files have separate case identifiers. `label_evidence` distinguishes the measured and UniProt groups.
- [Seed-level benchmark](benchmark/per_seed.csv) includes both external panels, a measured-positive versus UniProt-only comparison, and out-of-fold results. Training arm `augmented` uses the balanced training set, while `measured_training` uses the 1,181 measured cases.
- [Measured-cohort ablations](ablation/summary.csv) compare endpoints, adjustment of decision thresholds and removal of global properties. `native512` and `native1024` use their respective endpoint labels. `transfer512` and `recal512` evaluate a 512-trained model against the 1024 endpoint with the original or recalibrated decision threshold. `paired_external` contains the 191 peptides determinate at both endpoints. [Paired contrasts](ablation/paired_contrasts.csv) · [Fixed feature/classifier comparisons](ablation/s15_summary.csv)
- [Benchmark run checks](benchmark/verification.json) and [ablation run checks](ablation/verification.json) record the original run provenance. [SHA256SUMS](SHA256SUMS) covers the released numerical files.

This release supports retrospective evaluation of frozen predictions. Model weights and sequence-linked training records are outside this release.
