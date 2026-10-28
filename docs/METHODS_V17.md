# TIGER-pair V17 methods

## Frozen inputs

The release preserves V14 source sequences, MIC labels, terminal annotations, historical terminal-feature calculation and external CFU encoding. The loader retains standard amino-acid sequences of length 6–50, positive parsed MIC values and available structures. Sequence–CFU duplicates prioritize originally uppercase records, then select the smallest MIC among those candidates. The archived loader defines the actual inputs.

The activity table has 11,516 source rows. There are 7,530 records with valid E. coli MIC. Length filtering removes 129 records and sequence–CFU deduplication removes 184, leaving 7,217 rows and 6,847 unique sequences. All retained rows construct graphs successfully. The recorded first fold has 5,773 training rows and 1,444 validation rows. Length filtering and deduplication account for the 313-row difference. These are row counts, not unique-peptide counts.

The separate negative pool has 7,860 peptides with a capped pseudo-target of log2(10,000 µg/mL). Exact conflicts are removed from negatives; activity records take precedence. Negatives enter classification pretraining and regression, but not CV splitting, validation anchors or OOF calibration. P11 enables both stages. P00 disables the entire classification-pretraining stage and regression negatives; it does not isolate independent marginal effects. P10/P01 were not retrained against the V17 reference.

The historical matcher does not recognize the short N-terminal `ACT` tag as acetylation. Missing terminal fields do not confirm experimentally free termini. These behaviours are retained because they define the actual model inputs; labels and molecular masses are not recalculated. All 208 external peptides receive the historical `1E5 - 1E6` CFU code. Medium remains metadata. The study team confirmed consistent experimental conditions within each family.

## Architecture and optimization

P, S and G denote ten global descriptors, integer sequence tokens and a 3D-derived graph. Cα contacts within 5 Å define adjacency; distances are not retained as edge weights. GATv2 uses two layers, 90-dimensional embeddings, attention pooling and dropout 0.35. The cache stores 20 amino-acid prior and 20 energy channels; the baseline enables priors and masks energies. Energy, absolute coordinates, learned node embeddings and zero node attributes are separate interventions. Sequence-query cross-attention fuses modalities, followed by a difference head. The sequence-only control changes the query role to preserve information flow; it is not representative of every sequence model class.

Exact-sequence GroupKFold separates five folds. Descriptor and node scales are fitted within each training fold, then applied to validation and negative graphs. Seeds are 120–129. Classification pretraining runs for 15 epochs; regression for up to 70, with patience 16 and minimum improvement 0.0001. Learning rate is 0.001, weight decay 0, scheduler cosine, pair batch size 2,048. Five folds run sequentially per seed without extra per-fold RNG resets.

Positive-pair candidates use a ±5-residue window and at most 2,500 candidate visits per anchor. Directed Needleman–Wunsch score divided by anchor length must reach 0.7; this is not percentage identity. Unsigned bins are `int(round(abs(delta) * 100))`, each capped at 20,000 pairs, with a global cap of 250,000. Targets remain signed. Signed controls use `floor(delta / 1.0)`, changing both bin direction and granularity. Disabling the training gate differs from threshold zero, which still excludes negative scores.

The reference uses neighbour validation: threshold 0.7, top-50 neighbours and fallback-20. Strict within-fold validation is a separate control. Validation selects fold checkpoints and supplies OOF predictions, so CV is model-selection-aware, not a nested outer test. The linear calibrator is fitted on OOF predictions and applied to external data; external labels are not used to fit it. The reference reuses the complete unsigned-20,000 ten-seed batch. The single-head control is reused because SingleDataset bypasses pair balancing. Forty other controls were retrained in 400 jobs/2,000 folds.

## Endpoint, calibration and metrics

There are 208 external peptides, 11 families and 2,211 directed within-family pairs. Identity is `(family, anchor_sequence, query_sequence)`. Labels are `log2(MIC_anchor / MIC_query)`, with mass-concentration MIC. Every fold uses its own scales; predictions are averaged across five folds, then the seed's saved calibrator is applied once. Final-pool scales or calibration applied repeatedly are not equivalent.

| Metric | Within-family definition for truth y and prediction p |
| :--- | :--- |
| PCC | Pearson correlation of y and p |
| KCC | Kendall τb |
| log10MAE | `mean(abs(p-y)) * log10(2)` |
| RSE | `sum((p-y)^2) / sum((y-mean(y))^2)` |

The 11 family values are averaged equally; this is different from a pooled correlation over all pairs. Undefined correlations propagate to family-macro and ten-seed means. No family/seed is silently dropped. SD is the sample SD across ten seeds; intervals use Student t with 9 degrees of freedom. These intervals describe seed variation on a fixed panel. Pairs sharing peptides are not independent biological replicates.

The main model was selected using this external panel, so comparisons remain exploratory. All 41 controls × four metrics share one 164-test family of two-sided paired-seed Wilcoxon tests with Holm correction. Minimum adjusted P is 0.3203125. Descriptive differences must not be presented as Holm-significant effects. See [complete results](ABLATION_RESULTS.md).

## Reproduction evidence

The numerical core, 420 configurations, exact input structures/tables, 420 external predictions and 50 selected main-model checkpoints are supplied. Independent prediction replay recomputes 21,652 reference values. Checkpoint replay compares the full calibrated prediction vector, because positive affine transforms leave PCC unchanged while changing amplitudes. The training-input audit also reproduces positive/negative pair hashes in seed123/fold1.

The paired English study methods are preserved as `methods_dataset_TIGERpair_EN.md` and `.docx` alongside this document. Those describe the broader V17 study, including external baselines in the companion repository; this document describes this model release's scope.
