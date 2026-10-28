# TIGER data and unsigned-20,000 ablation methods (V17)

## Data and actual input freeze

V17 preserves the V14 source sequences, MIC labels, terminal annotations, historical terminal-feature implementation and the external uniform CFU encoding. Missing terminal annotations are not taken as experimental confirmation of free termini. The historical string matcher does not recognize the short N-terminal ACT tag as acetylation; this known encoding limitation is retained explicitly under the actual-input freeze. ACT-to-acetyl alias correction, molecular-mass relabelling and external CFU recoding were not applied. The approved positive-precedence rule for the external negative pool is retained independently: 7,962 source negatives minus 13 positive conflicts gives 7,949 sequences. The TIGER negative pool already has no exact positive overlap.

The external input dictionary contains 134 peptides with both terminal fields missing and 74 with recorded free N-termini and amidated C-termini. All 208 receive the historical 1E5–1E6 CFU input code. Measured medium is metadata, not an input feature. The study team confirmed that within-family experimental conditions are consistent; apparent record discrepancies are treated as curation noise and no extra condition-based edge removal is performed.

## Model inputs, pairing and registered interventions

The graph cache has 40 stored node channels. The historical TIGER-pair baseline enables the 20 amino-acid prior channels and masks the 20 Rosetta energy channels; energy and Cα-coordinate additions belong to separate ablations. Integer sequence tokens, ten global descriptors and CFU conditioning are retained. Fold-specific global and node z-scores and the saved out-of-fold linear calibrator are replayed at inference.

Sequence-grouped five-fold cross-validation retains the frozen V14 actual inputs and split indices. Positive-pair candidates use a ±5-residue window, at most 2,500 candidates per anchor, directed alignment score divided by anchor length and threshold 0.7. The main model balances unsigned bins int(round(abs(Δlog2 MIC) × 100)) with quota 20,000 and global cap 250,000; the supervised target remains signed. This is a per-bin quota, not the total training-pair count. Neighbour validation retains top-50/fallback-20 pairing and threshold 0.7. Strict validation is a separate ablation and retains the exact frozen row-level builder, including repeated sequences and ordered-pair multiplicity.

The full existing V16 unsigned-20,000 batch (seeds 120–129) supplies the V17 reference. Forty registered controls are trained anew (400 seed jobs, 2,000 folds); the single-head control is reused because pair balancing is bypassed in SingleDataset. P00 disables classification pretraining completely and regression negatives; P10/P01 are outside V17 scope. Training uses 15 classification-pretraining epochs when enabled, up to 70 regression epochs, patience 16, and five sequential folds without extra RNG resets. Checkpoint selection and out-of-fold linear calibration are retained. The main model keeps neighbour validation; unsigned-20,000 plus strict validation is not silently substituted. The main model was selected on this external panel, so subsequent comparisons remain exploratory.

AMPLiT uses one formal regression implementation. Word2Vec is fitted only on training-split sequences and stored as a residue-vector table. External queries only transform through that fixed table. Descriptor columns are aligned by saved names, and short-sequence lag handling follows the saved schema. Other external predictors retain their method-specific optimization, target transformations and filters; the benchmark does not assume identical effective training pools or stopping rules.

## Fixed endpoint and statistical analysis

The canonical endpoint is exactly 208 peptides, 11 families and 2,211 unique directed edges. Every source prediction is aligned by family, anchor sequence and query sequence, checked for completeness and finite values, and independently recomputed. Absolute predictions are expressed in log10 µg/ml before differencing and division by log10(2); native EvoGradient µM predictions receive the peptide-specific molecular-mass conversion exactly once. Each family contributes equally to macro PCC, Kendall τb, log10MAE and SSE/SST RSE.

Undefined family correlations propagate to macro and ten-seed means; no family or seed is dropped. Intervals use Student t with df=9 and sample SD. The 41 ablation controls × four metrics form one 164-test two-sided paired-seed Wilcoxon family with Holm correction. The P00 contrast belongs to that family and is not reclassified as an uncorrected new test. Benchmark condition tests use 23 contrasts per metric; ensemble tests use eight comparisons jointly corrected. All-zero differences have p=1 and undefined tests retain their multiplicity slot. These intervals quantify seed variation on a fixed test panel.

The 42-row landscape has 420 seed entries. Ten reused S00 entries and ten S11 aliases of the single-head control give 440 TIGER result slots, representing 430 distinct seed predictors. Together with 420 unchanged external prediction sets, 860 source files are independently replayed. The benchmark contains 460 method–condition–seed rows. All non-pair benchmark values are checked against V16. Archived discussion is retained verbatim in notebook history, not presented as current V17 results.

| Arm | Intervention relative to the unsigned-20,000 reference |
| --- | --- |
| s4_mod_pg | {"feature_modalities": "pg"} |
| s4_mod_ps | {"feature_modalities": "ps"} |
| s4_mod_sg | {"feature_modalities": "sg"} |
| s4_mod_p | {"feature_modalities": "p"} |
| s4_mod_g | {"feature_modalities": "g"} |
| s4_mod_s | {"fusion_attn_mode": "cross_qg", "feature_modalities": "s"} |
| S4_gat | {"gnn_type": "gat"} |
| S4_sage | {"gnn_type": "graphsage"} |
| S4_gcn | {"gnn_type": "gcn"} |
| S4_se | {"structure_features": "se"} |
| S4_sp | {"structure_features": "sp", "include_node_coords": true} |
| S4_sep | {"structure_features": "sep", "include_node_coords": true} |
| S4_nodeemb | {"node_features_mode": "prior+emb"} |
| S4_nodezero | {"structure_features": "z"} |
| S4_edge_chain | {"edge_mode": "chain"} |
| s4_fus_cross_qp | {"fusion_attn_mode": "cross_qp"} |
| s4_fus_cross_qg | {"fusion_attn_mode": "cross_qg"} |
| s4_fus_concat | {"fusion": "concat"} |
| s4_fus_self_gsp | {"fusion_attn_mode": "self_gsp"} |
| s4_fus_self_sgp | {"fusion_attn_mode": "self_sgp"} |
| S4_concat | {"pair_interaction": "concat"} |
| s4_single_n1n2 | {"model_kind": "single"} |
| s4_gate_off | {"train_similarity_gate_enabled": false} |
| S4_sim00 | {"similarity_threshold": 0.0} |
| S4_sim03 | {"similarity_threshold": 0.3} |
| S4_sim05 | {"similarity_threshold": 0.5} |
| S4_sim09 | {"similarity_threshold": 0.9} |
| s4_bal_signed_bal1000 | {"pair_balance_num": 1000, "use_signed_sampling": true} |
| s4_bal_signed_bal2000 | {"pair_balance_num": 2000, "use_signed_sampling": true} |
| s4_bal_signed_bal4000 | {"pair_balance_num": 4000, "use_signed_sampling": true} |
| s4_bal_signed_bal8000 | {"pair_balance_num": 8000, "use_signed_sampling": true} |
| s4_bal_signed_bal10000 | {"pair_balance_num": 10000, "use_signed_sampling": true} |
| s4_bal_signed_bal20000 | {"pair_balance_num": 20000, "use_signed_sampling": true} |
| s4_bal_unsigned_bal1000 | {"pair_balance_num": 1000, "use_signed_sampling": false} |
| s4_bal_unsigned_bal2000 | {"pair_balance_num": 2000, "use_signed_sampling": false} |
| s4_bal_unsigned_bal4000 | {"use_signed_sampling": false, "pair_balance_num": 4000} |
| s4_bal_unsigned_bal8000 | {"pair_balance_num": 8000, "use_signed_sampling": false} |
| s4_bal_unsigned_bal10000 | {"pair_balance_num": 10000, "use_signed_sampling": false} |
| S4_negpairs | {"neg_pair_mode": "topk"} |
| s4_valpair_strict | {"val_pair_mode": "within_fold_strict"} |
| factor_P00 | {"cls_pretrain_epochs": 0, "neg_train_inject": false} |
