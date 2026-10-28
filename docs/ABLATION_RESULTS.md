# Complete V17 ablation results

Mean ± sample SD across seeds 120–129. Each seed averages five fold predictions and applies its OOF calibrator before family-macro evaluation. There are 208 peptides, 11 families and 2,211 directed pairs. PCC/KCC: higher is better; log10MAE/RSE: lower is better.

The reference is TIGER-pair with unsigned quota 20,000 and neighbour validation. All 164 paired-seed tests are jointly Holm-corrected; no adjusted P is below 0.05. Main-model selection used this external panel, so comparisons are exploratory.

## Reference

| Configuration | PCC | KCC | log10MAE | RSE |
| :--- | ---: | ---: | ---: | ---: |
| TIGER-pair | 0.6543 ± 0.0168 | 0.4729 ± 0.0210 | 0.3813 ± 0.0077 | 0.6659 ± 0.0256 |

## Feature modalities

| Configuration | PCC | KCC | log10MAE | RSE |
| :--- | ---: | ---: | ---: | ---: |
| P + G | 0.5932 ± 0.0187 | 0.4256 ± 0.0109 | 0.4027 ± 0.0046 | 0.7460 ± 0.0167 |
| P + S | 0.5519 ± 0.0293 | 0.4027 ± 0.0213 | 0.4082 ± 0.0051 | 0.7851 ± 0.0187 |
| S + G | 0.5319 ± 0.0256 | 0.3838 ± 0.0193 | 0.4064 ± 0.0060 | 0.7688 ± 0.0220 |
| P only | 0.5320 ± 0.0145 | 0.3918 ± 0.0115 | 0.4047 ± 0.0026 | 0.7714 ± 0.0092 |
| G only | 0.4996 ± 0.0170 | 0.3599 ± 0.0119 | 0.4220 ± 0.0044 | 0.8033 ± 0.0130 |
| S only | 0.0562 ± 0.0484 | 0.0346 ± 0.0342 | 0.4914 ± 0.0030 | 1.1060 ± 0.0196 |

## Graph encoder

| Configuration | PCC | KCC | log10MAE | RSE |
| :--- | ---: | ---: | ---: | ---: |
| GAT | 0.6146 ± 0.0224 | 0.4425 ± 0.0192 | 0.3895 ± 0.0060 | 0.6998 ± 0.0214 |
| GraphSAGE | 0.6089 ± 0.0126 | 0.4375 ± 0.0100 | 0.3828 ± 0.0045 | 0.6810 ± 0.0147 |
| GCN | 0.5672 ± 0.0387 | 0.4070 ± 0.0310 | 0.4045 ± 0.0057 | 0.7414 ± 0.0252 |

## Node attributes

| Configuration | PCC | KCC | log10MAE | RSE |
| :--- | ---: | ---: | ---: | ---: |
| + Energy | 0.5501 ± 0.0325 | 0.3929 ± 0.0261 | 0.4052 ± 0.0063 | 0.7697 ± 0.0266 |
| + Coordinates | 0.4485 ± 0.0453 | 0.3164 ± 0.0351 | 0.4177 ± 0.0082 | 0.8298 ± 0.0400 |
| + Energy + coordinates | 0.5364 ± 0.0120 | 0.3841 ± 0.0077 | 0.4092 ± 0.0044 | 0.7923 ± 0.0193 |
| + Learned embedding | 0.5996 ± 0.0516 | 0.4280 ± 0.0421 | 0.3893 ± 0.0104 | 0.7092 ± 0.0425 |
| Zero node features | 0.5313 ± 0.0198 | 0.3869 ± 0.0168 | 0.4070 ± 0.0046 | 0.7812 ± 0.0155 |

## Edge geometry

| Configuration | PCC | KCC | log10MAE | RSE |
| :--- | ---: | ---: | ---: | ---: |
| Sequence-chain edges | 0.6323 ± 0.0302 | 0.4510 ± 0.0224 | 0.3869 ± 0.0083 | 0.6888 ± 0.0278 |

## Feature fusion

| Configuration | PCC | KCC | log10MAE | RSE |
| :--- | ---: | ---: | ---: | ---: |
| Cross-attention: P query | 0.6299 ± 0.0336 | 0.4503 ± 0.0308 | 0.3896 ± 0.0076 | 0.6986 ± 0.0304 |
| Cross-attention: G query | 0.5590 ± 0.0234 | 0.4005 ± 0.0165 | 0.3949 ± 0.0049 | 0.7298 ± 0.0233 |
| Feature concatenation | 0.6042 ± 0.0318 | 0.4265 ± 0.0256 | 0.4053 ± 0.0083 | 0.7572 ± 0.0291 |
| Self-attention: G/S/P | 0.5955 ± 0.0243 | 0.4288 ± 0.0164 | 0.3941 ± 0.0070 | 0.7173 ± 0.0256 |
| Self-attention: S/G/P | 0.6051 ± 0.0293 | 0.4388 ± 0.0227 | 0.3910 ± 0.0068 | 0.7071 ± 0.0280 |

## Prediction design

| Configuration | PCC | KCC | log10MAE | RSE |
| :--- | ---: | ---: | ---: | ---: |
| Concatenated pair head | 0.6009 ± 0.0410 | 0.4346 ± 0.0286 | 0.3914 ± 0.0053 | 0.7071 ± 0.0279 |
| Single-peptide regression | 0.6273 ± 0.0165 | 0.4510 ± 0.0127 | 0.4021 ± 0.0044 | 0.7522 ± 0.0162 |

## Similarity gate

| Configuration | PCC | KCC | log10MAE | RSE |
| :--- | ---: | ---: | ---: | ---: |
| No training gate | 0.4378 ± 0.0432 | 0.3004 ± 0.0309 | 0.4247 ± 0.0119 | 0.8334 ± 0.0475 |
| Similarity ≥ 0.0 | 0.4515 ± 0.0511 | 0.3113 ± 0.0348 | 0.4209 ± 0.0151 | 0.8229 ± 0.0658 |
| Similarity ≥ 0.3 | 0.4627 ± 0.0493 | 0.3294 ± 0.0390 | 0.4105 ± 0.0127 | 0.7851 ± 0.0440 |
| Similarity ≥ 0.5 | 0.5964 ± 0.0207 | 0.4267 ± 0.0159 | 0.3929 ± 0.0026 | 0.7064 ± 0.0145 |
| Similarity ≥ 0.9 | 0.6188 ± 0.0193 | 0.4482 ± 0.0141 | 0.3944 ± 0.0041 | 0.7107 ± 0.0158 |

## Signed pair quota

| Configuration | PCC | KCC | log10MAE | RSE |
| :--- | ---: | ---: | ---: | ---: |
| Signed quota 1000 | 0.6177 ± 0.0334 | 0.4515 ± 0.0261 | 0.3985 ± 0.0081 | 0.7332 ± 0.0310 |
| Signed quota 2000 | 0.6439 ± 0.0230 | 0.4672 ± 0.0209 | 0.3799 ± 0.0069 | 0.6678 ± 0.0281 |
| Signed quota 4000 | 0.6413 ± 0.0156 | 0.4636 ± 0.0145 | 0.3809 ± 0.0029 | 0.6682 ± 0.0087 |
| Signed quota 8000 | 0.6412 ± 0.0267 | 0.4647 ± 0.0258 | 0.3844 ± 0.0039 | 0.6799 ± 0.0169 |
| Signed quota 10000 | 0.6364 ± 0.0270 | 0.4633 ± 0.0203 | 0.3834 ± 0.0062 | 0.6765 ± 0.0246 |
| Signed quota 20000 | 0.6345 ± 0.0319 | 0.4634 ± 0.0242 | 0.3814 ± 0.0054 | 0.6710 ± 0.0247 |

## Unsigned pair quota

| Configuration | PCC | KCC | log10MAE | RSE |
| :--- | ---: | ---: | ---: | ---: |
| Unsigned quota 1000 | 0.6448 ± 0.0350 | 0.4653 ± 0.0304 | 0.3851 ± 0.0085 | 0.6832 ± 0.0312 |
| Unsigned quota 2000 | 0.6234 ± 0.0449 | 0.4511 ± 0.0311 | 0.3879 ± 0.0085 | 0.6951 ± 0.0364 |
| Unsigned quota 4000 | 0.6356 ± 0.0264 | 0.4600 ± 0.0222 | 0.3835 ± 0.0077 | 0.6763 ± 0.0319 |
| Unsigned quota 8000 | 0.6391 ± 0.0197 | 0.4581 ± 0.0212 | 0.3853 ± 0.0041 | 0.6801 ± 0.0153 |
| Unsigned quota 10000 | 0.6528 ± 0.0147 | 0.4738 ± 0.0116 | 0.3803 ± 0.0043 | 0.6630 ± 0.0164 |

## Training / validation pairs

| Configuration | PCC | KCC | log10MAE | RSE |
| :--- | ---: | ---: | ---: | ---: |
| Negative-pair strategy: top-k | 0.6195 ± 0.0217 | 0.4547 ± 0.0210 | 0.3904 ± 0.0053 | 0.7007 ± 0.0214 |
| Strict within-fold validation | 0.6469 ± 0.0207 | 0.4675 ± 0.0218 | 0.3841 ± 0.0062 | 0.6789 ± 0.0239 |

## Negative-sample exposure

| Configuration | PCC | KCC | log10MAE | RSE |
| :--- | ---: | ---: | ---: | ---: |
| P00: neither stage | 0.5484 ± 0.0338 | 0.3990 ± 0.0292 | 0.4094 ± 0.0075 | 0.7784 ± 0.0277 |

Signed versus unsigned controls also change bin granularity. Sequence-only performance describes this particular branch/query configuration. P00 removes complete classification pretraining as well as regression negatives. Sequence-chain edges are an edge-geometry intervention; zero node features are a node-attribute intervention.

Unrounded results and all tests: [`results/v17/reference/`](../results/v17/reference/). See [methods](METHODS_V17.md) for full interpretation.
