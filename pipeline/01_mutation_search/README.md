# Step 1 — Mutational search and E. coli AMP screening

The historical physicochemical mutant generator now screens sequences with the current **CatBoost-Tuned seed123** MIC128 classifier. The classifier uses 40 sequence descriptors and calibrated probabilities. It accepts full 6–50-residue sequences; the old 30-token plus 10-property pickle input is retired.

Install the [classification requirements](../../amp_classification/requirements.txt) in Python 3.10. From this directory:

```bash
python -m pip install -r requirements.txt
python search_mutations.py -s KSMLKSMK -k 1 -o outputs/demo --threads 4
```

| Option | Meaning | Default |
| :--- | :--- | :--- |
| `--sequence`, `-s` | Canonical peptide template, 6–50 residues | KSMLKSMPMTLK |
| `--search_length`, `-k` | Number of mutated positions | 3 |
| `--output_dir`, `-o` | Output directory | outputs |
| `--model` | Directory containing one released checkpoint and `release.json` | CatBoost-Tuned seed123 |
| `--operating-point` | Vop-derived decision threshold | recall95 |
| `--threads` | Worker processes and classifier CPU threads | 4 |

`recall95` is a target on the operating-point validation set, not a guarantee for novel mutants. CatBoost-Tuned is an example within the requested 15-method subset, not a newly validation-selected winner. Other released classifiers can be selected with `--model ../../amp_classification/checkpoints/<method>/seed123`; Cascade requires its separate Python 3.9 environment.

The output retains headerless `SEQ_positive_k.csv` and `SEQ_negative_k.csv` for Step 2 compatibility. `SEQ_MIC128_scores_k.csv` additionally records every scored sequence, calibrated probability, hard decision, threshold, model and seed. Depending on the template and physicochemical filters, a small demo may yield no candidates. Enumeration grows rapidly with mutation depth.

## Reproduce the complete classifier benchmark

```bash
python evaluate_models.py --output outputs/benchmark_replay
```

This command reproduces the frozen 15-method Test200 results; it no longer fits an unrelated generic demo split. For training, prediction, dataset construction, method definitions, both-class metrics and the two publication figures, use the [classification module](../../amp_classification/README.md). The historical classifier remains recoverable at commit `ad2102316e65a5e4467b5f19ea5136e3e2d78849`.
