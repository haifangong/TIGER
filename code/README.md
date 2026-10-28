# Current TIGER-pair implementation

Use `python -m code.main train` and `python -m code.main predict`; see the [root README](../README.md).

`train.py`, `dataloader.py`, `models.py`, `evaluation.py` and numerical helpers in `utils/` are byte-identical to the formal V17 runtime. [Runtime identities](../provenance/runtime_identity.json) record their hashes. The configuration class has defaults for several experiments; always load a released [V17 configuration](../configs/TIGER_pair/seed123.json).

`release.py` adds portable paths, explicit CPU/CUDA selection, complete endpoint checks, fold-specific normalization, five-fold averaging and saved calibration. `release_metrics.py` supplies independent family metrics. These wrappers do not alter the training core or hyperparameters.

The original MIC/activity/toxicity implementation is preserved under [legacy/runtime/code](../legacy/runtime/code/).
