# Historical workflow compatibility

The pre-V17 implementation is preserved in `runtime/code/`, with its original scripts and README. The [original root README](README_original.md) and its relative links can be viewed in [the previous revision](https://github.com/haifangong/TIGER/tree/b83b41c17a757722c4a49648359fdbc6c0f68ef6).

Toxicity and multi-species wet-lab weights remain at their original locations. `pipeline/04_metric_ranking/rank_candidates.py` explicitly imports this matching runtime. V17 uses the root-level `code/` and `data/v17/`. Old MIC ablation weights/results remain in Git history.

For the historical discovery workflow, install Git LFS and retrieve its structure archive from the repository root:

```bash
git lfs install
git lfs pull --include=data/trainval_dbassp/pdb/trainval_and_excluded_pdbs.zip
mkdir -p data/3D_data_train_eva_Rosetta
unzip -q data/trainval_dbassp/pdb/trainval_and_excluded_pdbs.zip -d data/3D_data_train_eva_Rosetta
```

V17 instead uses `python scripts/prepare_structures.py` and requires no LFS files.
