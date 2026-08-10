# data/wetlab — full-DBAASP per-species tables for production training

Per-species CFU-aware MIC tables used to train the wet-lab / prospective models
under [`checkpoints/wetlab/`](../../checkpoints/wetlab/) (**no LL37 holdout**).

Built locally from the full DBAASP dump
(`data/dbaasp_amp_training_by_cfu_group_ug_per_mL _123.csv`, **not redistributed**);
only the six remapped species tables below are published.

Each `train_<Species>.csv` has:

- `sequence`, termini, `cfu_group` (from that species’ CFU bins)
- `MIC_Escherichia_coli` — **remapped** species MIC (TIGER’s default target column)
- `source_mic_col`, `species` — provenance

Used by `code/scripts_train/run_wetlab_species_sim.sh` → `checkpoints/wetlab/`
(6 species × sim `{0.3, 0.7}` = 12 runs; five-fold ensembles for ranking).
