# data/wetlab — full-DBAASP per-species tables for production training

Built by `scripts/prepare_wetlab_dbassp_tables.py` from
`data/dbaasp_amp_training_by_cfu_group_ug_per_mL _123.csv`
(**no LL37 holdout**).

Each `train_<Species>.csv` has:

- `sequence`, termini, `cfu_group` (from that species’ CFU bins)
- `MIC_Escherichia_coli` — **remapped** species MIC (TIGER’s default target column)
- `source_mic_col`, `species` — provenance

Used by `code/scripts_train/run_wetlab_species_sim.sh` → `checkpoints/wetlab/`.
