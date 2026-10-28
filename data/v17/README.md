# Frozen V17 model inputs

`training.csv` and `negative.csv` preserve the exact source bytes used by the formal model. `features.txt` supplies amino-acid node descriptors. The legacy LL37 hook is retained for configuration completeness; formal configs set `eval_external=false` during training and evaluate the canonical panel afterwards.

`external_peptides.csv` contains 208 peptides with source metadata; `canonical_pairs.csv` specifies the 2,211 directed pairs and log2 targets. `external_pdb/` contains their exact structures. The published endpoint uses uniform historical CFU input encoding; see the [model methods](../../docs/METHODS_V17.md).

`structures/` contains nine ordinary ZIP files, covering 10,252 eligible activity-source structures and 7,860 negative structures. The activity archive includes structures for source rows later excluded by MIC/filtering/graph checks; archive counts are not final training counts. Run `python scripts/prepare_structures.py` from the repository root to extract into ignored `data/v17/pdb/`. Every member is verified against `provenance/structure_archive_members.csv`.
