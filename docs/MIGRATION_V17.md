# Migration to V17

The previous revision was `b83b41c17a757722c4a49648359fdbc6c0f68ef6`. This update is a normal descendant commit; its history is preserved.

| Historical convention | Current convention |
| :--- | :--- |
| `gsh` / `global_f` core | Frozen V17 `gps` / `p_f` core |
| `code/configs/default_fusion_attention.json` | `configs/TIGER_pair/seed123.json` and seeds 120–129 |
| MIC weights in `checkpoints/ablation/` | Main-model weights in `checkpoints/V17/TIGER_pair/` |
| Historical sequence-encoding ablation tables | Complete 42-configuration V17 tables in `results/v17/` |
| Single-checkpoint legacy inference helper | Five fold-specific scales/checkpoints, then one OOF calibrator |
| Historical LL37/APEX-GO packs | Canonical 208-peptide/11-family/2,211-pair panel in `data/v17/` |

Old code and documentation are retained in `legacy/`. Historical top-level metadata, discovery data, toxicity checkpoints and multi-species wet-lab checkpoints retain their previous contents. They are not inputs to the V17 default configuration. Pipeline Step 4 explicitly imports `legacy/runtime/code/` for its old model interface; candidate generation, toxicity filtering and structure prediction have not been revalidated as a new end-to-end design experiment.

The reference is unsigned quota 20,000 with neighbour validation. The quota-20,000 plus strict-validation experiment and rejected optimized-dataflow variants are not substituted. Numerical source files remain unchanged. Additions provide relative paths, CLI checks, independent replay and documentation; configuration provenance lists every relocated field.

Nine ordinary ZIP archives, each below 40 MiB, supply exact structure bytes with per-member SHA256. The 50 newly included weights are `fold*_best.pt`, not last-epoch checkpoints. Historical absolute source paths inside original checkpoint/summary metadata are provenance; the new entrypoint uses relocated relative-path configurations.
