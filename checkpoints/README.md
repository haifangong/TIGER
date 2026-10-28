# Released checkpoints

The current TIGER-pair main model is under [`V17/TIGER_pair/`](V17/TIGER_pair/): ten seeds (120–129), five selected folds per seed, fold-specific global/node scales, OOF linear calibrators, and split/pair audits. Run it with `python -m code.main predict --seed 123 --check-reference` from the repository root.

The old MIC ablation checkpoints and CV tables have been replaced; their original contents remain available at [the previous revision](https://github.com/haifangong/TIGER/tree/b83b41c17a757722c4a49648359fdbc6c0f68ef6/checkpoints/ablation). Current external results are in [`results/v17/`](../results/v17/).

The historical [`ablation_toxin/`](ablation_toxin/) and [`wetlab/`](wetlab/) models remain unchanged and use the preserved [`legacy/runtime/code/`](../legacy/runtime/code/) implementation. Their results are separate from the current V17 external endpoint.
