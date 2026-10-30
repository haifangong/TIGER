# Third-party methods and attribution

The released benchmark uses scikit-learn (RF, ExtraTrees, histogram/standard gradient boosting, AdaBoost, SVMs, logistic regression, Dummy), CatBoost, XGBoost, LightGBM, InterpretML (EBM), PyTorch (the study's TinyCNN), and Deep Forest (Cascade). Their implementations retain their respective upstream licences; these dependencies are installed separately.

`third_party/AMPActiPred/predict_lib.py` and `data/PAAC.txt` retain the descriptor implementation used in the actual Cascade experiment, adapted from [Lantian Yao's AMPActiPred](https://github.com/lantianyao/AMPActiPred). The upstream README is retained alongside the source. The local library adaptation removes the command-line execution flow and resolves the PAAC descriptor table relative to the module file. No pretrained AMPActiPred model is used: the Cascade classifier was trained on the released MIC128 training partition. Source hashes and adaptation diffs are recorded in `provenance/source_adaptations.json`.

Method names describe the implementations used in this benchmark. TinyCNN is the study's sequence CNN baseline, and Cascade is its Deep Forest adaptation using AMP553 descriptors. EBM uses InterpretML's internal early stopping; `max_rounds=5000` is a configured ceiling, not a claim that every ensemble bag completes 5,000 rounds.
