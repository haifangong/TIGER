# checkpoints

Released MIC / toxin / wetlab weights. Only **final selected** checkpoints are
shipped (`fold*_best.pt` for MIC; `*.joblib` for toxin). `fold*_last.pt` and
training dumps are omitted.

| Path | Contents |
|------|----------|
| [`ablation/`](ablation/) | MIC paper ablations (panels 01–06) |
| [`ablation_toxin/`](ablation_toxin/) | HC50 toxicity classifiers (`both` / `global` / `sequence`) |
| [`wetlab/`](wetlab/) | Species×similarity production MIC models |

Aggregate MIC CV table: [`ablation/leaderboard.csv`](ablation/leaderboard.csv)
(also mirrored as [`leaderboard.csv`](leaderboard.csv)).

**Not included:** similarity `sim0p5_bal10000`, and the invalid `cross_qs`
sequence-only modality run (`mod_s`).
