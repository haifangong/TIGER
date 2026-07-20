"""TIGER modular core package.

Layout
------
code/
  main.py           CLI entry (train / evaluate / infer)
  train.py          GroupKFold training + final retrain
  evaluation.py     Neighbor pair-delta evaluation + metrics I/O
  infer.py          Load checkpoint and score pairs / peptides
  dataloader.py     CSV preprocess, graph build, pair sampling, datasets
  models.py         GNN encoder + pair / single heads
  utils/            config, constants, features, metrics, scaling, seed
  configs/          JSON hyperparameter files

The legacy monolithic pipelines under ``src/poap_gpt/`` are retained unchanged.
"""

from .utils.config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "0.1.0"
