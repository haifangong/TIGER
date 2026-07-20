"""Seed helpers and small misc utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(root: str | None, path: str) -> str:
    """Resolve relative path against an optional project root."""
    from pathlib import Path

    p = Path(path)
    if p.is_absolute() or root is None:
        return str(p)
    return str(Path(root) / p)
