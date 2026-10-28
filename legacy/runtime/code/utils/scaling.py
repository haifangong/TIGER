"""Per-fold / pool z-score for global physicochemical and node features."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def cache_raw_global_f(graphs: list[Any]) -> None:
    for g in graphs:
        if not hasattr(g, "_global_f_raw"):
            g._global_f_raw = g.global_f.detach().cpu().clone()


def fit_global_f_stats(graphs: list[Any], indices: np.ndarray | list[int] | None = None) -> dict[str, list[float]]:
    subset = graphs if indices is None else [graphs[int(i)] for i in indices]
    cache_raw_global_f(subset)
    matrix = np.stack([g._global_f_raw.numpy().reshape(-1) for g in subset]).astype(np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_global_f_stats(
    graphs: list[Any],
    stats: dict[str, list[float]],
    indices: np.ndarray | list[int] | None = None,
) -> None:
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    targets = graphs if indices is None else [graphs[int(i)] for i in indices]
    cache_raw_global_f(targets)
    for g in targets:
        x = g._global_f_raw.numpy().reshape(-1)
        scaled = ((x - mean) / std).astype(np.float32)
        g.global_f = torch.from_numpy(scaled).view(1, -1)


def cache_raw_node_x(graphs: list[Any]) -> None:
    for g in graphs:
        if not hasattr(g, "_x_s_raw"):
            g._x_s_raw = g.x_s.detach().cpu().clone()


def fit_node_x_stats(graphs: list[Any], indices: np.ndarray | list[int] | None = None) -> dict[str, list[float]]:
    """Fit per-dimension mean/std over all residue nodes in the selected graphs."""
    subset = graphs if indices is None else [graphs[int(i)] for i in indices]
    cache_raw_node_x(subset)
    matrix = np.concatenate([g._x_s_raw.numpy() for g in subset], axis=0).astype(np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist(), "n_nodes": int(matrix.shape[0])}


def apply_node_x_stats(
    graphs: list[Any],
    stats: dict[str, list[float]],
    indices: np.ndarray | list[int] | None = None,
) -> None:
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    targets = graphs if indices is None else [graphs[int(i)] for i in indices]
    cache_raw_node_x(targets)
    for g in targets:
        x = g._x_s_raw.numpy()
        scaled = ((x - mean) / std).astype(np.float32)
        g.x_s = torch.from_numpy(scaled)
