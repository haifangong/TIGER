"""Inference: load a checkpoint and score peptide pairs or absolute MIC scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataloader import load_or_build_cache, precompute_neighbors
from .evaluation import eval_pair_delta, load_checkpoint_model
from .utils.config import Config
from .utils.metrics import apply_calibrator, regression_metrics
from .utils.scaling import (
    apply_global_f_stats,
    apply_node_x_stats,
    cache_raw_global_f,
    cache_raw_node_x,
)


def infer(cfg: Config, root: Path | None = None) -> pd.DataFrame:
    """Score LL37/external queries against the train/val anchor pool using ``cfg.checkpoint``."""
    if not cfg.checkpoint:
        raise ValueError("infer requires cfg.checkpoint pointing to a .pt file")
    out_dir = Path(cfg.out_dir)
    if root is not None and not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Force external graphs for inference
    cfg.eval_external = True
    payload = load_or_build_cache(cfg, root)
    graphs, rows = payload["graphs"], payload["rows"]
    test_graphs, test_rows = payload["test_graphs"], payload["test_rows"]
    labels = rows["label"].to_numpy(float)
    test_labels = test_rows["label"].to_numpy(float)

    device = torch.device(cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu")
    model, ckpt = load_checkpoint_model(Path(cfg.checkpoint), cfg, device)

    # Apply final-pool z-score stats if present next to the checkpoint / out_dir
    if str(cfg.global_feature_scaling).lower() == "zscore":
        cache_raw_global_f(graphs)
        cache_raw_global_f(test_graphs)
        stats_candidates = [
            Path(cfg.checkpoint).parent.parent / "intermediate" / "global_feature_zscore_final.json",
            out_dir / "intermediate" / "global_feature_zscore_final.json",
        ]
        for sp in stats_candidates:
            if sp.exists():
                apply_global_f_stats(graphs, json.loads(sp.read_text()))
                apply_global_f_stats(test_graphs, json.loads(sp.read_text()))
                break
    if str(getattr(cfg, "node_feature_scaling", "none")).lower() == "zscore":
        cache_raw_node_x(graphs)
        cache_raw_node_x(test_graphs)
        node_candidates = [
            Path(cfg.checkpoint).parent.parent / "intermediate" / "node_feature_zscore_final.json",
            out_dir / "intermediate" / "node_feature_zscore_final.json",
        ]
        for sp in node_candidates:
            if sp.exists():
                apply_node_x_stats(graphs, json.loads(sp.read_text()))
                apply_node_x_stats(test_graphs, json.loads(sp.read_text()))
                break

    neighbors = precompute_neighbors(rows, test_rows, cfg)
    pred, metrics = eval_pair_delta(
        model, graphs, rows, labels, test_graphs, test_rows, test_labels, cfg, device, neighbors=neighbors
    )

    cal_path = Path(cfg.checkpoint).parent.parent / "results" / "calibrator.json"
    if cal_path.exists() and len(pred):
        calibrator = json.loads(cal_path.read_text())
        pred = apply_calibrator(pred, calibrator)
        metrics = regression_metrics(pred.y_true_delta_log2_anchor_minus_query, pred.y_pred_delta_log2_anchor_minus_query)

    pred_path = out_dir / "infer_pair_predictions.csv"
    pred.to_csv(pred_path, index=False)
    (out_dir / "infer_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"metrics": metrics, "n_pairs": int(len(pred)), "wrote": str(pred_path)}, indent=2))
    return pred


def score_peptides(cfg: Config, sequences: list[str] | None = None) -> pd.DataFrame:
    """Optional helper: absolute score head for single-model checkpoints."""
    if cfg.model_kind != "single":
        raise ValueError("score_peptides requires model_kind=single")
    raise NotImplementedError("Absolute peptide scoring requires graphs for each sequence; use infer() for pair deltas.")
