"""Neighbor-based pair-delta evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dataloader import pair_collate, precompute_neighbors
from .utils.config import Config
from .utils.constants import CFU_COL
from .utils.metrics import apply_calibrator, regression_metrics


@torch.no_grad()
def eval_pair_delta(
    model,
    anchor_graphs,
    anchor_rows: pd.DataFrame,
    anchor_labels: np.ndarray,
    query_graphs,
    query_rows: pd.DataFrame,
    query_labels: np.ndarray,
    cfg: Config,
    device: torch.device,
    neighbors: list[list[tuple[int, float]]] | None = None,
    teacher_anchor: np.ndarray | None = None,
    teacher_query: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    if neighbors is None:
        neighbors = precompute_neighbors(anchor_rows, query_rows, cfg)
    rows = []
    for query_idx, sims in enumerate(neighbors):
        if cfg.use_cfu_protocol and CFU_COL in query_rows.columns:
            if query_rows.iloc[query_idx][CFU_COL] != cfg.primary_cfu_group:
                continue
        if not sims:
            continue
        items = []
        for anchor_idx, sim in sims:
            if cfg.use_cfu_protocol and CFU_COL in anchor_rows.columns:
                if anchor_rows.iloc[anchor_idx][CFU_COL] != cfg.primary_cfu_group:
                    continue
            teacher = np.nan
            if teacher_anchor is not None and teacher_query is not None:
                teacher = float(teacher_anchor[anchor_idx] - teacher_query[query_idx])
            items.append(
                (
                    anchor_graphs[anchor_idx],
                    query_graphs[query_idx],
                    torch.tensor(float(anchor_labels[anchor_idx] - query_labels[query_idx]), dtype=torch.float32),
                    torch.tensor(teacher, dtype=torch.float32),
                )
            )
        if not items:
            continue
        loader = torch.utils.data.DataLoader(items, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=pair_collate)
        cursor = 0
        kept_sims = [(ai, s) for ai, s in sims if (not cfg.use_cfu_protocol) or anchor_rows.iloc[ai][CFU_COL] == cfg.primary_cfu_group]
        for a_batch, b_batch, y_batch, teacher_batch in loader:
            a_batch, b_batch = a_batch.to(device), b_batch.to(device)
            pred = model(a_batch, b_batch).detach().cpu().numpy().ravel()
            for value in pred:
                anchor_idx, sim = kept_sims[cursor]
                rows.append(
                    {
                        "anchor_index": int(anchor_idx),
                        "query_index": int(query_idx),
                        "anchor_sequence": anchor_rows.iloc[anchor_idx]["sequence"],
                        "query_sequence": query_rows.iloc[query_idx]["sequence"],
                        "anchor_cfu_group": anchor_rows.iloc[anchor_idx].get(CFU_COL, "unknown"),
                        "query_cfu_group": query_rows.iloc[query_idx].get(CFU_COL, "unknown"),
                        "similarity": float(sim),
                        "y_true_delta_log2_anchor_minus_query": float(anchor_labels[anchor_idx] - query_labels[query_idx]),
                        "y_pred_delta_log2_anchor_minus_query": float(value),
                    }
                )
                cursor += 1
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df, {"n": 0}
    return df, regression_metrics(df["y_true_delta_log2_anchor_minus_query"], df["y_pred_delta_log2_anchor_minus_query"])


def save_eval_outputs(out_dir: Path, cv: pd.DataFrame, fold_metrics: list[dict], calibrator: dict, summary: dict) -> None:
    out_dir = Path(out_dir)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    cv.to_csv(out_dir / "results" / "cv_pair_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(out_dir / "results" / "fold_metrics.csv", index=False)
    import json

    (out_dir / "results" / "calibrator.json").write_text(json.dumps(calibrator, indent=2))
    (out_dir / "results" / "summary.json").write_text(json.dumps(summary, indent=2))


def load_checkpoint_model(ckpt_path: Path, cfg: Config, device: torch.device):
    from .models import build_model

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
