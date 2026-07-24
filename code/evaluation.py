"""Neighbor-based and APEXGO family pair-delta evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .dataloader import (
    build_graphs,
    pair_collate,
    precompute_neighbors,
    preprocess_dataset,
    sequence_similarity,
    setup_aligner,
)
from .utils.config import Config
from .utils.constants import CFU_COL, TARGET_COL
from .utils.metrics import LOG10_OF_2, apply_calibrator, regression_metrics

# Top APEXGO templates by within-family log10(MIC) span (E. coli ATCC11775).
APEXGO_HIGH_SPAN_FAMILIES: tuple[str, ...] = (
    "Mylodonin-2",  # ~35.5× MIC ratio, log10_span ≈ 1.55
    "Mylodonin-3",  # ~33.6×
    "Equusin-4",  # ~16.2×
    "Mammuthusin-3",  # ~14.9×
    "Hesperelin-3",  # ~14.9×
)


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
    bidirectional: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate pair deltas. If ``bidirectional``, use
    ``0.5 * (f(a,q) - f(q,a))`` to enforce antisymmetry at test time.
    """
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
            pred_fwd = model(a_batch, b_batch).detach().cpu().numpy().ravel()
            if bidirectional:
                pred_rev = model(b_batch, a_batch).detach().cpu().numpy().ravel()
                pred = 0.5 * (pred_fwd - pred_rev)
            else:
                pred = pred_fwd
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

    (out_dir / "results" / "calibrator.json").write_text(json.dumps(calibrator, indent=2))
    (out_dir / "results" / "summary.json").write_text(json.dumps(summary, indent=2))


def load_checkpoint_model(ckpt_path: Path, cfg: Config, device: torch.device):
    from .models import build_model

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _resolve_path(path: str | Path, root: Path | None) -> Path:
    p = Path(path)
    if p.is_absolute() or root is None:
        return p
    return root / p


def _metrics_payload(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {"n": 0}
    return regression_metrics(
        df["y_true_delta_log2_anchor_minus_query"],
        df["y_pred_delta_log2_anchor_minus_query"],
    )


@torch.no_grad()
def eval_explicit_pairs(
    model,
    graphs,
    rows: pd.DataFrame,
    pair_table: pd.DataFrame,
    cfg: Config,
    device: torch.device,
    *,
    bidirectional: bool = False,
    family_col: str = "family",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score an explicit directed pair table (e.g. within-family APEXGO pairs).

    Model predictions are log2-MIC deltas. Metrics include ``log10MAE``, ``RSE``,
    ``PCC``, and ``KCC`` (RSE/PCC/KCC are identical in log2 and log10 space).
    """
    model.eval()
    seq_to_idx = {seq: i for i, seq in enumerate(rows["sequence"].tolist())}
    aligner = setup_aligner()
    items = []
    meta: list[Any] = []
    missing = 0
    for rec in pair_table.itertuples(index=False):
        ai = seq_to_idx.get(rec.anchor_sequence)
        qi = seq_to_idx.get(rec.query_sequence)
        if ai is None or qi is None:
            missing += 1
            continue
        y = float(rows.iloc[ai]["label"] - rows.iloc[qi]["label"])
        items.append(
            (
                graphs[ai],
                graphs[qi],
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(float("nan"), dtype=torch.float32),
            )
        )
        meta.append(rec)

    preds: list[float] = []
    loader = torch.utils.data.DataLoader(
        items,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=pair_collate,
    )
    for a_batch, b_batch, _y_batch, _teacher_batch in loader:
        a_batch, b_batch = a_batch.to(device), b_batch.to(device)
        pred_fwd = model(a_batch, b_batch).detach().cpu().numpy().ravel()
        if bidirectional:
            pred_rev = model(b_batch, a_batch).detach().cpu().numpy().ravel()
            pred = 0.5 * (pred_fwd - pred_rev)
        else:
            pred = pred_fwd
        preds.extend(float(v) for v in pred)

    out_rows = []
    for rec, pred in zip(meta, preds):
        ai = seq_to_idx[rec.anchor_sequence]
        qi = seq_to_idx[rec.query_sequence]
        y_true = float(rows.iloc[ai]["label"] - rows.iloc[qi]["label"])
        family = getattr(rec, family_col, None)
        if family is None and "template" in pair_table.columns:
            family = getattr(rec, "template", None)
        out_rows.append(
            {
                "family": family,
                "anchor_id": getattr(rec, "anchor_id", None),
                "query_id": getattr(rec, "query_id", None),
                "anchor_sequence": rec.anchor_sequence,
                "query_sequence": rec.query_sequence,
                "anchor_cfu_group": rows.iloc[ai].get(CFU_COL, "unknown"),
                "query_cfu_group": rows.iloc[qi].get(CFU_COL, "unknown"),
                "similarity": float(sequence_similarity(aligner, rec.anchor_sequence, rec.query_sequence)),
                "anchor_mic_ug_mL": float(getattr(rec, "anchor_mic_ug_mL", rows.iloc[ai][TARGET_COL])),
                "query_mic_ug_mL": float(getattr(rec, "query_mic_ug_mL", rows.iloc[qi][TARGET_COL])),
                "y_true_delta_log2_anchor_minus_query": y_true,
                "y_pred_delta_log2_anchor_minus_query": float(pred),
                "y_true_delta_log10_anchor_minus_query": float(y_true * LOG10_OF_2),
                "y_pred_delta_log10_anchor_minus_query": float(pred * LOG10_OF_2),
                "query_is_more_active": bool(getattr(rec, "query_is_more_active", y_true > 0)),
                "activity_tie": bool(getattr(rec, "activity_tie", abs(y_true) < 1e-12)),
            }
        )

    df = pd.DataFrame(out_rows)
    metrics: dict[str, Any] = _metrics_payload(df)
    if missing:
        metrics["n_pairs_missing_graphs"] = int(missing)
    if len(df) and "family" in df.columns and df["family"].notna().any():
        per_family = {}
        for fam, group in df.groupby("family", dropna=True):
            per_family[str(fam)] = _metrics_payload(group)
        metrics["per_family"] = per_family
    return df, metrics


def evaluate_apexgo_high_span(
    cfg: Config,
    *,
    checkpoint: str | Path | None = None,
    root: Path | None = None,
    families: Sequence[str] | None = None,
    peptide_csv: str | Path | None = None,
    pair_csv: str | Path | None = None,
    pdb_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    bidirectional: bool = False,
    apply_saved_calibrator: bool = True,
) -> dict[str, Any]:
    """Evaluate a TIGER checkpoint on large-span APEXGO families.

    Default families (largest within-family MIC gaps among APEXGO templates):
    ``Mylodonin-2``, ``Mylodonin-3``, ``Equusin-4``, ``Mammuthusin-3``, ``Hesperelin-3``.

    Writes pair predictions and a metrics summary with ``log10MAE``, ``RSE``,
    ``PCC``, ``KCC`` (overall + per family).
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    ckpt_path = Path(checkpoint or cfg.checkpoint or "")
    if not ckpt_path or not ckpt_path.exists():
        raise ValueError("evaluate_apexgo_high_span requires a valid checkpoint path")

    peptide_csv = _resolve_path(
        peptide_csv or "metadata/test_apexgo_high_span_families.csv",
        root,
    )
    pair_csv = _resolve_path(
        pair_csv or "metadata/test_apexgo_high_span_pairs.csv",
        root,
    )
    pdb_dir = _resolve_path(
        pdb_dir or "data/3D_data_apexgo_Rosetta",
        root,
    )
    feature_path = _resolve_path(cfg.feature_path, root)
    out_dir = Path(out_dir) if out_dir is not None else (
        Path(cfg.out_dir) if Path(cfg.out_dir).is_absolute() else root / cfg.out_dir
    )
    out_dir = out_dir / "apexgo_high_span"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = list(families) if families is not None else list(APEXGO_HIGH_SPAN_FAMILIES)
    peptides = pd.read_csv(peptide_csv)
    pairs = pd.read_csv(pair_csv)
    if "family" in peptides.columns:
        peptides = peptides[peptides["family"].isin(selected)].copy()
    if "family" in pairs.columns:
        pairs = pairs[pairs["family"].isin(selected)].copy()
    if peptides.empty or pairs.empty:
        raise RuntimeError(f"No APEXGO peptides/pairs left after family filter={selected}")

    # Persist filtered CSVs for preprocess_dataset (expects MIC_Escherichia_coli).
    tmp_peptide_csv = out_dir / "peptides_filtered.csv"
    keep_cols = [c for c in ["sequence", "n_terminus", "c_terminus", CFU_COL, TARGET_COL] if c in peptides.columns]
    peptides[keep_cols].to_csv(tmp_peptide_csv, index=False)

    eval_cfg = Config(**{**cfg.to_dict()})
    eval_cfg.feature_path = str(feature_path)
    eval_cfg.use_cfu_protocol = True
    eval_cfg.use_cfu_feature = True
    eval_cfg.primary_cfu_group = "1E5 - 1E6"

    filt, prep_stats = preprocess_dataset(str(tmp_peptide_csv), str(pdb_dir), eval_cfg, "apexgo_high_span")
    if filt.empty:
        raise RuntimeError(f"No peptides survived preprocessing: {prep_stats}")
    graphs, rows, graph_stats = build_graphs(filt, eval_cfg)
    if not graphs:
        raise RuntimeError(f"Graph construction failed: {graph_stats}")

    # Re-attach family / peptide_id via sequence join for reporting.
    meta_by_seq = peptides.drop_duplicates("sequence").set_index("sequence")
    for col in ("peptide_id", "family", "variant_idx"):
        if col in meta_by_seq.columns:
            rows[col] = rows["sequence"].map(meta_by_seq[col])

    device = torch.device(cfg.device if torch.cuda.is_available() and "cuda" in str(cfg.device) else "cpu")
    model, ckpt = load_checkpoint_model(ckpt_path, eval_cfg, device)

    # Optional z-score stats from the training run.
    if str(eval_cfg.global_feature_scaling).lower() == "zscore":
        from .utils.scaling import apply_global_f_stats, cache_raw_global_f

        cache_raw_global_f(graphs)
        for sp in (
            ckpt_path.parent.parent / "intermediate" / "global_feature_zscore_final.json",
            Path(cfg.out_dir) / "intermediate" / "global_feature_zscore_final.json",
            root / cfg.out_dir / "intermediate" / "global_feature_zscore_final.json",
        ):
            if sp.exists():
                apply_global_f_stats(graphs, json.loads(sp.read_text()))
                break
    if str(getattr(eval_cfg, "node_feature_scaling", "none")).lower() == "zscore":
        from .utils.scaling import apply_node_x_stats, cache_raw_node_x

        cache_raw_node_x(graphs)
        for sp in (
            ckpt_path.parent.parent / "intermediate" / "node_feature_zscore_final.json",
            Path(cfg.out_dir) / "intermediate" / "node_feature_zscore_final.json",
            root / cfg.out_dir / "intermediate" / "node_feature_zscore_final.json",
        ):
            if sp.exists():
                apply_node_x_stats(graphs, json.loads(sp.read_text()))
                break

    pred, metrics = eval_explicit_pairs(
        model,
        graphs,
        rows,
        pairs,
        eval_cfg,
        device,
        bidirectional=bidirectional,
    )

    calibrator = {"slope": 1.0, "intercept": 0.0}
    cal_path = ckpt_path.parent.parent / "results" / "calibrator.json"
    if apply_saved_calibrator and cal_path.exists() and len(pred):
        calibrator = json.loads(cal_path.read_text())
        pred = apply_calibrator(pred, calibrator)
        # Keep log10 columns consistent after calibration.
        pred["y_pred_delta_log10_anchor_minus_query"] = (
            pred["y_pred_delta_log2_anchor_minus_query"] * LOG10_OF_2
        )
        metrics = _metrics_payload(pred)
        if "family" in pred.columns and pred["family"].notna().any():
            metrics["per_family"] = {
                str(fam): _metrics_payload(group) for fam, group in pred.groupby("family", dropna=True)
            }

    pred_path = out_dir / "pair_predictions.csv"
    pred.to_csv(pred_path, index=False)
    summary = {
        "checkpoint": str(ckpt_path),
        "families": selected,
        "n_peptides": int(len(rows)),
        "n_pairs": int(len(pred)),
        "bidirectional": bool(bidirectional),
        "calibrator": calibrator,
        "preprocess_stats": prep_stats,
        "graph_stats": graph_stats,
        "metrics": {k: v for k, v in metrics.items() if k != "per_family"},
        "per_family": metrics.get("per_family", {}),
        "note": "Model predicts log2-MIC deltas; log10MAE = log2MAE * log10(2). RSE/PCC/KCC are scale-invariant.",
        "wrote": str(pred_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"apexgo_high_span": summary["metrics"], "per_family": summary["per_family"], "n_pairs": summary["n_pairs"]}, indent=2))
    return summary
