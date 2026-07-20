"""Regression metrics and calibrators.

Primary selection metrics (targets are already log2-MIC pair deltas):
  log2MAE  – mean absolute error on log2-delta
  RSE      – relative squared error = SSE / SST
  PCC      – Pearson correlation
  KCC      – Kendall's tau (rank correlation)

Checkpoint / leaderboard score (lower better):
  selection_score = log2MAE + RSE - PCC - KCC
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    out: dict[str, float] = {"n": int(len(yt))}
    if len(yt) == 0:
        return out

    mae = float(mean_absolute_error(yt, yp))
    mse = float(mean_squared_error(yt, yp))
    rmse = float(math.sqrt(mse))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    rse = float(ss_res / ss_tot) if ss_tot > 0 else float("nan")
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 else float("nan")

    can_corr = len(yt) > 2 and np.std(yt) > 0 and np.std(yp) > 0
    pcc = float(pearsonr(yt, yp)[0]) if can_corr else float("nan")
    kcc = float(kendalltau(yt, yp)[0]) if can_corr else float("nan")
    spear = float(spearmanr(yt, yp)[0]) if can_corr else float("nan")

    # Canonical names used for checkpoint selection / reporting
    out.update(
        {
            "log2MAE": mae,
            "RSE": rse,
            "PCC": pcc,
            "KCC": kcc,
            # aliases kept for backward compatibility
            "mae": mae,
            "rmse": rmse,
            "mse": mse,
            "r2": r2,
            "rse": rse,
            "pearson": pcc,
            "spearman": spear,
            "kendall": kcc,
        }
    )
    out["selection_score"] = selection_score(out)
    return out


def selection_score(metrics: dict[str, float]) -> float:
    """Lower is better: log2MAE + RSE - PCC - KCC."""
    log2_mae = metrics.get("log2MAE", metrics.get("mae", float("nan")))
    rse = metrics.get("RSE", metrics.get("rse", float("nan")))
    pcc = metrics.get("PCC", metrics.get("pearson", float("nan")))
    kcc = metrics.get("KCC", metrics.get("kendall", float("nan")))
    vals = [log2_mae, rse, pcc, kcc]
    if any(not np.isfinite(v) for v in vals):
        return float("inf")
    return float(log2_mae + rse - pcc - kcc)


def is_better_selection(new: dict[str, float], best_score: float, min_delta: float = 0.0) -> bool:
    """Return True if ``new`` improves on ``best_score`` (lower selection_score)."""
    score = new.get("selection_score")
    if score is None or not np.isfinite(score):
        score = selection_score(new)
    return score < best_score - min_delta


def fit_calibrator(cv_predictions: pd.DataFrame) -> dict[str, float]:
    if cv_predictions.empty:
        return {"slope": 1.0, "intercept": 0.0}
    x = cv_predictions["y_pred_delta_log2_anchor_minus_query"].to_numpy(dtype=float).reshape(-1, 1)
    y = cv_predictions["y_true_delta_log2_anchor_minus_query"].to_numpy(dtype=float)
    mask = np.isfinite(x.ravel()) & np.isfinite(y)
    if mask.sum() < 3:
        return {"slope": 1.0, "intercept": 0.0}
    model = LinearRegression().fit(x[mask], y[mask])
    return {"slope": float(model.coef_[0]), "intercept": float(model.intercept_)}


def apply_calibrator(df: pd.DataFrame, calibrator: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["y_pred_delta_raw"] = out["y_pred_delta_log2_anchor_minus_query"]
    out["y_pred_delta_log2_anchor_minus_query"] = (
        calibrator["slope"] * out["y_pred_delta_log2_anchor_minus_query"] + calibrator["intercept"]
    )
    return out
