"""Regression metrics and calibrators.

Primary selection metrics (targets are already log2-MIC pair deltas):
  log2MAE  – mean absolute error on log2-delta
  log10MAE – mean absolute error on log10-delta (= log2MAE * log10(2))
  RSE      – relative squared error = SSE / SST  (scale-invariant)
  PCC      – Pearson correlation                 (scale-invariant)
  KCC      – Kendall's tau (rank correlation)    (scale-invariant)

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

LOG10_OF_2 = math.log10(2.0)


def regression_metrics(y_true, y_pred, *, input_log_base: int | float = 2) -> dict[str, float]:
    """Compute regression metrics for pair-delta predictions.

    Parameters
    ----------
    y_true, y_pred
        Pair deltas. By default these are in log2-MIC space (model training target).
    input_log_base
        Base of the input deltas. Use ``2`` (default) for model outputs, or ``10``
        if deltas are already in log10 space.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    out: dict[str, float] = {"n": int(len(yt))}
    if len(yt) == 0:
        return out

    # Normalize to log2 space for the canonical training-target metrics.
    if float(input_log_base) == 2.0:
        yt2, yp2 = yt, yp
    elif float(input_log_base) == 10.0:
        yt2, yp2 = yt / LOG10_OF_2, yp / LOG10_OF_2
    else:
        scale = math.log(2.0) / math.log(float(input_log_base))
        yt2, yp2 = yt * scale, yp * scale

    mae2 = float(mean_absolute_error(yt2, yp2))
    mae10 = float(mae2 * LOG10_OF_2)
    mse = float(mean_squared_error(yt2, yp2))
    rmse = float(math.sqrt(mse))
    ss_res = float(np.sum((yt2 - yp2) ** 2))
    ss_tot = float(np.sum((yt2 - yt2.mean()) ** 2))
    rse = float(ss_res / ss_tot) if ss_tot > 0 else float("nan")
    r2 = float(r2_score(yt2, yp2)) if len(yt2) > 1 else float("nan")

    can_corr = len(yt2) > 2 and np.std(yt2) > 0 and np.std(yp2) > 0
    pcc = float(pearsonr(yt2, yp2)[0]) if can_corr else float("nan")
    kcc = float(kendalltau(yt2, yp2)[0]) if can_corr else float("nan")
    spear = float(spearmanr(yt2, yp2)[0]) if can_corr else float("nan")

    # Canonical names used for checkpoint selection / reporting
    out.update(
        {
            "log2MAE": mae2,
            "log10MAE": mae10,
            "RSE": rse,
            "PCC": pcc,
            "KCC": kcc,
            # aliases kept for backward compatibility
            "mae": mae2,
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
