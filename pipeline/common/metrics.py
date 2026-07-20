"""Classification metrics used across TIGER pipeline classifiers.

Reports the full suite requested for manuscript evaluation:
  F1-Score, Precision, Recall, Accuracy, MCC, AUC-ROC, AUC-PR
for every CV fold and the held-out test set.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "auc_roc",
    "auc_pr",
]


def _safe_auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _safe_auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def classification_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    y_prob: Iterable | None = None,
    average: str = "binary",
    zero_division: int = 0,
) -> dict[str, float]:
    """Compute the full classification metric suite.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth / hard predictions (0/1).
    y_prob:
        Positive-class probabilities for AUC metrics. If omitted, `y_pred`
        is used as a degenerate score (not recommended).
    """
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    if y_prob is None:
        y_score = yp.astype(float)
    else:
        y_score = np.asarray(y_prob, dtype=float).ravel()

    out = {
        "n": int(len(yt)),
        "n_pos": int((yt == 1).sum()),
        "n_neg": int((yt == 0).sum()),
        "accuracy": float(accuracy_score(yt, yp)) if len(yt) else float("nan"),
        "precision": float(
            precision_score(yt, yp, average=average, zero_division=zero_division)
        )
        if len(yt)
        else float("nan"),
        "recall": float(recall_score(yt, yp, average=average, zero_division=zero_division))
        if len(yt)
        else float("nan"),
        "f1": float(f1_score(yt, yp, average=average, zero_division=zero_division))
        if len(yt)
        else float("nan"),
        "mcc": float(matthews_corrcoef(yt, yp)) if len(yt) > 1 else float("nan"),
        "auc_roc": _safe_auc_roc(yt, y_score),
        "auc_pr": _safe_auc_pr(yt, y_score),
    }
    return out


def metrics_to_row(split: str, model: str, fold: int | str, metrics: dict[str, float]) -> dict:
    row = {"split": split, "model": model, "fold": fold}
    row.update({k: metrics.get(k, float("nan")) for k in ["n", "n_pos", "n_neg", *METRIC_COLUMNS]})
    return row


def summarize_cv(fold_rows: list[dict]) -> pd.DataFrame:
    """Aggregate mean±std over CV folds for each model."""
    df = pd.DataFrame(fold_rows)
    if df.empty:
        return df
    rows = []
    for model, g in df.groupby("model"):
        row = {"split": "cv_mean", "model": model, "fold": "mean"}
        row_std = {"split": "cv_std", "model": model, "fold": "std"}
        for col in METRIC_COLUMNS:
            row[col] = float(g[col].mean())
            row_std[col] = float(g[col].std(ddof=0))
        row["n"] = int(g["n"].sum())
        row_std["n"] = int(g["n"].sum())
        rows.extend([row, row_std])
    return pd.DataFrame(rows)


def format_metrics_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> pd.DataFrame:
    out = df.copy()
    for col in METRIC_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(lambda x: float_fmt.format(x) if pd.notna(x) else "NA")
    return out
