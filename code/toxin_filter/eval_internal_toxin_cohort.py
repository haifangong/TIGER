#!/usr/bin/env python3
"""Evaluate published toxin checkpoints on the 88-peptide internal_toxin_cohort panel.

Reads the clean labels pack under
``data/test_external/test_toxin_internal_toxin_cohort/`` and scores the five-fold
``both`` (or chosen mode) classical-ML checkpoints under
``checkpoints/ablation_toxin/<mode>/checkpoints/``.

Writes predictions + metrics next to the labels pack (or ``--out-dir``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.toxin_filter.features import apply_zscore, build_tabular_features  # noqa: E402

DEFAULT_LABELS = (
    ROOT
    / "data"
    / "test_external"
    / "test_toxin_internal_toxin_cohort"
    / "internal_toxin_cohort_toxicity_labeled.csv"
)
DEFAULT_PANEL = (
    ROOT
    / "data"
    / "test_external"
    / "test_toxin_internal_toxin_cohort"
    / "internal_toxin_cohort_hemolysis_active_micmin_le128.csv"
)
DEFAULT_CKPT = ROOT / "checkpoints" / "ablation_toxin" / "both" / "checkpoints"


def _load_panel(labels_csv: Path, panel_csv: Path | None) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    # Normalize column names used across packs.
    colmap = {c.lower(): c for c in labels.columns}
    seq_col = colmap.get("sequence") or colmap.get("seq")
    if seq_col is None:
        raise SystemExit(f"No sequence column in {labels_csv}")
    out = pd.DataFrame({"sequence": labels[seq_col].astype(str).str.upper().str.strip()})
    if "label" in labels.columns:
        out["label"] = labels["label"].astype(int)
    elif "gt_toxin_le512" in labels.columns:
        out["label"] = labels["gt_toxin_le512"].astype(int)
    else:
        raise SystemExit(f"No binary label column in {labels_csv}")
    if "hc50" in labels.columns:
        out["hc50"] = labels["hc50"]
    elif "HC50" in labels.columns:
        out["hc50"] = labels["HC50"]
    if panel_csv is not None and panel_csv.exists():
        panel = pd.read_csv(panel_csv)
        # Attach mic_min / richer columns when available (by sequence).
        pseq = "sequence" if "sequence" in panel.columns else ("Seq" if "Seq" in panel.columns else None)
        if pseq:
            panel = panel.copy()
            panel["_seq"] = panel[pseq].astype(str).str.upper().str.strip()
            keep = [
                c
                for c in panel.columns
                if c
                not in {
                    pseq,
                    "_seq",
                    "label",
                    "hc50",
                    "HC50",
                    "sequence",
                    "Seq",
                    "row_id",
                    "ID",
                    "label_rule",
                    "label_deterministic",
                }
            ]
            merge = panel[["_seq"] + keep].drop_duplicates("_seq")
            out = out.merge(merge, left_on="sequence", right_on="_seq", how="left").drop(columns=["_seq"])
    out = out.drop(columns=["row_id"], errors="ignore")
    out.insert(0, "row_id", np.arange(1, len(out) + 1))
    return out


def predict(df: pd.DataFrame, ckpt_dir: Path, feature_mode: str = "both") -> pd.DataFrame:
    feat_df = pd.DataFrame(
        {
            "sequence": df["sequence"],
            "n_terminus": [""] * len(df),
            "c_terminus": [""] * len(df),
        }
    )
    X_raw = build_tabular_features(feat_df, feature_mode=feature_mode)
    model_names = sorted({p.name.split("_fold")[0] for p in ckpt_dir.glob("*_fold*.joblib")})
    if not model_names:
        raise SystemExit(f"No *_fold*.joblib under {ckpt_dir}")

    out = df.copy()
    all_probs: dict[str, np.ndarray] = {}
    for name in model_names:
        fold_probs = []
        fold_preds = []
        for fold in range(1, 6):
            path = ckpt_dir / f"{name}_fold{fold}.joblib"
            if not path.exists():
                raise SystemExit(f"Missing checkpoint {path}")
            payload = joblib.load(path)
            stats = {k: np.asarray(v) for k, v in payload["zscore_stats"].items()}
            X = apply_zscore(X_raw, stats)
            model = payload["model"]
            thr = float(payload.get("decision_threshold", 0.5))
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[:, 1]
            else:
                scores = model.decision_function(X)
                prob = 1.0 / (1.0 + np.exp(-scores))
            pred = (prob >= thr).astype(int)
            fold_probs.append(prob)
            fold_preds.append(pred)
            out[f"prob_{name}_fold{fold}"] = prob
            out[f"pred_{name}_fold{fold}"] = pred
        mean_prob = np.mean(fold_probs, axis=0)
        maj_pred = (np.mean(fold_preds, axis=0) >= 0.5).astype(int)
        out[f"prob_{name}"] = mean_prob
        out[f"pred_{name}"] = maj_pred
        all_probs[name] = mean_prob

    ens = np.mean(np.stack(list(all_probs.values()), axis=0), axis=0)
    out["prob_ensemble"] = ens
    out["pred_ensemble"] = (ens >= 0.5).astype(int)
    # Prefer CatBoost as the primary reported model when present.
    primary = "CatBoost" if "CatBoost" in all_probs else sorted(all_probs)[0]
    out["prob_toxin"] = out[f"prob_{primary}"]
    out["pred_toxin"] = out[f"pred_{primary}"]
    out["primary_model"] = primary
    out["label_threshold"] = 512
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--panel-csv", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--feature-mode", default="both", choices=["both", "global", "sequence"])
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "test_external" / "test_toxin_internal_toxin_cohort",
    )
    args = p.parse_args(argv)

    if not args.labels_csv.exists():
        raise SystemExit(f"Missing labels: {args.labels_csv}")
    if not args.ckpt_dir.exists():
        raise SystemExit(f"Missing checkpoints: {args.ckpt_dir}")

    df = _load_panel(args.labels_csv, args.panel_csv if args.panel_csv.exists() else None)
    pred = predict(df, args.ckpt_dir, feature_mode=args.feature_mode)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "internal_toxin_cohort_hemolysis_predictions_both.csv"
    keep = [
        c
        for c in [
            "row_id",
            "sequence",
            "hc50",
            "label",
            "mic_min",
            "prob_toxin",
            "pred_toxin",
            "primary_model",
            "prob_CatBoost",
            "pred_CatBoost",
            "prob_XGB",
            "pred_XGB",
            "prob_LGBM",
            "pred_LGBM",
            "prob_RF",
            "pred_RF",
            "prob_SVM",
            "pred_SVM",
            "prob_GB",
            "pred_GB",
            "prob_MLP",
            "pred_MLP",
            "prob_Adaboost",
            "pred_Adaboost",
            "prob_LR",
            "pred_LR",
            "prob_ensemble",
            "pred_ensemble",
            "label_threshold",
        ]
        if c in pred.columns
    ]
    pred[keep].to_csv(pred_path, index=False)
    pred.to_csv(out_dir / "internal_toxin_cohort_hemolysis_predictions_both_full.csv", index=False)

    y = pred["label"].to_numpy(dtype=int)
    summary = {
        "n": int(len(pred)),
        "feature_mode": args.feature_mode,
        "checkpoints": str(args.ckpt_dir),
        "labels_csv": str(args.labels_csv),
        "primary_model": str(pred["primary_model"].iloc[0]),
    }
    for name in ["toxin", "CatBoost", "XGB", "ensemble"]:
        pcol = f"pred_{name}"
        probcol = f"prob_{name}"
        if pcol not in pred.columns:
            continue
        summary[name] = _metrics(y, pred[pcol].to_numpy(dtype=int), pred[probcol].to_numpy(dtype=float))

    (out_dir / "internal_toxin_cohort_prediction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
