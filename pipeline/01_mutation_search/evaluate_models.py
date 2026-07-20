#!/usr/bin/env python3
"""
Evaluate antimicrobial activity classifiers with a full metric suite.

Reports for every model, every CV fold, and the held-out test set:
  Accuracy, Precision, Recall, F1, MCC, AUC-ROC, AUC-PR

Example (custom labeled data):
  python evaluate_models.py \\
    --csv examples/labeled_activity_demo.csv \\
    --sequence-col sequence --label-col label \\
    --out-dir outputs/metrics_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.metrics import (  # noqa: E402
    classification_metrics,
    format_metrics_table,
    metrics_to_row,
    summarize_cv,
)

AA_TO_IDX = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
VALID_AA = set(AA_TO_IDX)


def calculate_property(seq: str) -> list[float]:
    pa = ProteinAnalysis(seq)
    aa_counts = pa.count_amino_acids()
    length = len(seq)
    aliphatic = (aa_counts["A"] + 2.9 * aa_counts["V"] + 3.9 * (aa_counts["I"] + aa_counts["L"])) / length
    charged = sum(aa_counts[a] for a in ["D", "E", "R", "K", "H"]) / length
    helix, sheet, turn = pa.secondary_structure_fraction()
    return [
        round(pa.gravy(), 3) * 10,
        round(aliphatic, 3) * 10,
        round(pa.aromaticity(), 3) * 10,
        round(pa.instability_index(), 3),
        round(helix * 10, 3),
        round(sheet * 10, 3),
        round(turn * 10, 3),
        round(pa.charge_at_pH(7), 3),
        round(pa.isoelectric_point(), 3),
        round(charged, 3) * 10,
    ]


def featurize_sequences(sequences: list[str], max_len: int = 30) -> np.ndarray:
    rows = []
    for seq in sequences:
        idxs = [AA_TO_IDX[a] for a in seq]
        padded = idxs + [0] * (max_len - len(idxs))
        rows.append(padded[:max_len] + calculate_property(seq))
    return np.asarray(rows, dtype=float)


def load_labeled_csv(path: Path, sequence_col: str, label_col: str, max_len: int = 30):
    df = pd.read_csv(path)
    if sequence_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{sequence_col}' and '{label_col}'. "
            f"Found: {list(df.columns)}"
        )
    df = df[[sequence_col, label_col]].dropna().copy()
    df[sequence_col] = df[sequence_col].astype(str).str.upper().str.strip()
    df = df[df[sequence_col].str.len().between(6, max_len)]
    df = df[~df[sequence_col].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", regex=True)]
    df[label_col] = df[label_col].astype(int)
    if df[label_col].nunique() < 2:
        raise ValueError("Need both positive and negative labels for classification metrics.")
    X = featurize_sequences(df[sequence_col].tolist(), max_len=max_len)
    y = df[label_col].to_numpy(dtype=int)
    return df, X, y


def build_models(include_catboost: bool = True) -> dict:
    models = {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]
        ),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "adaboost": AdaBoostClassifier(random_state=42),
        "knn": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]),
        "svm_rbf": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=42))]
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=42)),
            ]
        ),
    }
    if include_catboost:
        try:
            from catboost import CatBoostClassifier

            models["catboost"] = CatBoostClassifier(
                iterations=300,
                learning_rate=0.1,
                depth=6,
                loss_function="Logloss",
                verbose=False,
                random_seed=42,
                allow_writing_files=False,
            )
        except ImportError:
            print("Warning: catboost not installed; skipping CatBoost.")
    return models


def predict_labels_and_probs(model, X):
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # map to [0,1] via logistic for AUC stability
        prob = 1.0 / (1.0 + np.exp(-scores))
    else:
        prob = pred.astype(float)
    return pred, prob


def evaluate_all(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    test_size: float = 0.2,
    seed: int = 42,
    include_catboost: bool = True,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    models = build_models(include_catboost=include_catboost)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_rows, test_rows = [], []
    fitted = {}

    for model_name, model in models.items():
        print(f"\n=== {model_name} ===")
        for fold_i, (tr, va) in enumerate(skf.split(X_train, y_train), start=1):
            clf = clone(model)
            clf.fit(X_train[tr], y_train[tr])
            pred, prob = predict_labels_and_probs(clf, X_train[va])
            metrics = classification_metrics(y_train[va], pred, prob)
            fold_rows.append(metrics_to_row("cv", model_name, fold_i, metrics))
            print(
                f"  fold {fold_i}: F1={metrics['f1']:.4f}  "
                f"AUC-ROC={metrics['auc_roc']:.4f}  MCC={metrics['mcc']:.4f}"
            )

        # Refit on all training data, evaluate held-out test
        clf = clone(model)
        clf.fit(X_train, y_train)
        pred, prob = predict_labels_and_probs(clf, X_test)
        test_metrics = classification_metrics(y_test, pred, prob)
        test_rows.append(metrics_to_row("test", model_name, "heldout", test_metrics))
        fitted[model_name] = clf
        print(
            f"  TEST: Acc={test_metrics['accuracy']:.4f}  F1={test_metrics['f1']:.4f}  "
            f"P={test_metrics['precision']:.4f}  R={test_metrics['recall']:.4f}  "
            f"MCC={test_metrics['mcc']:.4f}  AUC-ROC={test_metrics['auc_roc']:.4f}  "
            f"AUC-PR={test_metrics['auc_pr']:.4f}"
        )

    return fold_rows, test_rows, fitted, {"n_train": int(len(y_train)), "n_test": int(len(y_test))}


def parse_args():
    p = argparse.ArgumentParser(description="Full-suite CV/test evaluation for activity classifiers")
    p.add_argument("--csv", required=True, help="Labeled CSV with sequence + binary label columns")
    p.add_argument("--sequence-col", default="sequence", help="Sequence column name")
    p.add_argument("--label-col", default="label", help="Binary label column (0/1)")
    p.add_argument("--out-dir", default="outputs/metrics", help="Where to write metric tables")
    p.add_argument("--max-len", type=int, default=30)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-catboost", action="store_true")
    p.add_argument("--save-models", action="store_true", help="Save fitted models to out-dir/models")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, X, y = load_labeled_csv(Path(args.csv), args.sequence_col, args.label_col, args.max_len)
    fold_rows, test_rows, fitted, meta = evaluate_all(
        X,
        y,
        n_splits=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
        include_catboost=not args.no_catboost,
    )

    cv_df = pd.DataFrame(fold_rows)
    test_df = pd.DataFrame(test_rows)
    summary_df = pd.concat([summarize_cv(fold_rows), test_df], ignore_index=True)

    cv_df.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    test_df.to_csv(out_dir / "test_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    format_metrics_table(summary_df).to_csv(out_dir / "summary_metrics_pretty.csv", index=False)
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                **meta,
                "csv": str(args.csv),
                "n_splits": args.n_splits,
                "test_size": args.test_size,
                "seed": args.seed,
                "models": list(fitted.keys()),
            },
            indent=2,
        )
    )

    if args.save_models:
        model_dir = out_dir / "models"
        model_dir.mkdir(exist_ok=True)
        for name, clf in fitted.items():
            joblib.dump(clf, model_dir / f"{name}.pkl")

    print("\n=== Summary (CV mean / test) ===")
    print(format_metrics_table(summary_df).to_string(index=False))
    print(f"\nWrote metrics to {out_dir}")


if __name__ == "__main__":
    main()
