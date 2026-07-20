#!/usr/bin/env python3
"""
Evaluate hemolytic / toxicity classifiers with a full metric suite.

Reports for every model, every CV fold, and the held-out test set:
  Accuracy, Precision, Recall, F1, MCC, AUC-ROC, AUC-PR

Two modes:
  1) Train/evaluate classical ML baselines on a labeled CSV (recommended for custom data)
  2) Optionally also score a pretrained FusionPeptide checkpoint on the same held-out test set

Example:
  python evaluate_models.py \\
    --csv examples/labeled_toxicity_demo.csv \\
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common.metrics import (  # noqa: E402
    classification_metrics,
    format_metrics_table,
    metrics_to_row,
    summarize_cv,
)

AA_TO_IDX = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}


def calculate_property(seq: str) -> list[float]:
    pa = ProteinAnalysis(seq)
    aa_counts = pa.count_amino_acids()
    length = len(seq)
    aliphatic = (aa_counts["A"] + 2.9 * aa_counts["V"] + 3.9 * (aa_counts["I"] + aa_counts["L"])) / length
    charge_density = (
        sum(aa_counts.get(a, 0) for a in ["R", "K", "H"]) - sum(aa_counts.get(a, 0) for a in ["D", "E"])
    ) / length
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
        round(charge_density, 3) * 10,
    ]


def featurize_globf(sequences: list[str]) -> np.ndarray:
    return np.asarray([calculate_property(s) for s in sequences], dtype=float)


def load_labeled_csv(path: Path, sequence_col: str, label_col: str, max_len: int = 30):
    df = pd.read_csv(path)
    if sequence_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{sequence_col}' and '{label_col}'. Found: {list(df.columns)}"
        )
    df = df[[sequence_col, label_col]].dropna().copy()
    df[sequence_col] = df[sequence_col].astype(str).str.upper().str.strip()
    df = df[df[sequence_col].str.len().between(6, max_len)]
    df = df[~df[sequence_col].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", regex=True)]
    df[label_col] = df[label_col].astype(int)
    if df[label_col].nunique() < 2:
        raise ValueError("Need both classes (0/1) for classification metrics.")
    X = featurize_globf(df[sequence_col].tolist())
    y = df[label_col].to_numpy(dtype=int)
    return df.reset_index(drop=True), X, y


def build_models() -> dict:
    return {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]
        ),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }


def predict_labels_and_probs(model, X):
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = pred.astype(float)
    return pred, prob


def evaluate_baselines(X, y, n_splits=5, test_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    models = build_models()
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
            print(f"  fold {fold_i}: F1={metrics['f1']:.4f} AUC-ROC={metrics['auc_roc']:.4f}")

        clf = clone(model)
        clf.fit(X_train, y_train)
        pred, prob = predict_labels_and_probs(clf, X_test)
        test_metrics = classification_metrics(y_test, pred, prob)
        test_rows.append(metrics_to_row("test", model_name, "heldout", test_metrics))
        fitted[model_name] = clf
        print(
            f"  TEST: Acc={test_metrics['accuracy']:.4f} F1={test_metrics['f1']:.4f} "
            f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} "
            f"MCC={test_metrics['mcc']:.4f} AUC-ROC={test_metrics['auc_roc']:.4f} "
            f"AUC-PR={test_metrics['auc_pr']:.4f}"
        )

    return fold_rows, test_rows, fitted, X_test, y_test, {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_indices_note": "held-out stratified split",
    }


def evaluate_pretrained_checkpoint(df: pd.DataFrame, y: np.ndarray, test_mask: np.ndarray, args):
    """Score pretrained FusionPeptide on sequences corresponding to the held-out labels."""
    import torch
    from torch.utils.data import DataLoader

    from dataset import PeptideInferenceDataset
    from network import FusionPeptide

    # Write a temporary headerless CSV for PeptideInferenceDataset
    tmp_csv = Path(args.out_dir) / "_tmp_test_sequences.csv"
    test_df = df.iloc[np.where(test_mask)[0]].copy()
    # PeptideInferenceDataset expects headerless seq[,...]
    test_df[["sequence", "sequence"]].to_csv(tmp_csv, header=False, index=False)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = FusionPeptide(
        classes=1,
        q_encoder=args.q_encoder,
        v_encoder="resnet34",
        channels=args.channels,
        mode=args.mode,
    ).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = PeptideInferenceDataset(csv=str(tmp_csv), max_length=args.max_len, model_mode=args.mode)
    # Align labels by sequence string (dataset may drop invalid rows)
    heldout = df.iloc[np.where(test_mask)[0]]
    seq_to_label = {
        str(s).upper(): int(l)
        for s, l in zip(heldout[args.sequence_col], y[test_mask])
    }

    loader = DataLoader(ds, batch_size=32, shuffle=False)
    probs, labels, kept = [], [], []
    with torch.no_grad():
        for voxel, seq, globf, _gt, seq_str in loader:
            out = torch.sigmoid(model((voxel.to(device), seq.to(device), globf.to(device)))).cpu().numpy().ravel()
            for s, p in zip(seq_str, out):
                if s not in seq_to_label:
                    continue
                probs.append(float(p))
                labels.append(seq_to_label[s])
                kept.append(s)

    if not labels:
        print("Warning: pretrained checkpoint evaluation produced no aligned labels.")
        return None

    pred = (np.asarray(probs) >= args.threshold).astype(int)
    metrics = classification_metrics(labels, pred, probs)
    print(
        f"\n=== pretrained FusionPeptide ({Path(args.model_path).name}) on held-out-like set ===\n"
        f"  n={metrics['n']} Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
        f"MCC={metrics['mcc']:.4f} AUC-ROC={metrics['auc_roc']:.4f} AUC-PR={metrics['auc_pr']:.4f}"
    )
    return metrics_to_row("test", "fusionpeptide_pretrained", "heldout", metrics)


def parse_args():
    p = argparse.ArgumentParser(description="Full-suite toxicity classifier evaluation")
    p.add_argument("--csv", required=True, help="Labeled CSV: sequence + binary toxicity label")
    p.add_argument("--sequence-col", default="sequence")
    p.add_argument("--label-col", default="label")
    p.add_argument("--out-dir", default="outputs/metrics")
    p.add_argument("--max-len", type=int, default=30)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-pretrained", action="store_true",
                   help="Also evaluate checkpoints/model_1.pth on a held-out split")
    p.add_argument("--model-path", default=str(HERE / "checkpoints" / "model_1.pth"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--q-encoder", default="gru")
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--mode", default="101")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, X, y = load_labeled_csv(Path(args.csv), args.sequence_col, args.label_col, args.max_len)
    # Keep sequence column name consistent for optional pretrained eval
    if args.sequence_col != "sequence":
        df = df.rename(columns={args.sequence_col: "sequence"})
        args.sequence_col = "sequence"

    fold_rows, test_rows, fitted, X_test, y_test, meta = evaluate_baselines(
        X, y, n_splits=args.n_splits, test_size=args.test_size, seed=args.seed
    )

    if args.eval_pretrained:
        # Recreate the same held-out mask via the same split
        idx = np.arange(len(y))
        _, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=y)
        test_mask = np.zeros(len(y), dtype=bool)
        test_mask[test_idx] = True
        pretrained_row = evaluate_pretrained_checkpoint(df, y, test_mask, args)
        if pretrained_row is not None:
            test_rows.append(pretrained_row)

    cv_df = pd.DataFrame(fold_rows)
    test_df = pd.DataFrame(test_rows)
    summary_df = pd.concat([summarize_cv(fold_rows), test_df], ignore_index=True)

    cv_df.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    test_df.to_csv(out_dir / "test_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    format_metrics_table(summary_df).to_csv(out_dir / "summary_metrics_pretty.csv", index=False)
    (out_dir / "run_meta.json").write_text(json.dumps({**meta, "csv": str(args.csv), "models": list(fitted)}, indent=2, default=str))

    print("\n=== Summary ===")
    print(format_metrics_table(summary_df).to_string(index=False))
    print(f"\nWrote metrics to {out_dir}")


if __name__ == "__main__":
    main()
