#!/usr/bin/env python3
"""Run 5-fold CV toxicity classification (sequence + global features only).

Label (inequality-aware): toxin=1 if HC50 is determinably <= threshold.
Default threshold = 512 ug/mL. Ambiguous censored values (e.g. >128 when
T=512) are excluded from the main-task training set.

Feature modes for classical ML:
  both     : length + 10 global props + 20 AA frequencies  (default)
  global   : 10 global physicochemical props only
  sequence : length + 20 AA frequencies only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
TIGER_ROOT = CODE_ROOT.parent
if str(TIGER_ROOT) not in sys.path:
    sys.path.insert(0, str(TIGER_ROOT))

from code.toxin_filter.data import load_toxicity_table, summarize_label_balance  # noqa: E402
from code.toxin_filter.features import apply_zscore, fit_zscore  # noqa: E402
from code.toxin_filter.metrics import (  # noqa: E402
    format_metrics_table,
    metrics_to_row,
    summarize_cv,
)
from code.toxin_filter.train_eval import (  # noqa: E402
    _imbalance_ratio,
    build_ml_models,
    evaluate_ml_fold,
    prepare_feature_matrices,
    train_fusion_fold,
    train_metric_fold,
    train_tiger_fold,
)


def parse_args():
    p = argparse.ArgumentParser(description="Toxicity filter 5-fold CV evaluation")
    p.add_argument("--csv", default=None, help="Optional legacy numeric CSV (not recommended)")
    p.add_argument("--json-dir", default=None, help="DBAASP JSON export directory")
    p.add_argument(
        "--source",
        choices=["json", "csv"],
        default="json",
        help="Label source: inequality-aware JSON (default) or legacy numeric CSV",
    )
    p.add_argument("--threshold", type=float, default=512.0, help="HC50 toxin boundary (ug/mL)")
    p.add_argument(
        "--feature-mode",
        choices=["both", "global", "sequence"],
        default="both",
        help="Classical ML feature set",
    )
    p.add_argument("--min-len", type=int, default=6)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--gpu", type=int, default=0, help="GPU index; -1 for CPU")
    p.add_argument("--skip-dl", action="store_true", help="Only run classical ML")
    p.add_argument(
        "--save-models",
        action="store_true",
        default=True,
        help="Save per-fold classical ML checkpoints (default: on)",
    )
    p.add_argument("--no-save-models", action="store_true", help="Disable model checkpoint saving")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: outputs/outputs_toxin_filter_thr{threshold}_{feature_mode})",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = str(
            TIGER_ROOT / "outputs" / f"outputs_toxin_filter_thr{int(args.threshold)}_{args.feature_mode}"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    save_models = bool(args.save_models) and (not args.no_save_models)
    if save_models:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    df = load_toxicity_table(
        args.csv,
        json_dir=args.json_dir,
        threshold=args.threshold,
        min_len=args.min_len,
        max_len=args.max_len,
        source=args.source if args.csv is None else "csv",
    )
    balance = summarize_label_balance(df)
    print("=== Dataset ===")
    print(json.dumps(balance, indent=2))
    print(f"feature_mode: {args.feature_mode}")
    df.to_csv(out_dir / "toxicity_labeled_dataset.csv", index=False)
    (out_dir / "label_filter_stats.json").write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "source": args.source if args.csv is None else "csv",
                "feature_mode": args.feature_mode,
                "balance": balance,
            },
            indent=2,
        )
    )

    sequences, labels, X_tab, global_raw = prepare_feature_matrices(df, feature_mode=args.feature_mode)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    import torch

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"X_tab shape: {X_tab.shape}")

    fold_rows: list[dict] = []

    for fold_i, (tr, va) in enumerate(skf.split(X_tab, labels), start=1):
        print(f"\n======== Fold {fold_i}/{args.n_splits} ======== "
              f"train={len(tr)} val={len(va)} "
              f"val_pos={int(labels[va].sum())} val_neg={int((labels[va]==0).sum())}")

        # Classical ML: z-score tabular features on train fold only
        stats = fit_zscore(X_tab[tr])
        X_tr = apply_zscore(X_tab[tr], stats)
        X_va = apply_zscore(X_tab[va], stats)
        ml_models = build_ml_models(
            seed=args.seed,
            scale_pos_weight=_imbalance_ratio(labels[tr]),
        )
        ml_metrics, fitted = evaluate_ml_fold(
            ml_models, X_tr, labels[tr], X_va, labels[va], return_fitted=True
        )
        for name, metrics in ml_metrics.items():
            fold_rows.append(metrics_to_row("cv", name, fold_i, metrics))
            print(
                f"  [ML] {name:22s} Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"MCC={metrics['mcc']:.4f} AUC-ROC={metrics['auc_roc']:.4f} "
                f"AUC-PR={metrics['auc_pr']:.4f}"
            )
            if save_models:
                payload = {
                    "model_name": name,
                    "fold": fold_i,
                    "feature_mode": args.feature_mode,
                    "threshold_label": args.threshold,
                    "decision_threshold": fitted[name]["threshold"],
                    "zscore_stats": {k: v.tolist() for k, v in stats.items()},
                    "train_idx": tr.tolist(),
                    "val_idx": va.tolist(),
                    "metrics": metrics,
                    "model": fitted[name]["model"],
                }
                joblib.dump(payload, ckpt_dir / f"{name}_fold{fold_i}.joblib")

        if args.skip_dl:
            continue

        # DL always uses sequence encoding + 10-D global props (independent of ML feature_mode)
        common = dict(
            device=device,
            max_len=args.max_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            seed=args.seed + fold_i,
        )

        fusion_m = train_fusion_fold(sequences, global_raw, labels, tr, va, **common)
        fold_rows.append(metrics_to_row("cv", "fusion_seq_glob", fold_i, fusion_m))
        print(
            f"  [DL] fusion_seq_glob       Acc={fusion_m['accuracy']:.4f} F1={fusion_m['f1']:.4f} "
            f"MCC={fusion_m['mcc']:.4f} AUC-ROC={fusion_m['auc_roc']:.4f} AUC-PR={fusion_m['auc_pr']:.4f}"
        )

        tiger_m = train_tiger_fold(sequences, global_raw, labels, tr, va, **common)
        fold_rows.append(metrics_to_row("cv", "tiger_seq_glob", fold_i, tiger_m))
        print(
            f"  [DL] tiger_seq_glob        Acc={tiger_m['accuracy']:.4f} F1={tiger_m['f1']:.4f} "
            f"MCC={tiger_m['mcc']:.4f} AUC-ROC={tiger_m['auc_roc']:.4f} AUC-PR={tiger_m['auc_pr']:.4f}"
        )

        metric_m = train_metric_fold(sequences, global_raw, labels, tr, va, **common)
        fold_rows.append(metrics_to_row("cv", "metric_learning", fold_i, metric_m))
        print(
            f"  [DL] metric_learning       Acc={metric_m['accuracy']:.4f} F1={metric_m['f1']:.4f} "
            f"MCC={metric_m['mcc']:.4f} AUC-ROC={metric_m['auc_roc']:.4f} AUC-PR={metric_m['auc_pr']:.4f}"
        )

    cv_df = pd.DataFrame(fold_rows)
    summary_df = summarize_cv(fold_rows)
    pretty = format_metrics_table(summary_df)

    cv_df.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    pretty.to_csv(out_dir / "summary_metrics_pretty.csv", index=False)

    # mean±std table
    means = summary_df[summary_df["split"] == "cv_mean"].set_index("model")
    stds = summary_df[summary_df["split"] == "cv_std"].set_index("model")
    cols = ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc", "auc_pr"]
    rows = []
    for model in means.index:
        row = {"model": model}
        for c in cols:
            row[c] = f"{means.loc[model, c]:.4f}±{stds.loc[model, c]:.4f}"
        rows.append(row)
    mean_std_df = pd.DataFrame(rows)
    mean_std_df.to_csv(out_dir / "summary_mean_std_pretty.csv", index=False)

    meta = {
        "csv": args.csv,
        "threshold": args.threshold,
        "feature_mode": args.feature_mode,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "device": str(device),
        "balance": balance,
        "x_tab_dim": int(X_tab.shape[1]),
        "checkpoints_dir": str(ckpt_dir) if save_models else None,
        "models": sorted(cv_df["model"].unique().tolist()),
        "note": (
            "Inequality-aware binary labels from DBAASP JSON: keep exact values and "
            "determinable inequalities only; ambiguous censored values are excluded. "
            f"Classical ML feature_mode={args.feature_mode}. "
            "Per-fold models saved under checkpoints/ when enabled."
        ),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== CV mean ± std ===")
    print(mean_std_df.to_string(index=False))
    if save_models:
        print(f"\nSaved model checkpoints to {ckpt_dir}")
    print(f"Wrote metrics to {out_dir}")


if __name__ == "__main__":
    main()
