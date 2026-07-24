#!/usr/bin/env python3
"""Run 5-fold CV activity filter (sequence + global features, DL only).

Label (inequality-aware MIC from DBAASP targetActivities):
  inactive / filter-out (1) if MIC is determinably >= threshold (default 128 ug/mL)
  active / keep         (0) if MIC is determinably <  threshold

Uses the same sequence + 10-D global physicochemical features as toxin_filter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
TIGER_ROOT = CODE_ROOT.parent
if str(TIGER_ROOT) not in sys.path:
    sys.path.insert(0, str(TIGER_ROOT))

from code.activity_filter.data import load_activity_table, summarize_label_balance  # noqa: E402
from code.activity_filter.train_eval import (  # noqa: E402
    prepare_feature_matrices,
    save_fold_checkpoint,
    train_fusion_fold,
    train_metric_fold,
    train_tiger_fold,
)
from code.toxin_filter.metrics import (  # noqa: E402
    format_metrics_table,
    metrics_to_row,
    summarize_cv,
)


def parse_args():
    p = argparse.ArgumentParser(description="Activity filter 5-fold DL CV (MIC>=threshold)")
    p.add_argument("--json-dir", default=None, help="DBAASP JSON export directory")
    p.add_argument("--csv", default=None, help="Optional numeric MIC CSV fallback")
    p.add_argument(
        "--source",
        choices=["json", "csv"],
        default="json",
        help="Label source: inequality-aware JSON (default) or numeric CSV",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=128.0,
        help="MIC inactivity boundary (ug/mL); label=1 if MIC >= threshold",
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
    p.add_argument(
        "--models",
        nargs="+",
        default=["fusion_seq_glob", "tiger_seq_glob", "metric_learning"],
        choices=["fusion_seq_glob", "tiger_seq_glob", "metric_learning"],
        help="Which DL models to train",
    )
    p.add_argument("--save-models", action="store_true", default=True)
    p.add_argument("--no-save-models", action="store_true")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: outputs/outputs_activity_filter_mic{threshold})",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = str(TIGER_ROOT / "outputs" / f"outputs_activity_filter_mic{int(args.threshold)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    save_models = bool(args.save_models) and (not args.no_save_models)
    if save_models:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    source = "csv" if args.csv is not None else args.source
    df = load_activity_table(
        json_dir=args.json_dir,
        csv_path=args.csv,
        threshold=args.threshold,
        min_len=args.min_len,
        max_len=args.max_len,
        source=source,
    )
    balance = summarize_label_balance(df)
    print("=== Dataset ===")
    print(json.dumps(balance, indent=2))
    df.to_csv(out_dir / "activity_labeled_dataset.csv", index=False)
    (out_dir / "label_filter_stats.json").write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "source": source,
                "balance": balance,
                "json_dir": df.attrs.get("json_dir"),
                "csv_path": df.attrs.get("csv_path"),
            },
            indent=2,
        )
    )

    # Same features as toxin_filter DL: sequence encoding + 10-D global props
    sequences, labels, _X_tab, global_raw = prepare_feature_matrices(df, feature_mode="both")
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    import torch

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Models: {args.models}")

    fold_rows: list[dict] = []
    trainers = {
        "fusion_seq_glob": train_fusion_fold,
        "tiger_seq_glob": train_tiger_fold,
        "metric_learning": train_metric_fold,
    }

    for fold_i, (tr, va) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(
            f"\n======== Fold {fold_i}/{args.n_splits} ======== "
            f"train={len(tr)} val={len(va)} "
            f"val_inactive={int(labels[va].sum())} val_active={int((labels[va] == 0).sum())}"
        )
        common = dict(
            device=device,
            max_len=args.max_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            seed=args.seed + fold_i,
        )
        for name in args.models:
            metrics, payload = trainers[name](
                sequences, global_raw, labels, tr, va, **common
            )
            fold_rows.append(metrics_to_row("cv", name, fold_i, metrics))
            print(
                f"  [DL] {name:22s} Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"MCC={metrics['mcc']:.4f} AUC-ROC={metrics['auc_roc']:.4f} "
                f"AUC-PR={metrics['auc_pr']:.4f}"
            )
            if save_models:
                path = save_fold_checkpoint(ckpt_dir, fold_i, payload)
                print(f"       saved {path.name}")

    cv_df = pd.DataFrame(fold_rows)
    summary_df = summarize_cv(fold_rows)
    pretty = format_metrics_table(summary_df)

    cv_df.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    pretty.to_csv(out_dir / "summary_metrics_pretty.csv", index=False)

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
        "task": "activity_filter",
        "threshold": args.threshold,
        "label": "1=inactive(MIC>=T), 0=active(MIC<T)",
        "source": source,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "device": str(device),
        "balance": balance,
        "models": list(args.models),
        "features": "sequence encoding + 10-D global physicochemical props (same as toxin_filter DL)",
        "checkpoints_dir": str(ckpt_dir) if save_models else None,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== CV mean ± std ===")
    print(mean_std_df.to_string(index=False))
    if save_models:
        print(f"\nSaved model checkpoints to {ckpt_dir}")
    print(f"Wrote metrics to {out_dir}")


if __name__ == "__main__":
    main()
