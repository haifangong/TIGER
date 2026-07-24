#!/usr/bin/env python3
"""TIGER modular CLI.

Examples
--------
# Train fusion-attention pair model (default config)
python -m code.main train --config code/configs/default_fusion_attention.json

# Evaluate / resume summary already written by train
python -m code.main evaluate --config outputs/tiger_code_run/config.json

# Infer on LL37 test with a trained fold checkpoint
python -m code.main infer --config outputs/tiger_code_run/config.json \\
    --checkpoint outputs/tiger_code_run/checkpoints/fold1_best.pt

# APEXGO high-span family eval (log10MAE / RSE / PCC / KCC)
python -m code.main evaluate-apexgo --config outputs/tiger_code_run/config.json \\
    --checkpoint outputs/tiger_code_run/checkpoints/fold1_best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow `python -m code.main` from TIGER root and `python code/main.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.utils.config import load_config  # noqa: E402
from code.utils.constants import AA_TO_IDX  # noqa: E402


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/default_fusion_attention.json")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--name", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None, dest="weight_decay")
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default=None, dest="lr_scheduler")
    parser.add_argument("--seq-encoding", choices=["integer", "embedding", "onehot"], default=None, dest="seq_encoding")
    parser.add_argument(
        "--global-feature-scaling",
        choices=["legacy_x10", "zscore"],
        default=None,
        dest="global_feature_scaling",
    )
    parser.add_argument(
        "--node-feature-scaling",
        choices=["none", "zscore"],
        default=None,
        dest="node_feature_scaling",
    )
    parser.add_argument(
        "--include-node-coords",
        dest="include_node_coords",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--fusion", default=None)
    parser.add_argument(
        "--fusion-attn-mode",
        choices=["self_gsh", "self_sgh", "self_hgs", "cross_qg", "cross_qs", "cross_qh"],
        default=None,
        dest="fusion_attn_mode",
    )
    parser.add_argument("--feature-modalities", default=None, dest="feature_modalities")
    parser.add_argument("--structure-features", default=None, dest="structure_features")
    parser.add_argument("--gnn-type", default=None, dest="gnn_type")
    parser.add_argument("--pair-interaction", default=None, dest="pair_interaction")
    parser.add_argument("--model-kind", choices=["pair", "single"], default=None, dest="model_kind")
    parser.add_argument("--pair-batch-size", type=int, default=None, dest="pair_batch_size")
    parser.add_argument("--similarity-threshold", type=float, default=None, dest="similarity_threshold")
    parser.add_argument("--delta-bin-width", type=float, default=None, dest="delta_bin_width")
    parser.add_argument("--pair-balance-num", type=int, default=None, dest="pair_balance_num")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-csv", type=Path, default=None, dest="train_csv")
    parser.add_argument("--test-csv", type=Path, default=None, dest="test_csv")
    parser.add_argument("--train-pdb-dir", type=Path, default=None, dest="train_pdb_dir")
    parser.add_argument("--test-pdb-dir", type=Path, default=None, dest="test_pdb_dir")
    parser.add_argument("--shared-cache-dir", type=Path, default=None, dest="shared_cache_dir")
    parser.add_argument("--target-col", default=None, dest="target_col")
    parser.add_argument("--csv-encoding", default=None, dest="csv_encoding")
    parser.add_argument(
        "--use-signed-sampling",
        dest="use_signed_sampling",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)


def _cfg_from_args(args):
    overrides = {
        "out_dir": str(args.out_dir) if args.out_dir else None,
        "name": args.name,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_scheduler": args.lr_scheduler,
        "seq_encoding": args.seq_encoding,
        "global_feature_scaling": args.global_feature_scaling,
        "node_feature_scaling": args.node_feature_scaling,
        "include_node_coords": args.include_node_coords,
        "fusion": args.fusion,
        "fusion_attn_mode": args.fusion_attn_mode,
        "feature_modalities": args.feature_modalities,
        "structure_features": args.structure_features,
        "gnn_type": args.gnn_type,
        "pair_interaction": args.pair_interaction,
        "model_kind": args.model_kind,
        "pair_batch_size": args.pair_batch_size,
        "similarity_threshold": args.similarity_threshold,
        "delta_bin_width": args.delta_bin_width,
        "pair_balance_num": args.pair_balance_num,
        "use_signed_sampling": args.use_signed_sampling,
        "seed": args.seed,
        "train_csv": str(args.train_csv) if args.train_csv else None,
        "test_csv": str(args.test_csv) if args.test_csv else None,
        "train_pdb_dir": str(args.train_pdb_dir) if args.train_pdb_dir else None,
        "test_pdb_dir": str(args.test_pdb_dir) if args.test_pdb_dir else None,
        "shared_cache_dir": str(args.shared_cache_dir) if args.shared_cache_dir else None,
        "target_col": args.target_col,
        "csv_encoding": args.csv_encoding,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "device": "cuda:0",
        "smoke": bool(args.smoke) or None,
    }
    return load_config(args.config, overrides={k: v for k, v in overrides.items() if v is not None})


def cmd_train(args) -> None:
    from code.train import train

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = _cfg_from_args(args)
    print(f"[train] aa_encoding={AA_TO_IDX} (A=1..Y=20)", flush=True)
    print(
        f"[train] seq_encoding={cfg.seq_encoding} global_scale={cfg.global_feature_scaling} "
        f"node_scale={cfg.node_feature_scaling} coords={cfg.include_node_coords} "
        f"fusion={cfg.fusion} attn={cfg.fusion_attn_mode} mod={cfg.feature_modalities} "
        f"struct={cfg.structure_features} sim={cfg.similarity_threshold} "
        f"bin={cfg.delta_bin_width} bal={cfg.pair_balance_num} signed={cfg.use_signed_sampling} "
        f"seed={cfg.seed} target={cfg.target_col} train_csv={cfg.train_csv}",
        flush=True,
    )
    train(cfg, root=ROOT)


def cmd_evaluate(args) -> None:
    """Print / refresh metrics from an existing run directory (config.out_dir)."""
    cfg = _cfg_from_args(args)
    summary_path = Path(cfg.out_dir) if Path(cfg.out_dir).is_absolute() else ROOT / cfg.out_dir
    summary_path = summary_path / "results" / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"No summary at {summary_path}; run train first")
    print(summary_path.read_text())


def cmd_infer(args) -> None:
    from code.infer import infer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = _cfg_from_args(args)
    if args.checkpoint:
        cfg.checkpoint = str(args.checkpoint)
    infer(cfg, root=ROOT)


def cmd_evaluate_apexgo(args) -> None:
    """Evaluate checkpoint on large-span APEXGO families (log10MAE/RSE/PCC/KCC)."""
    from code.evaluation import APEXGO_HIGH_SPAN_FAMILIES, evaluate_apexgo_high_span

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = _cfg_from_args(args)
    if args.checkpoint:
        cfg.checkpoint = str(args.checkpoint)
    families = args.families if args.families else list(APEXGO_HIGH_SPAN_FAMILIES)
    evaluate_apexgo_high_span(
        cfg,
        checkpoint=cfg.checkpoint,
        root=ROOT,
        families=families,
        bidirectional=bool(args.bidirectional),
        apply_saved_calibrator=not bool(args.no_calibrator),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TIGER modular MIC pair-delta toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="GroupKFold train + final retrain")
    _add_common(p_train)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Show saved CV/external summary")
    _add_common(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = sub.add_parser("infer", help="Score external/LL37 pairs with a checkpoint")
    _add_common(p_infer)
    p_infer.set_defaults(func=cmd_infer)

    p_apex = sub.add_parser(
        "evaluate-apexgo",
        help="Evaluate on high-span APEXGO families (log10MAE/RSE/PCC/KCC)",
    )
    _add_common(p_apex)
    p_apex.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Optional subset of APEXGO families (default: 5 largest-span templates)",
    )
    p_apex.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use antisymmetric 0.5*(f(a,q)-f(q,a)) predictions",
    )
    p_apex.add_argument(
        "--no-calibrator",
        action="store_true",
        help="Skip applying results/calibrator.json next to the checkpoint",
    )
    p_apex.set_defaults(func=cmd_evaluate_apexgo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
