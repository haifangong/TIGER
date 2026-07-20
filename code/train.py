"""GroupKFold training and final retrain for pair / single models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

from .dataloader import (
    PairDataset,
    SingleDataset,
    build_training_pairs,
    load_or_build_cache,
    pair_collate,
    precompute_neighbors,
    single_collate,
)
from .evaluation import eval_pair_delta, save_eval_outputs
from .models import build_model, build_optimizer, build_scheduler
from .utils.config import Config
from .utils.metrics import apply_calibrator, fit_calibrator, is_better_selection, regression_metrics
from .utils.scaling import (
    apply_global_f_stats,
    apply_node_x_stats,
    cache_raw_global_f,
    cache_raw_node_x,
    fit_global_f_stats,
    fit_node_x_stats,
)
from .utils.seed import set_seed


def _apply_fold_zscore(cfg: Config, graphs, tr_idx, va_idx, out_dir: Path, fold: int):
    idx_all = np.concatenate([np.asarray(tr_idx), np.asarray(va_idx)])
    global_stats = None
    node_stats = None
    if str(cfg.global_feature_scaling).lower() == "zscore":
        cache_raw_global_f(graphs)
        global_stats = fit_global_f_stats(graphs, tr_idx)
        apply_global_f_stats(graphs, global_stats, idx_all)
        (out_dir / "intermediate" / f"global_feature_zscore_fold{fold}.json").write_text(
            json.dumps(global_stats, indent=2)
        )
    if str(getattr(cfg, "node_feature_scaling", "none")).lower() == "zscore":
        cache_raw_node_x(graphs)
        node_stats = fit_node_x_stats(graphs, tr_idx)
        apply_node_x_stats(graphs, node_stats, idx_all)
        (out_dir / "intermediate" / f"node_feature_zscore_fold{fold}.json").write_text(
            json.dumps(node_stats, indent=2)
        )
    return {"global": global_stats, "node": node_stats}


def _apply_final_zscore(cfg: Config, graphs, test_graphs, out_dir: Path):
    global_stats = None
    node_stats = None
    if str(cfg.global_feature_scaling).lower() == "zscore":
        cache_raw_global_f(graphs)
        if test_graphs:
            cache_raw_global_f(test_graphs)
        global_stats = fit_global_f_stats(graphs)
        apply_global_f_stats(graphs, global_stats)
        if test_graphs:
            apply_global_f_stats(test_graphs, global_stats)
        (out_dir / "intermediate" / "global_feature_zscore_final.json").write_text(
            json.dumps(global_stats, indent=2)
        )
    if str(getattr(cfg, "node_feature_scaling", "none")).lower() == "zscore":
        cache_raw_node_x(graphs)
        if test_graphs:
            cache_raw_node_x(test_graphs)
        node_stats = fit_node_x_stats(graphs)
        apply_node_x_stats(graphs, node_stats)
        if test_graphs:
            apply_node_x_stats(test_graphs, node_stats)
        (out_dir / "intermediate" / "node_feature_zscore_final.json").write_text(
            json.dumps(node_stats, indent=2)
        )
    return {"global": global_stats, "node": node_stats}


def train(cfg: Config, root: Path | None = None) -> dict:
    """Run full GroupKFold CV + final retrain. Returns summary dict."""
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    if root is not None and not out_dir.is_absolute():
        out_dir = root / out_dir
    for sub in ["checkpoints", "logs", "results", "intermediate"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    started = time.time()
    payload = load_or_build_cache(cfg, root)
    graphs, rows = payload["graphs"], payload["rows"]
    test_graphs, test_rows = payload.get("test_graphs", []), payload.get("test_rows", pd.DataFrame())
    labels = rows["label"].to_numpy(float)
    device = torch.device(cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu")

    if str(cfg.global_feature_scaling).lower() == "zscore":
        cache_raw_global_f(graphs)
        if test_graphs:
            cache_raw_global_f(test_graphs)
    if str(getattr(cfg, "node_feature_scaling", "none")).lower() == "zscore":
        cache_raw_node_x(graphs)
        if test_graphs:
            cache_raw_node_x(test_graphs)

    splits = list(GroupKFold(cfg.folds).split(np.arange(len(rows)), groups=rows.sequence))
    frames, fold_metrics = [], []

    for fold, (tr, va) in enumerate(splits, 1):
        _apply_fold_zscore(cfg, graphs, tr, va, out_dir, fold)
        tr_rows = rows.iloc[tr].reset_index(drop=True)
        va_rows = rows.iloc[va].reset_index(drop=True)
        tr_graphs = [graphs[i] for i in tr]
        va_graphs = [graphs[i] for i in va]
        model = build_model(cfg).to(device)
        opt = build_optimizer(model, cfg)
        sched = build_scheduler(opt, cfg, cfg.graph_epochs)

        if cfg.model_kind == "single":
            loader = torch.utils.data.DataLoader(
                SingleDataset(tr_graphs, labels[tr], range(len(tr))),
                batch_size=256,
                shuffle=True,
                collate_fn=single_collate,
            )
        else:
            pairs = build_training_pairs(tr_rows, labels[tr], cfg, cfg.seed + fold)
            loader = torch.utils.data.DataLoader(
                PairDataset(tr_graphs, labels[tr], pairs),
                batch_size=cfg.pair_batch_size,
                shuffle=True,
                collate_fn=pair_collate,
            )
        neighbors = precompute_neighbors(tr_rows, va_rows, cfg)
        # Best checkpoint by selection_score = log2MAE + RSE - PCC - KCC (lower better)
        best_score, best_epoch, stale, history, last_epoch = float("inf"), 0, 0, [], 0
        best_met: dict = {}

        for epoch in range(1, cfg.graph_epochs + 1):
            model.train()
            losses = []
            for batch in loader:
                opt.zero_grad()
                if cfg.model_kind == "single":
                    g, y = batch
                    g, y = g.to(device), y.to(device)
                    loss = F.mse_loss(model.score(g), y)
                else:
                    a, b, y, t = batch
                    a, b, y = a.to(device), b.to(device), y.to(device)
                    loss = F.mse_loss(model(a, b), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            if sched is not None:
                sched.step()
            last_epoch = epoch
            if epoch == 1 or epoch % 2 == 0:
                pred, met = eval_pair_delta(
                    model, tr_graphs, tr_rows, labels[tr], va_graphs, va_rows, labels[va], cfg, device, neighbors=neighbors
                )
                improved = is_better_selection(met, best_score, cfg.min_delta)
                if improved:
                    best_score = float(met["selection_score"])
                    best_epoch = epoch
                    best_met = {k: met[k] for k in ("log2MAE", "RSE", "PCC", "KCC", "selection_score") if k in met}
                    stale = 0
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "cfg": cfg.to_dict(),
                            "epoch": epoch,
                            "fold": fold,
                            "checkpoint_type": "best_val",
                            "selection_metrics": best_met,
                        },
                        out_dir / "checkpoints" / f"fold{fold}_best.pt",
                    )
                else:
                    stale += 2
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses)),
                        "best_epoch": best_epoch,
                        "best_selection_score": best_score if np.isfinite(best_score) else None,
                        "lr": float(opt.param_groups[0]["lr"]),
                        **met,
                    }
                )
                pd.DataFrame(history).to_csv(out_dir / "logs" / f"fold{fold}_history.csv", index=False)
                print(
                    f"[{cfg.name}/fold{fold}] epoch={epoch} "
                    f"log2MAE={met.get('log2MAE', float('nan')):.5f} "
                    f"RSE={met.get('RSE', float('nan')):.5f} "
                    f"PCC={met.get('PCC', float('nan')):.5f} "
                    f"KCC={met.get('KCC', float('nan')):.5f} "
                    f"score={met.get('selection_score', float('nan')):.5f} "
                    f"best_score={best_score:.5f} lr={opt.param_groups[0]['lr']:.6g}",
                    flush=True,
                )
                if stale >= cfg.early_stop_patience:
                    break

        # Always keep the last-epoch weights for this fold.
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": cfg.to_dict(),
                "epoch": last_epoch,
                "fold": fold,
                "checkpoint_type": "last_epoch",
            },
            out_dir / "checkpoints" / f"fold{fold}_last.pt",
        )
        best_path = out_dir / "checkpoints" / f"fold{fold}_best.pt"
        if not best_path.exists():
            # Fallback: no val improvement recorded → treat last as best.
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.to_dict(),
                    "epoch": last_epoch,
                    "fold": fold,
                    "checkpoint_type": "best_val_fallback_last",
                },
                best_path,
            )
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck["model"])
        pred, met = eval_pair_delta(
            model, tr_graphs, tr_rows, labels[tr], va_graphs, va_rows, labels[va], cfg, device, neighbors=neighbors
        )
        pred.insert(0, "fold", fold)
        frames.append(pred)
        fold_metrics.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "last_epoch": last_epoch,
                "best_ckpt": f"fold{fold}_best.pt",
                "last_ckpt": f"fold{fold}_last.pt",
                **met,
            }
        )

    cv_raw = pd.concat(frames, ignore_index=True)
    calibrator = fit_calibrator(cv_raw)
    cv = apply_calibrator(cv_raw, calibrator)

    # No full-data final retrain: CV OOF from fold*_best.pt is the primary report.
    external_metrics: dict = {
        "skipped": True,
        "reason": "no_final_retrain; use fold{k}_best.pt / fold{k}_last.pt for inference",
    }
    if cfg.eval_external and len(test_rows):
        external_metrics = {
            "skipped": True,
            "reason": "eval_external requires choosing a fold checkpoint; final retrain disabled",
        }

    summary = {
        "experiment": cfg.name,
        "model_kind": cfg.model_kind,
        "aa_encoding": "sequential_1_to_20_alphabetical",
        "cv": regression_metrics(cv.y_true_delta_log2_anchor_minus_query, cv.y_pred_delta_log2_anchor_minus_query),
        "external": external_metrics,
        "folds": fold_metrics,
        "calibrator": calibrator,
        "checkpoints": {
            "per_fold": ["fold{k}_best.pt", "fold{k}_last.pt"],
            "final_retrain": False,
        },
        "runtime_seconds": round(time.time() - started, 2),
        "config": cfg.to_dict(),
    }
    save_eval_outputs(out_dir, cv, fold_metrics, calibrator, summary)
    print(json.dumps({"cv": summary["cv"], "external": summary["external"], "checkpoints": summary["checkpoints"]}, indent=2))
    return summary
