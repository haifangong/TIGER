"""DL fold trainers for activity filter (reuse toxin_filter models/features)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from code.toxin_filter.features import ToxinDataset, apply_zscore, fit_zscore
from code.toxin_filter.metrics import classification_metrics
from code.toxin_filter.models_dl import (
    FusionSeqGlob,
    MetricLearningNet,
    TIGERSeqGlob,
    supervised_contrastive_loss,
)
from code.toxin_filter.train_eval import (
    _best_threshold,
    _class_weights,
    _make_loader,
    _predict_probs,
    prepare_feature_matrices,
)


def _fit_dl_loop_with_checkpoint(
    model: nn.Module,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    *,
    device: torch.device,
    kind: str,
    epochs: int,
    patience: int,
    lr: float,
    y_train: np.ndarray,
    loss_fn,
    zscore_stats: dict,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    best_state, best_auc, bad = None, -1.0, 0
    for _ in range(epochs):
        model.train()
        for seq, globf, y, _ in tr_loader:
            seq, globf, y = seq.to(device), globf.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model, seq, globf, y)
            loss.backward()
            opt.step()
        y_true, y_prob = _predict_probs(model, va_loader, device, kind)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = classification_metrics(y_true, y_pred, y_prob)
        auc = metrics["auc_roc"] if np.isfinite(metrics["auc_roc"]) else metrics["f1"]
        if auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    tr_eval = DataLoader(tr_loader.dataset, batch_size=tr_loader.batch_size, shuffle=False)
    _, tr_prob = _predict_probs(model, tr_eval, device, kind)
    thr = _best_threshold(y_train, tr_prob)
    y_true, y_prob = _predict_probs(model, va_loader, device, kind)
    y_pred = (y_prob >= thr).astype(int)
    metrics = classification_metrics(y_true, y_pred, y_prob)
    payload = {
        "state_dict": copy.deepcopy(model.state_dict()),
        "kind": kind,
        "decision_threshold": float(thr),
        "zscore_stats": {k: np.asarray(v) for k, v in zscore_stats.items()},
        "metrics": metrics,
    }
    return metrics, payload


def train_fusion_fold(
    sequences,
    global_raw,
    labels,
    train_idx,
    val_idx,
    *,
    device,
    max_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
):
    torch.manual_seed(seed)
    g_train = global_raw[train_idx]
    stats = fit_zscore(g_train)
    g_all = apply_zscore(global_raw, stats).astype(np.float32)

    tr_ds = ToxinDataset(
        [sequences[i] for i in train_idx],
        g_all[train_idx],
        labels[train_idx],
        max_len=max_len,
        seq_style="poap",
    )
    va_ds = ToxinDataset(
        [sequences[i] for i in val_idx],
        g_all[val_idx],
        labels[val_idx],
        max_len=max_len,
        seq_style="poap",
    )
    tr_loader = _make_loader(tr_ds, labels[train_idx], batch_size, shuffle=True, balance=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model = FusionSeqGlob(q_encoder="gru", glob_dim=g_all.shape[1]).to(device)
    pos_weight = _class_weights(labels[train_idx]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def loss_fn(m, seq, globf, y):
        return criterion(m(seq, globf), y)

    metrics, payload = _fit_dl_loop_with_checkpoint(
        model,
        tr_loader,
        va_loader,
        device=device,
        kind="fusion",
        epochs=epochs,
        patience=patience,
        lr=lr,
        y_train=labels[train_idx],
        loss_fn=loss_fn,
        zscore_stats=stats,
    )
    payload.update({"model_name": "fusion_seq_glob", "max_len": max_len, "glob_dim": int(g_all.shape[1])})
    return metrics, payload


def train_tiger_fold(
    sequences,
    global_raw,
    labels,
    train_idx,
    val_idx,
    *,
    device,
    max_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
):
    torch.manual_seed(seed)
    g_train = global_raw[train_idx]
    stats = fit_zscore(g_train)
    g_all = apply_zscore(global_raw, stats).astype(np.float32)

    tr_ds = ToxinDataset(
        [sequences[i] for i in train_idx],
        g_all[train_idx],
        labels[train_idx],
        max_len=max_len,
        seq_style="integer",
    )
    va_ds = ToxinDataset(
        [sequences[i] for i in val_idx],
        g_all[val_idx],
        labels[val_idx],
        max_len=max_len,
        seq_style="integer",
    )
    tr_loader = _make_loader(tr_ds, labels[train_idx], batch_size, shuffle=True, balance=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model = TIGERSeqGlob(max_len=max_len, emb_dim=128, glob_dim=g_all.shape[1]).to(device)
    pos_weight = _class_weights(labels[train_idx]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def loss_fn(m, seq, globf, y):
        return criterion(m(seq, globf), y)

    metrics, payload = _fit_dl_loop_with_checkpoint(
        model,
        tr_loader,
        va_loader,
        device=device,
        kind="tiger",
        epochs=epochs,
        patience=patience,
        lr=lr,
        y_train=labels[train_idx],
        loss_fn=loss_fn,
        zscore_stats=stats,
    )
    payload.update({"model_name": "tiger_seq_glob", "max_len": max_len, "glob_dim": int(g_all.shape[1])})
    return metrics, payload


def train_metric_fold(
    sequences,
    global_raw,
    labels,
    train_idx,
    val_idx,
    *,
    device,
    max_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
    lambda_con: float = 0.5,
):
    torch.manual_seed(seed)
    g_train = global_raw[train_idx]
    stats = fit_zscore(g_train)
    g_all = apply_zscore(global_raw, stats).astype(np.float32)

    tr_ds = ToxinDataset(
        [sequences[i] for i in train_idx],
        g_all[train_idx],
        labels[train_idx],
        max_len=max_len,
        seq_style="integer",
    )
    va_ds = ToxinDataset(
        [sequences[i] for i in val_idx],
        g_all[val_idx],
        labels[val_idx],
        max_len=max_len,
        seq_style="integer",
    )
    tr_loader = _make_loader(tr_ds, labels[train_idx], batch_size, shuffle=True, balance=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model = MetricLearningNet(max_len=max_len, emb_dim=128, glob_dim=g_all.shape[1]).to(device)
    pos_weight = _class_weights(labels[train_idx]).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def loss_fn(m, seq, globf, y):
        logits, z = m(seq, globf)
        return bce(logits, y) + lambda_con * supervised_contrastive_loss(z, y.view(-1))

    metrics, payload = _fit_dl_loop_with_checkpoint(
        model,
        tr_loader,
        va_loader,
        device=device,
        kind="metric",
        epochs=epochs,
        patience=patience,
        lr=lr,
        y_train=labels[train_idx],
        loss_fn=loss_fn,
        zscore_stats=stats,
    )
    payload.update({"model_name": "metric_learning", "max_len": max_len, "glob_dim": int(g_all.shape[1])})
    return metrics, payload


def save_fold_checkpoint(ckpt_dir: Path, fold: int, payload: dict) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = payload["model_name"]
    out = ckpt_dir / f"{name}_fold{fold}.pt"
    # numpy in zscore_stats → list for portability
    zstats = {k: np.asarray(v).tolist() for k, v in payload["zscore_stats"].items()}
    torch.save(
        {
            "model_name": name,
            "fold": fold,
            "kind": payload["kind"],
            "state_dict": payload["state_dict"],
            "decision_threshold": payload["decision_threshold"],
            "zscore_stats": zstats,
            "max_len": payload["max_len"],
            "glob_dim": payload["glob_dim"],
            "metrics": payload["metrics"],
        },
        out,
    )
    return out
