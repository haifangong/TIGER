"""5-fold CV training/evaluation for traditional ML, DL, and metric learning."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, WeightedRandomSampler

from .features import (
    ToxinDataset,
    apply_zscore,
    build_tabular_features,
    calculate_global_properties,
    fit_zscore,
)
from .metrics import classification_metrics
from .models_dl import FusionSeqGlob, MetricLearningNet, TIGERSeqGlob, supervised_contrastive_loss


def _imbalance_ratio(y: np.ndarray) -> float:
    n_pos = max(int((np.asarray(y) == 1).sum()), 1)
    n_neg = max(int((np.asarray(y) == 0).sum()), 1)
    return float(n_neg) / float(n_pos)


def build_ml_models(seed: int = 42, scale_pos_weight: float = 1.0) -> dict:
    """Classical baselines matching the manuscript legend order.

    CatBoost, LGBM, RF, SVM, GB, XGB, MLP, Adaboost, LR
    """
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "CatBoost": CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        ),
        "LGBM": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True,
        ),
        "RF": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "GB": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        ),
        "XGB": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        max_iter=500,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Adaboost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=2,
                class_weight="balanced",
                random_state=seed,
            ),
            n_estimators=300,
            learning_rate=0.05,
            random_state=seed,
        ),
        "LR": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def predict_labels_and_probs(model, X):
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # squash to (0,1) for AUCs
        prob = 1.0 / (1.0 + np.exp(-scores))
    else:
        prob = pred.astype(float)
    return pred.astype(int), np.asarray(prob, dtype=float)


def evaluate_ml_fold(
    models: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    return_fitted: bool = False,
):
    out = {}
    fitted = {}
    for name, model in models.items():
        clf = clone(model)
        X_fit, y_fit = X_train, y_train
        # Models without native class balancing: oversample minority class.
        if name in {"MLP", "Adaboost"}:
            rng = np.random.RandomState(42)
            pos = np.where(y_train == 1)[0]
            neg = np.where(y_train == 0)[0]
            if len(neg) and len(pos):
                neg_up = rng.choice(neg, size=len(pos), replace=True)
                idx = np.concatenate([pos, neg_up])
                rng.shuffle(idx)
                X_fit, y_fit = X_train[idx], y_train[idx]
        clf.fit(X_fit, y_fit)
        pred, prob = predict_labels_and_probs(clf, X_val)
        best_t = 0.5
        # Calibrate decision threshold on the original train distribution.
        if hasattr(clf, "predict_proba"):
            tr_prob = clf.predict_proba(X_train)[:, 1]
            best_f1 = -1.0
            for t in np.linspace(0.05, 0.95, 19):
                m = classification_metrics(y_train, (tr_prob >= t).astype(int), tr_prob)
                if m["f1"] > best_f1:
                    best_f1 = m["f1"]
                    best_t = float(t)
            pred = (prob >= best_t).astype(int)
        metrics = classification_metrics(y_val, pred, prob)
        out[name] = metrics
        if return_fitted:
            fitted[name] = {"model": clf, "threshold": best_t, "metrics": metrics}
    if return_fitted:
        return out, fitted
    return out


def _class_weights(y: np.ndarray) -> torch.Tensor:
    # pos_weight for BCEWithLogits: n_neg / n_pos
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def _make_loader(dataset: ToxinDataset, y: np.ndarray, batch_size: int, shuffle: bool, balance: bool):
    if shuffle and balance:
        class_count = np.bincount(y.astype(int), minlength=2).astype(float)
        class_count[class_count == 0] = 1.0
        weights = 1.0 / class_count[y.astype(int)]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict_probs(model: nn.Module, loader: DataLoader, device: torch.device, kind: str):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for seq, globf, y, _ in loader:
            seq = seq.to(device)
            globf = globf.to(device)
            if kind == "metric":
                logits, _ = model(seq, globf)
            else:
                logits = model(seq, globf)
            p = torch.sigmoid(logits).view(-1).cpu().numpy()
            probs.append(p)
            labels.append(y.view(-1).cpu().numpy())
    y_true = np.concatenate(labels).astype(int)
    y_prob = np.concatenate(probs).astype(float)
    return y_true, y_prob


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick probability threshold maximizing F1 on a reference split (train)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (y_prob >= t).astype(int)
        m = classification_metrics(y_true, pred, y_prob)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
    return best_t


def _predict_dl(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    kind: str,
    threshold: float = 0.5,
):
    y_true, y_prob = _predict_probs(model, loader, device, kind)
    y_pred = (y_prob >= threshold).astype(int)
    return classification_metrics(y_true, y_pred, y_prob)


def _fit_dl_loop(
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
        metrics = _predict_dl(model, va_loader, device, kind=kind, threshold=0.5)
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
    # Calibrate hard-decision threshold on the training split.
    tr_eval = DataLoader(tr_loader.dataset, batch_size=tr_loader.batch_size, shuffle=False)
    _, tr_prob = _predict_probs(model, tr_eval, device, kind)
    thr = _best_threshold(y_train, tr_prob)
    return _predict_dl(model, va_loader, device, kind=kind, threshold=thr)


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
) -> dict:
    torch.manual_seed(seed)
    g_train = global_raw[train_idx]
    stats = fit_zscore(g_train)
    g_all = apply_zscore(global_raw, stats).astype(np.float32)

    tr_ds = ToxinDataset(
        [sequences[i] for i in train_idx], g_all[train_idx], labels[train_idx], max_len=max_len, seq_style="poap"
    )
    va_ds = ToxinDataset(
        [sequences[i] for i in val_idx], g_all[val_idx], labels[val_idx], max_len=max_len, seq_style="poap"
    )
    tr_loader = _make_loader(tr_ds, labels[train_idx], batch_size, shuffle=True, balance=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model = FusionSeqGlob(q_encoder="gru", glob_dim=g_all.shape[1]).to(device)
    pos_weight = _class_weights(labels[train_idx]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def loss_fn(m, seq, globf, y):
        return criterion(m(seq, globf), y)

    return _fit_dl_loop(
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
    )


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
) -> dict:
    torch.manual_seed(seed)
    g_train = global_raw[train_idx]
    stats = fit_zscore(g_train)
    g_all = apply_zscore(global_raw, stats).astype(np.float32)

    tr_ds = ToxinDataset(
        [sequences[i] for i in train_idx], g_all[train_idx], labels[train_idx], max_len=max_len, seq_style="integer"
    )
    va_ds = ToxinDataset(
        [sequences[i] for i in val_idx], g_all[val_idx], labels[val_idx], max_len=max_len, seq_style="integer"
    )
    tr_loader = _make_loader(tr_ds, labels[train_idx], batch_size, shuffle=True, balance=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model = TIGERSeqGlob(max_len=max_len, emb_dim=128, glob_dim=g_all.shape[1]).to(device)
    pos_weight = _class_weights(labels[train_idx]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def loss_fn(m, seq, globf, y):
        return criterion(m(seq, globf), y)

    return _fit_dl_loop(
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
    )


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
) -> dict:
    """Joint supervised contrastive + BCE classification."""
    torch.manual_seed(seed)
    g_train = global_raw[train_idx]
    stats = fit_zscore(g_train)
    g_all = apply_zscore(global_raw, stats).astype(np.float32)

    tr_ds = ToxinDataset(
        [sequences[i] for i in train_idx], g_all[train_idx], labels[train_idx], max_len=max_len, seq_style="integer"
    )
    va_ds = ToxinDataset(
        [sequences[i] for i in val_idx], g_all[val_idx], labels[val_idx], max_len=max_len, seq_style="integer"
    )
    tr_loader = _make_loader(tr_ds, labels[train_idx], batch_size, shuffle=True, balance=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model = MetricLearningNet(max_len=max_len, emb_dim=128, glob_dim=g_all.shape[1]).to(device)
    pos_weight = _class_weights(labels[train_idx]).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def loss_fn(m, seq, globf, y):
        logits, z = m(seq, globf)
        return bce(logits, y) + lambda_con * supervised_contrastive_loss(z, y.view(-1))

    return _fit_dl_loop(
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
    )


def prepare_feature_matrices(df, feature_mode: str = "both"):
    sequences = df["sequence"].tolist()
    labels = df["label"].to_numpy(dtype=int)
    X_tab = build_tabular_features(df, feature_mode=feature_mode)
    global_raw = np.stack(
        [
            calculate_global_properties(r.sequence, r.n_terminus, r.c_terminus)
            for r in df.itertuples()
        ]
    )
    return sequences, labels, X_tab, global_raw
