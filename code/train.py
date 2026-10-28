"""GroupKFold training for pair / single models."""

from __future__ import annotations

import json
import hashlib
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
    build_neg_pairs,
    build_training_pairs,
    build_within_fold_pairs,
    build_within_fold_strict_pairs,
    load_or_build_cache,
    load_or_build_neg_cache,
    pair_collate,
    precompute_neighbors,
    single_collate,
)
from .evaluation import eval_explicit_pairs, eval_pair_delta, eval_peptide_mic, save_eval_outputs
from .models import build_model, build_optimizer, build_scheduler
from .utils.config import Config
from .utils.metrics import (
    apply_calibrator,
    apply_calibrator_nested,
    fit_calibrator,
    is_better_selection,
    regression_metrics,
)
from .utils.scaling import (
    apply_p_f_stats,
    apply_node_x_stats,
    cache_raw_p_f,
    cache_raw_node_x,
    fit_p_f_stats,
    fit_node_x_stats,
)
from .utils.seed import set_seed


def _apply_fold_zscore(cfg: Config, graphs, tr_idx, va_idx, out_dir: Path, fold: int, extra_graphs=None):
    idx_all = np.concatenate([np.asarray(tr_idx), np.asarray(va_idx)])
    p_stats = None
    node_stats = None
    if str(cfg.p_feature_scaling).lower() == "zscore":
        cache_raw_p_f(graphs)
        p_stats = fit_p_f_stats(graphs, tr_idx)
        apply_p_f_stats(graphs, p_stats, idx_all)
        if extra_graphs:
            # Negatives: transformed by the fold's AMP-fit stats (never fitted on).
            cache_raw_p_f(extra_graphs)
            apply_p_f_stats(extra_graphs, p_stats)
        (out_dir / "intermediate" / f"p_feature_zscore_fold{fold}.json").write_text(
            json.dumps(p_stats, indent=2)
        )
    if str(getattr(cfg, "node_feature_scaling", "none")).lower() == "zscore":
        cache_raw_node_x(graphs)
        node_stats = fit_node_x_stats(graphs, tr_idx)
        apply_node_x_stats(graphs, node_stats, idx_all)
        if extra_graphs:
            cache_raw_node_x(extra_graphs)
            apply_node_x_stats(extra_graphs, node_stats)
        (out_dir / "intermediate" / f"node_feature_zscore_fold{fold}.json").write_text(
            json.dumps(node_stats, indent=2)
        )
    return {"global": p_stats, "node": node_stats}


def _cls_pretrain_encoder(cfg: Config, model, tr_graphs, neg_graphs, out_dir: Path, fold: int, device):
    """N2: BCE pretraining of the shared encoder on fold-train AMPs (label 1)
    vs the negative pool (label 0) via a temporary Linear head, before the
    normal objective. The main optimizer/scheduler are built afterwards, so
    fine-tuning starts fresh. Only used when cls_pretrain_epochs > 0."""
    graphs = list(tr_graphs) + list(neg_graphs)
    y = np.concatenate([np.ones(len(tr_graphs)), np.zeros(len(neg_graphs))]).astype(np.float32)
    head = torch.nn.Linear(cfg.emb_dim, 1).to(device)
    params = list(model.encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(
        params, lr=float(cfg.cls_lr if getattr(cfg, "cls_lr", None) else cfg.lr),
        weight_decay=cfg.weight_decay,
    )
    loader = torch.utils.data.DataLoader(
        SingleDataset(graphs, y, range(len(graphs))),
        batch_size=256, shuffle=True, collate_fn=single_collate,
    )
    model.train()
    log_rows = []
    for ep in range(1, int(cfg.cls_pretrain_epochs) + 1):
        losses = []
        for g, t in loader:
            g, t = g.to(device), t.to(device)
            opt.zero_grad()
            logits = head(model.encoder(g)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        log_rows.append({"epoch": ep, "bce": float(np.mean(losses))})
        print(f"[{cfg.name}/fold{fold}] clspretrain ep{ep} BCE={np.mean(losses):.4f}", flush=True)
    pd.DataFrame(log_rows).to_csv(out_dir / "logs" / f"fold{fold}_clspretrain.csv", index=False)


def _single_pretrain_encoder(cfg: Config, model, tr_graphs, labels_tr, neg_graphs, neg_labels,
                             va_graphs, labels_va, out_dir: Path, fold: int, device):
    """方案 B / 实验一: pretrain the shared encoder on the SINGLE objective
    (per-peptide log2MIC MSE, incl. the cap-anchored negative rows) before pair
    fine-tuning. Head shape mirrors SingleDeltaModel.score; the temporary head
    is discarded afterwards.

    single_pretrain_full=True (实验一) replicates the single+N1 formal regime:
    cosine schedule over cfg.single_pretrain_epochs, per-epoch validation on the
    fold's val AMPs (peptide MSE, negatives never in val), early stopping with
    cfg.early_stop_patience, and the best-epoch encoder is kept."""
    full = bool(getattr(cfg, "single_pretrain_full", False))
    graphs = list(tr_graphs) + list(neg_graphs)
    y = np.concatenate([labels_tr, neg_labels]).astype(np.float32)
    head = torch.nn.Sequential(
        torch.nn.Linear(cfg.emb_dim, cfg.emb_dim),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Dropout(cfg.dropout_ratio),
        torch.nn.Linear(cfg.emb_dim, 1),
    ).to(device)
    params = list(model.encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(
        params, lr=float(cfg.cls_lr if getattr(cfg, "cls_lr", None) else cfg.lr),
        weight_decay=cfg.weight_decay,
    )
    sched = build_scheduler(opt, cfg, int(cfg.single_pretrain_epochs)) if full else None
    loader = torch.utils.data.DataLoader(
        SingleDataset(graphs, y, range(len(graphs))),
        batch_size=256, shuffle=True, collate_fn=single_collate,
    )
    va_loader = None
    if full and len(va_graphs):
        va_loader = torch.utils.data.DataLoader(
            SingleDataset(list(va_graphs), labels_va.astype(np.float32), range(len(va_graphs))),
            batch_size=512, shuffle=False, collate_fn=single_collate,
        )
    model.train()
    log_rows, best_val, stale, best_state = [], float("inf"), 0, None
    for ep in range(1, int(cfg.single_pretrain_epochs) + 1):
        losses = []
        for g, t in loader:
            g, t = g.to(device), t.to(device)
            opt.zero_grad()
            loss = F.mse_loss(head(model.encoder(g)).squeeze(-1), t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if sched is not None:
            sched.step()
        row = {"epoch": ep, "mse": float(np.mean(losses))}
        if va_loader is not None:
            model.eval()
            with torch.no_grad():
                errs, n = 0.0, 0
                for g, t in va_loader:
                    g, t = g.to(device), t.to(device)
                    p = head(model.encoder(g)).squeeze(-1)
                    errs += float(((p - t) ** 2).sum().cpu())
                    n += len(t)
            val_mse = errs / max(n, 1)
            row["val_mse"] = val_mse
            model.train()
            if val_mse < best_val - float(cfg.min_delta):
                best_val, stale = val_mse, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
            else:
                stale += 1
            print(f"[{cfg.name}/fold{fold}] singlepretrain ep{ep} MSE={np.mean(losses):.4f} val={val_mse:.4f} best={best_val:.4f}", flush=True)
            if stale >= cfg.early_stop_patience:
                break
        else:
            print(f"[{cfg.name}/fold{fold}] singlepretrain ep{ep} MSE={np.mean(losses):.4f}", flush=True)
        log_rows.append(row)
    if best_state is not None:
        model.encoder.load_state_dict(best_state)
    pd.DataFrame(log_rows).to_csv(out_dir / "logs" / f"fold{fold}_singlepretrain.csv", index=False)


def _apply_final_zscore(cfg: Config, graphs, test_graphs, out_dir: Path):
    p_stats = None
    node_stats = None
    if str(cfg.p_feature_scaling).lower() == "zscore":
        cache_raw_p_f(graphs)
        if test_graphs:
            cache_raw_p_f(test_graphs)
        p_stats = fit_p_f_stats(graphs)
        apply_p_f_stats(graphs, p_stats)
        if test_graphs:
            apply_p_f_stats(test_graphs, p_stats)
        (out_dir / "intermediate" / "p_feature_zscore_final.json").write_text(
            json.dumps(p_stats, indent=2)
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
    return {"global": p_stats, "node": node_stats}


def train(cfg: Config, root: Path | None = None) -> dict:
    """Run GroupKFold CV (no full-data final retrain). Returns summary dict."""
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
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    (out_dir / 'intermediate' / 'input_identity.json').write_text(json.dumps({
        'graph_cache_identity':payload.get('cache_identity'),
        'n_training_rows':len(rows),'n_sequences':int(rows.sequence.nunique()),
        'ACT_rows':int(rows.n_terminus.fillna('').astype(str).str.upper().eq('ACT').sum()),
        'cpu_threads':torch.get_num_threads()},ensure_ascii=False,indent=2))

    # Non-AMP negative pool (N1/N2): SEPARATE lists — negatives never enter the
    # fold split, the validation anchor pool, the OOF table or the calibrator.
    neg_payload = load_or_build_neg_cache(cfg, root)
    neg_graphs: list = []
    neg_rows = pd.DataFrame()
    neg_labels = np.asarray([], dtype=float)
    if neg_payload is not None:
        keep = ~neg_payload["neg_rows"].sequence.isin(set(rows.sequence))
        neg_graphs = [g for g, k in zip(neg_payload["neg_graphs"], keep) if k]
        neg_rows = neg_payload["neg_rows"][keep].reset_index(drop=True)
        neg_labels = neg_rows["label"].to_numpy(float)
        print(
            f"[{cfg.name}] negatives: {len(neg_graphs)} (dropped {int((~keep).sum())} train overlaps), "
            f"cap log2={cfg.neg_cap_log2:.3f}, stats={neg_payload['neg_stats']}",
            flush=True,
        )
    has_neg = len(neg_graphs) > 0

    if str(cfg.p_feature_scaling).lower() == "zscore":
        cache_raw_p_f(graphs)
        if test_graphs:
            cache_raw_p_f(test_graphs)
    if str(getattr(cfg, "node_feature_scaling", "none")).lower() == "zscore":
        cache_raw_node_x(graphs)
        if test_graphs:
            cache_raw_node_x(test_graphs)

    # Fold grouping: default = exact sequence; "family" = NW>=0.7 clusters so
    # whole pair-connected families stay in one fold (cross-family validation,
    # matching the sequence-disjoint external regime). See utils/family_cluster.py.
    if str(getattr(cfg, "fold_grouping", "sequence")).lower() == "family":
        from .utils.family_cluster import cluster_families

        fam_map = cluster_families(rows.sequence.tolist(), threshold=0.7)
        groups = rows.sequence.map(fam_map)
        n_clusters = len(set(fam_map.values()))
        print(f"[{cfg.name}] fold_grouping=family: {n_clusters} clusters over {len(rows)} rows", flush=True)
    else:
        groups = rows.sequence
    splits = list(GroupKFold(cfg.folds).split(np.arange(len(rows)), groups=groups))
    frames, peptide_frames, fold_metrics = [], [], []

    for fold, (tr, va) in enumerate(splits, 1):
        (out_dir/'intermediate'/f'fold{fold}_indices.json').write_text(json.dumps({'train':tr.tolist(),'validation':va.tolist()}))
        _apply_fold_zscore(cfg, graphs, tr, va, out_dir, fold, extra_graphs=neg_graphs if has_neg else None)
        tr_rows = rows.iloc[tr].reset_index(drop=True)
        va_rows = rows.iloc[va].reset_index(drop=True)
        tr_graphs = [graphs[i] for i in tr]
        va_graphs = [graphs[i] for i in va]
        # v2-6 intervention B (Config.val_far_only): extrapolative checkpoint
        # selection — the selection metric is scored only on validation peptides
        # with NO >=0.7 NW neighbor in the training fold. Training split, OOF
        # table and calibrator keep the full validation fold.
        va_sel_graphs, va_sel_rows, va_sel_idx = va_graphs, va_rows, va
        if bool(getattr(cfg, "val_far_only", False)):
            from .dataloader import far_val_mask
            far = far_val_mask(tr_rows, va_rows, cfg)
            if int(far.sum()) >= 50:
                va_sel_graphs = [va_graphs[i] for i in np.flatnonzero(far)]
                va_sel_rows = va_rows.iloc[np.flatnonzero(far)].reset_index(drop=True)
                va_sel_idx = va[np.flatnonzero(far)]
                print(f"[{cfg.name}/fold{fold}] val_far_only: {int(far.sum())}/{len(va)} extrapolative val peptides", flush=True)
            else:
                print(f"[{cfg.name}/fold{fold}] val_far_only: only {int(far.sum())} far peptides — fallback to full val", flush=True)
        model = build_model(cfg).to(device)
        if int(getattr(cfg, "cls_pretrain_epochs", 0) or 0) > 0 and has_neg:
            _cls_pretrain_encoder(cfg, model, tr_graphs, neg_graphs, out_dir, fold, device)
        if int(getattr(cfg, "single_pretrain_epochs", 0) or 0) > 0 and has_neg:
            _single_pretrain_encoder(cfg, model, tr_graphs, labels[tr], neg_graphs, neg_labels,
                                     va_graphs, labels[va], out_dir, fold, device)
        opt = build_optimizer(model, cfg)
        sched = build_scheduler(opt, cfg, cfg.graph_epochs)

        if cfg.model_kind == "single":
            if cfg.neg_train_inject and has_neg:
                fold_graphs = list(tr_graphs) + list(neg_graphs)
                fold_labels = np.concatenate([labels[tr], neg_labels])
                fold_indices = range(len(fold_graphs))
            else:
                fold_graphs, fold_labels, fold_indices = tr_graphs, labels[tr], range(len(tr))
            loader = torch.utils.data.DataLoader(
                SingleDataset(fold_graphs, fold_labels, fold_indices),
                batch_size=256,
                shuffle=True,
                collate_fn=single_collate,
            )
        else:
            pairs = build_training_pairs(tr_rows, labels[tr], cfg, cfg.seed + fold)
            pair_audit = {'positive_pairs':len(pairs),
                          'positive_pair_sha256':hashlib.sha256(np.asarray(pairs,dtype=np.int64).tobytes()).hexdigest(),
                          'negative_pairs':0,'negative_unique_peptides':0,'negative_pool_size':len(neg_rows),
                          'negative_sampling_mode':cfg.neg_pair_mode,
                          'train_similarity_gate_enabled':cfg.train_similarity_gate_enabled,
                          'train_similarity_threshold':cfg.similarity_threshold}
            fold_graphs, fold_labels = tr_graphs, labels[tr]
            if cfg.neg_train_inject and has_neg:
                neg_pairs = build_neg_pairs(tr_rows, neg_rows, cfg)
                used_neg={x-len(tr_rows) for pair in neg_pairs for x in pair if x>=len(tr_rows)}
                pair_audit.update(negative_pairs=len(neg_pairs),negative_unique_peptides=len(used_neg),
                                  negative_pair_sha256=hashlib.sha256(np.asarray(neg_pairs,dtype=np.int64).tobytes()).hexdigest())
                print(f"[{cfg.name}/fold{fold}] injected neg pairs: {len(neg_pairs)}", flush=True)
                pairs = pairs + neg_pairs
                fold_graphs = list(tr_graphs) + list(neg_graphs)
                fold_labels = np.concatenate([labels[tr], neg_labels])
            (out_dir/'intermediate'/f'fold{fold}_pair_audit.json').write_text(json.dumps(pair_audit,indent=2))
            loader = torch.utils.data.DataLoader(
                PairDataset(fold_graphs, fold_labels, pairs),
                batch_size=cfg.pair_batch_size,
                shuffle=True,
                collate_fn=pair_collate,
            )
        # 方案 A anchor multitask: standalone negative rows feed the absolute
        # score head only (never the pair-delta loss).
        neg_score_loader = None
        if (float(getattr(cfg, "anchor_multitask", 0) or 0) > 0
                and bool(getattr(cfg, "anchor_neg_rows", False)) and has_neg
                and getattr(model, "anchor_head", None) is not None):
            neg_score_loader = torch.utils.data.DataLoader(
                SingleDataset(list(neg_graphs), neg_labels, range(len(neg_graphs))),
                batch_size=cfg.pair_batch_size,
                shuffle=True,
                collate_fn=single_collate,
            )
        # Neighbor graph is only for pair-Δ *evaluation* after the selected
        # checkpoint. Single-head training never samples pairs.
        neighbors = None if str(cfg.model_kind).lower() == "single" else precompute_neighbors(tr_rows, va_rows, cfg)
        # Experiment 16: within-fold same-family validation pairs for checkpoint
        # selection (requires family fold grouping; fam_map computed above).
        # Only the selection metric changes — OOF/calibrator stay neighbor-based.
        val_pair_table = None
        if str(getattr(cfg, "val_pair_mode", "neighbor")).lower() == "within_fold_family":
            assert str(getattr(cfg, "fold_grouping", "sequence")).lower() == "family", \
                "val_pair_mode=within_fold_family requires fold_grouping=family"
            val_pair_table = build_within_fold_pairs(va_rows, fam_map, int(cfg.val_pair_cap), cfg.seed + fold)
        elif str(getattr(cfg, "val_pair_mode", "neighbor")).lower() == "within_fold_strict":
            val_pair_table = build_within_fold_strict_pairs(va_rows, cfg, int(cfg.val_pair_cap), cfg.seed + fold)
        # Checkpoint selection uses THIS fold's validation metrics (selection_score).
        # After training, fold{k}_best.pt is reloaded and scored on the same validation
        # fold; those predictions are concatenated into the OOF CV table. This is
        # model-selection-aware OOF (possible mild within-fold optimistic bias), not a
        # fully nested outer test. Prefer raw OOF as the primary CV report.
        # Single head: save/early-stop on per-sequence log2(MIC) only. Pair-Δ
        # (score(template)-score(mutated)) is computed later on that checkpoint.
        best_score, best_epoch, stale, history, last_epoch = float("inf"), 0, 0, [], 0
        best_met: dict = {}
        is_single = str(cfg.model_kind).lower() == "single"

        for epoch in range(1, cfg.graph_epochs + 1):
            model.train()
            losses = []
            for batch in loader:
                opt.zero_grad()
                if is_single:
                    g, y = batch
                    g, y = g.to(device), y.to(device)
                    loss = F.mse_loss(model.score(g), y)
                else:
                    a, b, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    y_a = batch[4] if len(batch) >= 6 else None
                    y_b = batch[5] if len(batch) >= 6 else None
                    if str(getattr(cfg, "antisym_loss", "none")).lower() == "symmetric":
                        # Antisymmetrized training prediction: delta(a,b) = -delta(b,a)
                        # by construction (evaluation stays single-direction).
                        pred = 0.5 * (model(a, b) - model(b, a))
                    else:
                        pred = model(a, b)
                    loss = F.mse_loss(pred, y)
                    # Prior-anchor regularizer on the sequence AA embedding table
                    # (Config.seq_emb_prior_reg): keeps the learnable table near the
                    # fixed orthonormal projection of the handcrafted descriptor.
                    lam_prior = float(getattr(cfg, "seq_emb_prior_reg", 0) or 0)
                    if lam_prior > 0 and getattr(model.encoder, "aa_prior90", None) is not None:
                        w_tab = model.encoder.aa_embedding.weight[1:]
                        loss = loss + lam_prior * F.mse_loss(w_tab, model.encoder.aa_prior90[1:])
                    # 方案 A anchor multitask: auxiliary absolute-score loss on both
                    # endpoints (shared-encoder head), weight = anchor_multitask.
                    lam = float(getattr(cfg, "anchor_multitask", 0) or 0)
                    if lam > 0 and getattr(model, "anchor_head", None) is not None and y_a is not None:
                        y_a, y_b = y_a.to(device), y_b.to(device)
                        loss = loss + lam * (F.mse_loss(model.score(a), y_a)
                                             + F.mse_loss(model.score(b), y_b)) / 2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            # 方案 A: one pass over standalone negative rows per epoch (score
            # head only) — the single+N1 anchor mechanism ported to the pair model.
            if neg_score_loader is not None:
                lam = float(getattr(cfg, "anchor_multitask", 0) or 0)
                model.train()
                for g, yy in neg_score_loader:
                    g, yy = g.to(device), yy.to(device)
                    opt.zero_grad()
                    nloss = lam * F.mse_loss(model.score(g), yy)
                    nloss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    losses.append(float(nloss.detach().cpu()))
            if sched is not None:
                sched.step()
            last_epoch = epoch
            if epoch == 1 or epoch % 2 == 0:
                if is_single:
                    _pep_pred, sel_met = eval_peptide_mic(
                        model, va_sel_graphs, va_sel_rows, labels[va_sel_idx], cfg, device
                    )
                    pep_met = sel_met
                    pair_met = None
                else:
                    if val_pair_table is not None:
                        _pair_pred, sel_met = eval_explicit_pairs(
                            model, va_graphs, va_rows, val_pair_table, cfg, device
                        )
                    else:
                        _pair_pred, sel_met = eval_pair_delta(
                            model, tr_graphs, tr_rows, labels[tr], va_graphs, va_rows, labels[va], cfg, device, neighbors=neighbors
                        )
                    pep_met = None
                    pair_met = sel_met
                improved = is_better_selection(sel_met, best_score, cfg.min_delta)
                if improved:
                    best_score = float(sel_met["selection_score"])
                    best_epoch = epoch
                    best_met = {k: sel_met[k] for k in ("log2MAE", "RSE", "PCC", "KCC", "selection_score") if k in sel_met}
                    best_met["selection_target"] = "peptide_log2_mic" if is_single else "pair_delta"
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
                hist_row = {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "best_epoch": best_epoch,
                    "best_selection_score": best_score if np.isfinite(best_score) else None,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "selection_target": "peptide_log2_mic" if is_single else "pair_delta",
                }
                if pep_met:
                    hist_row.update({f"peptide_{k}": v for k, v in pep_met.items()})
                if pair_met:
                    hist_row.update(pair_met)
                history.append(hist_row)
                pd.DataFrame(history).to_csv(out_dir / "logs" / f"fold{fold}_history.csv", index=False)
                if is_single:
                    print(
                        f"[{cfg.name}/fold{fold}] epoch={epoch} "
                        f"peptide_log2MAE={sel_met.get('log2MAE', float('nan')):.5f} "
                        f"peptide_RSE={sel_met.get('RSE', float('nan')):.5f} "
                        f"peptide_PCC={sel_met.get('PCC', float('nan')):.5f} "
                        f"peptide_KCC={sel_met.get('KCC', float('nan')):.5f} "
                        f"sel={sel_met.get('selection_score', float('nan')):.5f} "
                        f"best={best_score:.5f} lr={opt.param_groups[0]['lr']:.6g}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{cfg.name}/fold{fold}] epoch={epoch} "
                        f"pair_log2MAE={sel_met.get('log2MAE', float('nan')):.5f} "
                        f"pair_RSE={sel_met.get('RSE', float('nan')):.5f} "
                        f"pair_PCC={sel_met.get('PCC', float('nan')):.5f} "
                        f"pair_KCC={sel_met.get('KCC', float('nan')):.5f} "
                        f"sel={sel_met.get('selection_score', float('nan')):.5f} "
                        f"best={best_score:.5f} lr={opt.param_groups[0]['lr']:.6g}",
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
        # Reported CV is always neighbor pair-Δ. For the single head this is
        # score(template)-score(mutated) from the peptide-selected checkpoint.
        if neighbors is None:
            neighbors = precompute_neighbors(tr_rows, va_rows, cfg)
        pred, met = eval_pair_delta(
            model, tr_graphs, tr_rows, labels[tr], va_graphs, va_rows, labels[va], cfg, device, neighbors=neighbors
        )
        pred.insert(0, "fold", fold)
        frames.append(pred)
        pep_best = None
        if str(cfg.model_kind).lower() == "single":
            pep_best, pep_best_met = eval_peptide_mic(
                model, va_graphs, va_rows, labels[va], cfg, device
            )
            pep_best.insert(0, "fold", fold)
            peptide_frames.append(pep_best)
        else:
            pep_best_met = None
        fold_row = {
            "fold": fold,
            "best_epoch": best_epoch,
            "last_epoch": last_epoch,
            "best_ckpt": f"fold{fold}_best.pt",
            "last_ckpt": f"fold{fold}_last.pt",
            **met,
        }
        if pep_best_met:
            fold_row.update({f"peptide_{k}": v for k, v in pep_best_met.items()})
        fold_metrics.append(fold_row)

    cv_raw = pd.concat(frames, ignore_index=True)
    # Primary CV = raw OOF (no calibrator fit on the same predictions).
    # Nested leave-one-fold calibration is reported separately for reference.
    # The saved calibrator is fit on all OOF rows for *external* application only.
    cv_nested, nested_cals = apply_calibrator_nested(cv_raw)
    calibrator = fit_calibrator(cv_raw)
    cv_insample = apply_calibrator(cv_raw, calibrator)

    peptide_cv = pd.concat(peptide_frames, ignore_index=True) if peptide_frames else pd.DataFrame()
    peptide_metrics = None
    if len(peptide_cv):
        peptide_metrics = regression_metrics(peptide_cv.y_true_log2_mic, peptide_cv.y_pred_log2_mic)

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

    is_single = str(cfg.model_kind).lower() == "single"
    summary = {
        "experiment": cfg.name,
        "model_kind": cfg.model_kind,
        "aa_encoding": "sequential_1_to_20_alphabetical",
        "cv": regression_metrics(
            cv_raw.y_true_delta_log2_anchor_minus_query,
            cv_raw.y_pred_delta_log2_anchor_minus_query,
        ),
        "cv_calibrated_nested": regression_metrics(
            cv_nested.y_true_delta_log2_anchor_minus_query,
            cv_nested.y_pred_delta_log2_anchor_minus_query,
        ),
        "cv_calibrated_insample": regression_metrics(
            cv_insample.y_true_delta_log2_anchor_minus_query,
            cv_insample.y_pred_delta_log2_anchor_minus_query,
        ),
        "external": external_metrics,
        "folds": fold_metrics,
        "calibrator": calibrator,
        "calibrator_nested_folds": nested_cals,
        "calibration_protocol": {
            "primary_cv": "raw_oof",
            "nested": "leave_one_fold_linear",
            "saved_calibrator": "fit_on_all_oof_for_external_only",
            "note": (
                "Do not treat cv_calibrated_insample as out-of-sample. "
                "Primary reported CV is raw OOF; nested calibration is secondary."
            ),
        },
        "checkpoints": {
            "per_fold": ["fold{k}_best.pt", "fold{k}_last.pt"],
            "final_retrain": False,
        },
        "runtime_seconds": round(time.time() - started, 2),
        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
        "cpu_threads":torch.get_num_threads(),
        "config": cfg.to_dict(),
    }
    if is_single:
        summary["cv_peptide"] = peptide_metrics
        summary["selection_target"] = "peptide_log2_mic"
        summary["comparison_note"] = (
            "Internal ablation (not an external baseline). Train and checkpoint "
            "selection use per-sequence log2(MIC) only — no pair sampling. "
            "Reported `cv` is still neighbor pair-Δ: score(template)-score(mutated) "
            "on the same protocol as the siamese arm. `cv_peptide` is the "
            "per-sequence log2(MIC) used to save fold*_best.pt."
        )
    # Persist raw OOF predictions as the primary CV table.
    save_eval_outputs(
        out_dir,
        cv_raw,
        fold_metrics,
        calibrator,
        summary,
        peptide_cv=peptide_cv if len(peptide_cv) else None,
    )
    print(
        json.dumps(
            {
                "cv": summary["cv"],
                "cv_peptide": summary.get("cv_peptide"),
                "cv_calibrated_nested": summary["cv_calibrated_nested"],
                "external": summary["external"],
                "checkpoints": summary["checkpoints"],
                "calibration_protocol": summary["calibration_protocol"],
            },
            indent=2,
        )
    )
    return summary
