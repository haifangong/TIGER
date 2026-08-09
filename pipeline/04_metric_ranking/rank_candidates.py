#!/usr/bin/env python3
"""Step 4 — Rank candidate peptides by predicted MIC (TIGER wetlab models).

Uses 5-fold ensemble checkpoints from ``checkpoints/wetlab/sim{0p3|0p7}_<Species>/``.

Modes
-----
1. **template** (default): score each candidate vs a template anchor with known MIC.
   ``MIC_query = MIC_template / 2 ** delta``, where
   ``delta = log2(MIC_template) - log2(MIC_query)`` is the model prediction.

2. **neighbor**: score each candidate against the DBAASP training anchor pool
   (same neighbor protocol as training eval) and aggregate absolute MIC estimates.

Requires Rosetta/HelixFold PDBs named ``{SEQUENCE}.pdb`` for template + candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
TIGER = PIPELINE.parent
if str(TIGER) not in sys.path:
    sys.path.insert(0, str(TIGER))

from code.dataloader import (  # noqa: E402
    build_graphs,
    load_or_build_cache,
    pair_collate,
    precompute_neighbors,
    preprocess_dataset,
)
from code.evaluation import load_checkpoint_model  # noqa: E402
from code.utils.config import load_config  # noqa: E402
from code.utils.constants import CFU_COL, TARGET_COL  # noqa: E402
from code.utils.metrics import apply_calibrator  # noqa: E402
from code.utils.scaling import (  # noqa: E402
    apply_global_f_stats,
    apply_node_x_stats,
    cache_raw_global_f,
    cache_raw_node_x,
)

DEFAULT_RUNS = TIGER / "checkpoints/wetlab"
DEFAULT_DATA = TIGER / "data" / "wetlab"


def _tag_float(x: float) -> str:
    return str(x).replace(".", "p")


def resolve_run_dir(runs_root: Path, species: str, similarity: float) -> Path:
    run_dir = runs_root / f"sim{_tag_float(similarity)}_{species}"
    if not run_dir.exists():
        # allow MIC_ prefix or slight naming variants
        cands = sorted(runs_root.glob(f"sim{_tag_float(similarity)}_*{species}*"))
        if not cands:
            raise FileNotFoundError(
                f"No wetlab run at {run_dir}. Train first via "
                f"bash code/scripts_train/run_wetlab_species_sim.sh"
            )
        run_dir = cands[0]
    return run_dir


def load_run_cfg(run_dir: Path):
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    return load_config(cfg_path)


def list_fold_ckpts(run_dir: Path, prefer: str = "best") -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    paths = sorted(ckpt_dir.glob(f"fold*_{prefer}.pt"))
    if not paths:
        paths = sorted(ckpt_dir.glob("fold*_best.pt"))
    if not paths:
        raise FileNotFoundError(f"No fold checkpoints under {ckpt_dir}")
    return paths


def read_sequences(csv_path: Path, sequence_col: str = "sequence") -> pd.DataFrame:
    # Support headerless "seq,seq" from step 1 positives.
    text = csv_path.read_text(encoding="utf-8-sig").splitlines()
    if not text:
        return pd.DataFrame(columns=[sequence_col])
    first = text[0].strip().lower()
    if "sequence" in first or first.startswith("seq"):
        df = pd.read_csv(csv_path)
        col = sequence_col if sequence_col in df.columns else df.columns[0]
    else:
        seqs = []
        for line in text:
            if not line.strip():
                continue
            seqs.append(line.strip().split(",")[0].strip().upper())
        df = pd.DataFrame({sequence_col: seqs})
        col = sequence_col
    df = df.copy()
    df[col] = df[col].astype(str).str.strip().str.upper()
    df = df[df[col].str.len() > 0].reset_index(drop=True)
    if col != "sequence":
        df["sequence"] = df[col]
    return df


def build_query_table(
    sequences: list[str],
    pdb_dir: Path,
    *,
    cfu_group: str = "1E5 - 1E6",
    dummy_mic: float = 1.0,
) -> pd.DataFrame:
    rows = []
    for seq in sequences:
        pdb = pdb_dir / f"{seq}.pdb"
        if not pdb.exists():
            # HelixFold may sanitize spaces; try underscore form.
            alt = pdb_dir / f"{seq.replace(' ', '_')}.pdb"
            pdb = alt if alt.exists() else pdb
        rows.append(
            {
                "sequence": seq,
                "n_terminus": "",
                "c_terminus": "",
                CFU_COL: cfu_group,
                TARGET_COL: dummy_mic,
                "pdb_path": str(pdb) if pdb.exists() else "",
                "has_pdb": pdb.exists(),
            }
        )
    return pd.DataFrame(rows)


def _apply_zscore_stats(graphs, run_dir: Path, cfg) -> None:
    if str(cfg.global_feature_scaling).lower() == "zscore":
        cache_raw_global_f(graphs)
        sp = run_dir / "intermediate" / "global_feature_zscore_final.json"
        if sp.exists():
            apply_global_f_stats(graphs, json.loads(sp.read_text()))
    if str(getattr(cfg, "node_feature_scaling", "none")).lower() == "zscore":
        cache_raw_node_x(graphs)
        sp = run_dir / "intermediate" / "node_feature_zscore_final.json"
        if sp.exists():
            apply_node_x_stats(graphs, json.loads(sp.read_text()))


@torch.no_grad()
def score_template_pairs(
    model,
    anchor_graph,
    query_graphs: list,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Return pred delta log2(MIC_anchor) - log2(MIC_query) for each query."""
    model.eval()
    preds = []
    for start in range(0, len(query_graphs), batch_size):
        chunk = query_graphs[start : start + batch_size]
        items = [
            (
                anchor_graph,
                q,
                torch.tensor(0.0),
                torch.tensor(float("nan")),
            )
            for q in chunk
        ]
        loader = torch.utils.data.DataLoader(
            items, batch_size=len(chunk), shuffle=False, collate_fn=pair_collate
        )
        for a_batch, b_batch, _, _ in loader:
            a_batch, b_batch = a_batch.to(device), b_batch.to(device)
            pred = model(a_batch, b_batch).detach().cpu().numpy().ravel()
            preds.append(pred)
    return np.concatenate(preds) if preds else np.zeros(0, dtype=float)


def calibrate_deltas(deltas: np.ndarray, calibrator: dict | None) -> np.ndarray:
    if not calibrator:
        return deltas
    slope = float(calibrator.get("slope", 1.0))
    intercept = float(calibrator.get("intercept", 0.0))
    return slope * deltas + intercept


def rank_template_mode(args) -> pd.DataFrame:
    run_dir = resolve_run_dir(Path(args.runs_root), args.species, args.similarity)
    cfg = load_run_cfg(run_dir)
    cfg.device = "cpu" if args.cpu else f"cuda:{args.gpu}"
    device = torch.device(cfg.device if torch.cuda.is_available() and not args.cpu else "cpu")

    cand_df = read_sequences(Path(args.csv), args.sequence_col)
    sequences = cand_df["sequence"].tolist()
    if args.template.upper() not in {s.upper() for s in sequences}:
        # still score template separately; do not require it in the candidate list
        pass

    pdb_dir = Path(args.pdb_dir)
    query_table = build_query_table(sequences, pdb_dir, cfu_group=args.cfu_group)
    missing = query_table.loc[~query_table["has_pdb"], "sequence"].tolist()
    if missing:
        print(f"[warn] {len(missing)} candidates missing PDB under {pdb_dir} (skipped)", flush=True)
    query_table = query_table[query_table["has_pdb"]].reset_index(drop=True)
    if query_table.empty:
        raise SystemExit("No candidates with PDBs to score")

    # Build graphs via a temporary CSV + preprocess/build_graphs
    tmp_csv = Path(args.out_dir) / "_queries_tmp.csv"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    query_table.to_csv(tmp_csv, index=False)
    q_df, _ = preprocess_dataset(str(tmp_csv), str(pdb_dir), cfg, "query")
    q_graphs, q_rows, _ = build_graphs(q_df, cfg)
    _apply_zscore_stats(q_graphs, run_dir, cfg)

    # Template anchor
    tmpl = args.template.strip().upper()
    tmpl_pdb = pdb_dir / f"{tmpl}.pdb"
    if not tmpl_pdb.exists():
        raise SystemExit(f"Template PDB not found: {tmpl_pdb}")
    if not np.isfinite(args.template_mic) or args.template_mic <= 0:
        raise SystemExit("--template-mic must be a positive µg/mL value in template mode")
    a_table = build_query_table([tmpl], pdb_dir, cfu_group=args.cfu_group, dummy_mic=float(args.template_mic))
    a_tmp = Path(args.out_dir) / "_anchor_tmp.csv"
    a_table.to_csv(a_tmp, index=False)
    a_df, _ = preprocess_dataset(str(a_tmp), str(pdb_dir), cfg, "anchor")
    a_graphs, _, _ = build_graphs(a_df, cfg)
    _apply_zscore_stats(a_graphs, run_dir, cfg)
    if not a_graphs:
        raise SystemExit("Failed to build template graph")
    anchor_graph = a_graphs[0]

    cal_path = run_dir / "results" / "calibrator.json"
    calibrator = json.loads(cal_path.read_text()) if cal_path.exists() and not args.no_calibrator else None
    ckpts = list_fold_ckpts(run_dir, prefer=args.ckpt)
    fold_deltas = []
    for ckpt in ckpts:
        model, _ = load_checkpoint_model(ckpt, cfg, device)
        deltas = score_template_pairs(model, anchor_graph, q_graphs, device, batch_size=args.batch_size)
        deltas = calibrate_deltas(deltas, calibrator)
        fold_deltas.append(deltas)

    stack = np.vstack(fold_deltas)  # [n_folds, n_query]
    delta_mean = stack.mean(axis=0)
    delta_std = stack.std(axis=0, ddof=0)
    # delta = log2(MIC_a) - log2(MIC_q)  →  MIC_q = MIC_a / 2^delta
    mic_pred = float(args.template_mic) / np.power(2.0, delta_mean)
    improvement = float(args.template_mic) / mic_pred  # >1 means more potent than template

    out = q_rows[["sequence"]].copy()
    out["template_sequence"] = tmpl
    out["template_MIC_ug_mL"] = float(args.template_mic)
    out["pred_delta_log2_template_minus_query"] = delta_mean
    out["pred_delta_log2_std"] = delta_std
    out["pred_MIC_ug_mL"] = mic_pred
    out["pred_MIC_improvement_vs_template"] = improvement
    out["n_folds"] = len(ckpts)
    out["species"] = args.species
    out["similarity_threshold"] = args.similarity
    out["run_dir"] = str(run_dir)
    out = out.sort_values("pred_MIC_ug_mL", ascending=True).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def rank_neighbor_mode(args) -> pd.DataFrame:
    """Estimate absolute MIC via DBAASP neighbor anchors (needs train cache)."""
    run_dir = resolve_run_dir(Path(args.runs_root), args.species, args.similarity)
    cfg = load_run_cfg(run_dir)
    # Keep train-only cache; build candidate graphs separately (avoid baking queries into cache).
    cfg.eval_external = False
    cfg.device = "cpu" if args.cpu else f"cuda:{args.gpu}"
    device = torch.device(cfg.device if torch.cuda.is_available() and not args.cpu else "cpu")

    cand_df = read_sequences(Path(args.csv), args.sequence_col)
    pdb_dir = Path(args.pdb_dir)
    query_table = build_query_table(cand_df["sequence"].tolist(), pdb_dir, cfu_group=args.cfu_group)
    query_table = query_table[query_table["has_pdb"]].reset_index(drop=True)
    if query_table.empty:
        raise SystemExit("No candidates with PDBs to score")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tmp_csv = Path(args.out_dir) / "_queries_neighbor_tmp.csv"
    query_table.to_csv(tmp_csv, index=False)

    payload = load_or_build_cache(cfg, root=TIGER)
    graphs, rows = payload["graphs"], payload["rows"]
    _apply_zscore_stats(graphs, run_dir, cfg)
    labels = rows["label"].to_numpy(float)

    q_df, _ = preprocess_dataset(str(tmp_csv), str(pdb_dir), cfg, "query")
    test_graphs, test_rows, _ = build_graphs(q_df, cfg)
    _apply_zscore_stats(test_graphs, run_dir, cfg)
    if not test_graphs:
        raise SystemExit("Failed to build candidate graphs")

    from code.evaluation import eval_pair_delta

    cal_path = run_dir / "results" / "calibrator.json"
    calibrator = json.loads(cal_path.read_text()) if cal_path.exists() and not args.no_calibrator else None
    neighbors = precompute_neighbors(rows, test_rows, cfg)
    ckpts = list_fold_ckpts(run_dir, prefer=args.ckpt)

    per_fold_mic = []
    for ckpt in ckpts:
        model, _ = load_checkpoint_model(ckpt, cfg, device)
        dummy_q_labels = (
            test_rows[TARGET_COL].to_numpy(float)
            if TARGET_COL in test_rows.columns
            else np.ones(len(test_rows))
        )
        pred, _ = eval_pair_delta(
            model,
            graphs,
            rows,
            labels,
            test_graphs,
            test_rows,
            np.log2(np.clip(dummy_q_labels, 1e-12, None)),
            cfg,
            device,
            neighbors=neighbors,
        )
        if calibrator and len(pred):
            pred = apply_calibrator(pred, calibrator)
        if pred.empty:
            per_fold_mic.append(np.full(len(test_rows), np.nan))
            continue
        pred = pred.copy()
        pred["anchor_mic"] = pred["anchor_sequence"].map(dict(zip(rows["sequence"], np.power(2.0, labels))))
        pred["query_mic_est"] = pred["anchor_mic"] / np.power(
            2.0, pred["y_pred_delta_log2_anchor_minus_query"]
        )
        mic_by_q = pred.groupby("query_sequence")["query_mic_est"].median()
        per_fold_mic.append(test_rows["sequence"].map(mic_by_q).to_numpy(float))

    stack = np.vstack(per_fold_mic)
    mic_mean = np.nanmean(stack, axis=0)
    mic_std = np.nanstd(stack, axis=0)

    out = test_rows[["sequence"]].copy()
    out["pred_MIC_ug_mL"] = mic_mean
    out["pred_MIC_std"] = mic_std
    out["n_folds"] = len(ckpts)
    out["species"] = args.species
    out["similarity_threshold"] = args.similarity
    out["run_dir"] = str(run_dir)
    out = out.sort_values("pred_MIC_ug_mL", ascending=True).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Candidate sequences CSV (sequence column or headerless)")
    ap.add_argument("--species", required=True, help="Species slug, e.g. Escherichia_coli")
    ap.add_argument("--similarity", type=float, default=0.3, choices=[0.3, 0.7])
    ap.add_argument("--pdb-dir", required=True, help="Directory of {SEQUENCE}.pdb files")
    ap.add_argument("--out-dir", default=str(HERE / "outputs" / "rank"))
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS))
    ap.add_argument("--mode", choices=["template", "neighbor"], default="template")
    ap.add_argument("--template", default=None, help="Template / WT sequence (required for template mode)")
    ap.add_argument("--template-mic", type=float, default=None, help="Template MIC in µg/mL")
    ap.add_argument("--sequence-col", default="sequence")
    ap.add_argument("--cfu-group", default="1E5 - 1E6")
    ap.add_argument("--ckpt", choices=["best", "last"], default="best")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-calibrator", action="store_true")
    ap.add_argument("--top-k", type=int, default=0, help="If >0, keep only top-K ranked rows")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "template":
        if not args.template or args.template_mic is None:
            raise SystemExit("template mode requires --template and --template-mic")
        ranked = rank_template_mode(args)
    else:
        ranked = rank_neighbor_mode(args)

    if args.top_k and args.top_k > 0:
        ranked = ranked.head(int(args.top_k)).copy()

    out_csv = out_dir / f"ranked_{args.species}_sim{_tag_float(args.similarity)}.csv"
    ranked.to_csv(out_csv, index=False)
    meta = {
        "mode": args.mode,
        "species": args.species,
        "similarity": args.similarity,
        "n_ranked": int(len(ranked)),
        "wrote": str(out_csv),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(ranked.head(min(10, len(ranked))).to_string(index=False))
    return out_csv


if __name__ == "__main__":
    main()
