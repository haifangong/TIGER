#!/usr/bin/env python3
"""Compare TIGER (hard MoE / 0.3 / 0.7) vs EvoGradient on LL37 + APEXGO.

Methods reported:
  - TIGER MoE   (hard gate: similarity < tau → expert 0.3, else → 0.7)
  - TIGER 0.3
  - TIGER 0.7
  - EvoGradient (absolute log10(MIC) → pair Δlog2)

Metrics per peptide group: PCC, KCC, MAE (=log2MAE), RSE.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
EVO_ROOT = ROOT.parent / "baseline" / "AMP-potency-prediction-EvoGradient"
DEFAULT_EXP_03 = ROOT / "checkpoints/ablation" / "04_similarity" / "sim0p3_bal10000"
DEFAULT_EXP_07 = ROOT / "checkpoints/ablation" / "04_similarity" / "sim0p7_bal10000"
LL37_PAIR_CSV = ROOT / "data" / "test_ll37" / "ll37_pairs_neighbor.csv"
APEX_PAIR_CSV = (
    ROOT / "data" / "test_apexgo" / "pairs" / "pairs_template_centric_alldelta_geo3.csv"
)
LL37_PDB = ROOT / "data" / "test_ll37" / "pdb"
APEX_PDB = ROOT / "data" / "test_apexgo" / "pdb"
APEX_PDB_ROSETTA = ROOT / "data" / "3D_data_apexgo_Rosetta"
TRAIN_PDB = ROOT / "data" / "3D_data_train_eva_Rosetta"

from code.dataloader import build_graphs, preprocess_dataset  # noqa: E402
from code.evaluation import eval_explicit_pairs, load_checkpoint_model  # noqa: E402
from code.utils.config import load_config  # noqa: E402
from code.utils.constants import CFU_COL, TARGET_COL  # noqa: E402
from code.utils.metrics import LOG10_OF_2, apply_calibrator, regression_metrics  # noqa: E402
from code.utils.scaling import (  # noqa: E402
    apply_global_f_stats,
    apply_node_x_stats,
    cache_raw_global_f,
    cache_raw_node_x,
)

REPORT_COLS = ("n", "PCC", "KCC", "MAE", "RSE", "log2MAE", "log10MAE")
METHOD_ORDER = ("TIGER MoE", "TIGER 0.7", "TIGER 0.3", "EvoGradient")


def _as_1d(series_or_df) -> np.ndarray:
    if isinstance(series_or_df, pd.DataFrame):
        series_or_df = series_or_df.iloc[:, 0]
    return np.asarray(series_or_df, dtype=float).reshape(-1)


def _metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0}
    m = regression_metrics(
        _as_1d(df["y_true_delta_log2_anchor_minus_query"]),
        _as_1d(df["y_pred_delta_log2_anchor_minus_query"]),
    )
    # User-facing MAE = log2MAE (training target space)
    m["MAE"] = m.get("log2MAE", float("nan"))
    return m


def _apply_fold_scaling(graphs, cfg, exp_dir: Path, fold: int) -> None:
    cache_raw_global_f(graphs)
    cache_raw_node_x(graphs)
    if str(cfg.global_feature_scaling).lower() == "zscore":
        sp = exp_dir / "intermediate" / f"global_feature_zscore_fold{fold}.json"
        if sp.exists():
            apply_global_f_stats(graphs, json.loads(sp.read_text()))
    if str(getattr(cfg, "node_feature_scaling", "none")).lower() == "zscore":
        sp = exp_dir / "intermediate" / f"node_feature_zscore_fold{fold}.json"
        if sp.exists():
            apply_node_x_stats(graphs, json.loads(sp.read_text()))


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_pairs(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Unify column names across LL37 / APEXGO pair CSVs."""
    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    a_mic = _pick_col(out, ["anchor_MIC_ug_per_mL", "anchor_mic_ug_mL", "anchor_MIC"])
    q_mic = _pick_col(out, ["query_MIC_ug_per_mL", "query_mic_ug_mL", "query_MIC"])
    if a_mic is None or q_mic is None:
        raise ValueError(f"[{dataset}] missing MIC columns: {list(out.columns)}")
    out["anchor_MIC_ug_per_mL"] = out[a_mic].astype(float)
    out["query_MIC_ug_per_mL"] = out[q_mic].astype(float)

    if "y_true_delta_log2_anchor_minus_query" not in out.columns:
        if "delta_log2_anchor_minus_query" in out.columns:
            out["y_true_delta_log2_anchor_minus_query"] = out["delta_log2_anchor_minus_query"]
        else:
            out["y_true_delta_log2_anchor_minus_query"] = np.log2(
                out["anchor_MIC_ug_per_mL"]
            ) - np.log2(out["query_MIC_ug_per_mL"])
    out["y_true_delta_log10_anchor_minus_query"] = (
        out["y_true_delta_log2_anchor_minus_query"] * LOG10_OF_2
    )

    out["dataset"] = dataset
    if "family" not in out.columns or out["family"].isna().all():
        out["family"] = dataset
    else:
        out["family"] = out["family"].fillna(dataset).astype(str)
    out["peptide_group"] = out["family"].astype(str)
    return out


def build_combined_pdb_dir(
    sequences: list[str], out_dir: Path, pdb_bases: list[Path]
) -> Path:
    pdb_dir = out_dir / "pdb_combined"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for seq in sequences:
        dst = pdb_dir / f"{seq}.pdb"
        if dst.exists() or dst.is_symlink():
            continue
        src = None
        for base in pdb_bases:
            cand = base / f"{seq}.pdb"
            if cand.exists():
                src = cand
                break
        if src is None:
            missing.append(seq)
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    if missing:
        print(f"[warn] missing pdb for {len(missing)} sequences", flush=True)
        for s in missing[:10]:
            print(f"  missing: {s}", flush=True)
    return pdb_dir


def build_peptide_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for r in pairs.itertuples(index=False):
        for seq, mic in (
            (r.anchor_sequence, r.anchor_MIC_ug_per_mL),
            (r.query_sequence, r.query_MIC_ug_per_mL),
        ):
            if seq not in rows:
                rows[seq] = {
                    "sequence": seq,
                    "n_terminus": "",
                    "c_terminus": "",
                    CFU_COL: "1E5 - 1E6",
                    TARGET_COL: float(mic),
                }
    return pd.DataFrame(list(rows.values()))


def eval_tiger(
    exp_dir: Path,
    pairs: pd.DataFrame,
    out_dir: Path,
    device,
    pdb_bases: list[Path],
    tag: str,
) -> tuple[pd.DataFrame, dict]:
    cfg = load_config(exp_dir / "config.json")
    for key in ("feature_path", "shared_cache_dir"):
        p = Path(getattr(cfg, key))
        if not p.is_absolute():
            setattr(cfg, key, str(ROOT / p))
    cfg.device = "cuda:0"

    pair_table = pairs.copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    peptides = build_peptide_table(pairs)
    pdb_dir = build_combined_pdb_dir(peptides["sequence"].tolist(), out_dir, pdb_bases)
    peptides.to_csv(out_dir / "peptides.csv", index=False)
    filt, prep = preprocess_dataset(
        str(out_dir / "peptides.csv"), str(pdb_dir), cfg, f"{tag}_cmp"
    )
    graphs, rows, gstat = build_graphs(filt, cfg)
    print(f"[tiger {tag}] prep={prep} graphs={gstat}", flush=True)

    have = set(rows["sequence"])
    covered = pair_table[
        pair_table["anchor_sequence"].isin(have) & pair_table["query_sequence"].isin(have)
    ].copy()
    print(f"[tiger {tag}] coverage {len(covered)}/{len(pair_table)}", flush=True)

    cal_path = exp_dir / "results" / "calibrator.json"
    calibrator = (
        json.loads(cal_path.read_text()) if cal_path.exists() else {"slope": 1.0, "intercept": 0.0}
    )

    pred_frames = []
    for fold in range(1, 6):
        ckpt = exp_dir / "checkpoints" / f"fold{fold}_best.pt"
        if not ckpt.exists():
            continue
        _apply_fold_scaling(graphs, cfg, exp_dir, fold)
        model, _ = load_checkpoint_model(ckpt, cfg, device)
        pred, _ = eval_explicit_pairs(model, graphs, rows, covered, cfg, device)
        if len(pred):
            pred = apply_calibrator(pred, calibrator)
            key = ["anchor_sequence", "query_sequence"]
            truth_map = covered.drop_duplicates(key).set_index(key)[
                "y_true_delta_log2_anchor_minus_query"
            ]
            if isinstance(truth_map, pd.DataFrame):
                truth_map = truth_map.iloc[:, 0]
            idx = pd.MultiIndex.from_frame(pred[key])
            pred["y_true_delta_log2_anchor_minus_query"] = truth_map.reindex(idx).to_numpy(
                dtype=float
            )
            pred["y_true_delta_log10_anchor_minus_query"] = (
                pred["y_true_delta_log2_anchor_minus_query"] * LOG10_OF_2
            )
        pred["fold"] = fold
        pred_frames.append(pred)
        m = _metrics(pred)
        print(
            f"[tiger {tag} fold{fold}] n={m.get('n')} PCC={m.get('PCC', float('nan')):.4f} "
            f"MAE={m.get('MAE', float('nan')):.4f}",
            flush=True,
        )

    all_pred = pd.concat(pred_frames, ignore_index=True)
    ens = (
        all_pred.groupby(["anchor_sequence", "query_sequence"], as_index=False)
        .agg(
            y_true_delta_log2_anchor_minus_query=("y_true_delta_log2_anchor_minus_query", "first"),
            y_pred_delta_log2_anchor_minus_query=("y_pred_delta_log2_anchor_minus_query", "mean"),
            similarity=("similarity", "first"),
        )
    )
    # Restore group labels from covered pairs
    meta = covered[
        ["anchor_sequence", "query_sequence", "dataset", "family", "peptide_group"]
    ].drop_duplicates(["anchor_sequence", "query_sequence"])
    ens = ens.merge(meta, on=["anchor_sequence", "query_sequence"], how="left")
    ens["y_pred_delta_log10_anchor_minus_query"] = (
        ens["y_pred_delta_log2_anchor_minus_query"] * LOG10_OF_2
    )
    ens["y_true_delta_log10_anchor_minus_query"] = (
        ens["y_true_delta_log2_anchor_minus_query"] * LOG10_OF_2
    )
    ens_m = _metrics(ens)
    all_pred.to_csv(out_dir / "tiger_pair_predictions_all_folds.csv", index=False)
    ens.to_csv(out_dir / "tiger_pair_predictions_ensemble.csv", index=False)
    return ens, ens_m


def eval_evo(pairs: pd.DataFrame, out_dir: Path, device) -> tuple[pd.DataFrame, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(EVO_ROOT))
    from eval_tiger_style_pair_delta import predict_log10_mic, score_pairs  # noqa: E402

    all_seqs = sorted(set(pairs["query_sequence"]) | set(pairs["anchor_sequence"]))
    print(f"[evo] predicting absolute MIC for {len(all_seqs)} sequences", flush=True)
    pred_df = predict_log10_mic(all_seqs, device)
    pred_map = dict(zip(pred_df["sequence"], pred_df["pred_log2_mic"]))
    pair_pred = score_pairs(
        pairs,
        pred_map,
        anchor_col="anchor_sequence",
        query_col="query_sequence",
        true_delta_col="y_true_delta_log2_anchor_minus_query",
    )
    meta = pairs[
        ["anchor_sequence", "query_sequence", "dataset", "family", "peptide_group"]
    ].drop_duplicates(["anchor_sequence", "query_sequence"])
    pair_pred = pair_pred.merge(meta, on=["anchor_sequence", "query_sequence"], how="left")
    m = _metrics(pair_pred)
    pred_df.to_csv(out_dir / "evo_sequence_predictions.csv", index=False)
    pair_pred.to_csv(out_dir / "evo_pair_predictions.csv", index=False)
    print(
        f"[evo] n={m.get('n')} PCC={m.get('PCC', float('nan')):.4f} "
        f"MAE={m.get('MAE', float('nan')):.4f}",
        flush=True,
    )
    return pair_pred, m


def apply_moe_hard_two_expert(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Hard gate: s < τ → 0.3, else → 0.7."""
    out = df.copy()
    s = out["similarity"].to_numpy(dtype=float)
    use_03 = s < float(tau)
    out["expert"] = np.where(use_03, "0.3", "0.7")
    out["y_pred_delta_log2_anchor_minus_query"] = np.where(
        use_03, out["pred_0.3"].to_numpy(float), out["pred_0.7"].to_numpy(float)
    )
    out["y_pred_delta_log10_anchor_minus_query"] = (
        out["y_pred_delta_log2_anchor_minus_query"] * LOG10_OF_2
    )
    return out


def build_moe_table(ens_03: pd.DataFrame, ens_07: pd.DataFrame) -> pd.DataFrame:
    keep_meta = ["dataset", "family", "peptide_group"]
    a = ens_03.rename(columns={"y_pred_delta_log2_anchor_minus_query": "pred_0.3"})
    cols_a = [
        "anchor_sequence",
        "query_sequence",
        "y_true_delta_log2_anchor_minus_query",
        "similarity",
        "pred_0.3",
        *[c for c in keep_meta if c in a.columns],
    ]
    a = a[cols_a]
    b = ens_07.rename(columns={"y_pred_delta_log2_anchor_minus_query": "pred_0.7"})[
        ["anchor_sequence", "query_sequence", "pred_0.7"]
    ]
    return a.merge(b, on=["anchor_sequence", "query_sequence"], how="inner")


def _row(peptide_group: str, method: str, metrics: dict, dataset: str = "") -> dict:
    return {
        "dataset": dataset,
        "peptide_group": peptide_group,
        "method": method,
        **{k: metrics.get(k) for k in REPORT_COLS},
    }


def metrics_by_group(pred: pd.DataFrame, method: str, dataset_label: str) -> list[dict]:
    """Overall + per peptide_group metrics for one method on one dataset slice."""
    rows = []
    overall = _metrics(pred)
    rows.append(_row(f"{dataset_label} (all)", method, overall, dataset=dataset_label))
    if "peptide_group" in pred.columns:
        for g, sub in pred.groupby("peptide_group", sort=True):
            # Skip redundant single-group datasets (e.g. LL37 family == LL37)
            if str(g) == dataset_label and len(pred["peptide_group"].unique()) == 1:
                continue
            rows.append(_row(str(g), method, _metrics(sub), dataset=dataset_label))
    return rows


def load_dataset(name: str, path: Path) -> tuple[pd.DataFrame, list[Path]]:
    df = normalize_pairs(pd.read_csv(path), name)
    if name == "LL37":
        pdb_bases = [LL37_PDB, TRAIN_PDB]
    else:
        pdb_bases = [APEX_PDB, APEX_PDB_ROSETTA, TRAIN_PDB]
    return df, pdb_bases


def evaluate_dataset(
    name: str,
    pairs: pd.DataFrame,
    pdb_bases: list[Path],
    out_dir: Path,
    exp_03: Path,
    exp_07: Path,
    moe_tau: float,
    device,
) -> tuple[list[dict], dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== {name}: {len(pairs)} pairs ==========", flush=True)

    evo_pred, evo_m = eval_evo(pairs, out_dir / "evo", device)
    ens_03, m_03 = eval_tiger(
        exp_03, pairs, out_dir / "tiger_sim0p3", device, pdb_bases, f"{name}_0.3"
    )
    ens_07, m_07 = eval_tiger(
        exp_07, pairs, out_dir / "tiger_sim0p7", device, pdb_bases, f"{name}_0.7"
    )

    moe_base = build_moe_table(ens_03, ens_07)
    hard = apply_moe_hard_two_expert(moe_base, moe_tau)
    hard_m = _metrics(hard)

    hard.to_csv(out_dir / "tiger_moe_hard_predictions.csv", index=False)
    moe_base.to_csv(out_dir / "tiger_experts_merged.csv", index=False)
    ens_03.to_csv(out_dir / "tiger_0p3_predictions.csv", index=False)
    ens_07.to_csv(out_dir / "tiger_0p7_predictions.csv", index=False)

    # Align all methods for side-by-side dump
    aligned = hard.rename(
        columns={"y_pred_delta_log2_anchor_minus_query": "tiger_moe_pred_log2"}
    )[
        [
            "anchor_sequence",
            "query_sequence",
            "dataset",
            "family",
            "peptide_group",
            "similarity",
            "expert",
            "y_true_delta_log2_anchor_minus_query",
            "pred_0.3",
            "pred_0.7",
            "tiger_moe_pred_log2",
        ]
    ]
    aligned = aligned.merge(
        evo_pred[
            ["anchor_sequence", "query_sequence", "y_pred_delta_log2_anchor_minus_query"]
        ].rename(columns={"y_pred_delta_log2_anchor_minus_query": "evo_pred_log2"}),
        on=["anchor_sequence", "query_sequence"],
        how="inner",
    )
    aligned.to_csv(out_dir / "pair_predictions_aligned.csv", index=False)

    method_preds = {
        "TIGER MoE": hard,
        "TIGER 0.7": ens_07,
        "TIGER 0.3": ens_03,
        "EvoGradient": evo_pred,
    }
    leaderboard_rows: list[dict] = []
    for method in METHOD_ORDER:
        leaderboard_rows.extend(metrics_by_group(method_preds[method], method, name))

    summary = {
        "dataset": name,
        "n_pairs_input": int(len(pairs)),
        "n_overlap": int(len(aligned)),
        "moe_tau": moe_tau,
        "route_counts": hard["expert"].value_counts().to_dict(),
        "methods": {
            "TIGER MoE": hard_m,
            "TIGER 0.7": m_07,
            "TIGER 0.3": m_03,
            "EvoGradient": evo_m,
        },
        "protocol": {
            "tiger_moe": f"hard MoE only: similarity < {moe_tau} → expert 0.3, else → 0.7",
            "tiger_experts": "pair-delta model, 5-fold ensemble + calibrator",
            "evogradient": "absolute log10(MIC) ensemble → Δlog2 = pred(anchor)-pred(query)",
            "MAE": "log2MAE on pair Δlog2",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[{name} MoE hard] routes={summary['route_counts']} "
        f"PCC={hard_m.get('PCC', float('nan')):.4f} MAE={hard_m.get('MAE', float('nan')):.4f}",
        flush=True,
    )
    return leaderboard_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir-03", type=Path, default=DEFAULT_EXP_03)
    parser.add_argument("--exp-dir-07", type=Path, default=DEFAULT_EXP_07)
    parser.add_argument("--ll37-pair-csv", type=Path, default=LL37_PAIR_CSV)
    parser.add_argument("--apexgo-pair-csv", type=Path, default=APEX_PAIR_CSV)
    parser.add_argument(
        "--datasets",
        default="LL37,APEXGO",
        help="comma-separated subset: LL37,APEXGO",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--moe-tau",
        type=float,
        default=0.5,
        help="hard MoE gate: similarity < tau → expert 0.3, else → 0.7",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "checkpoints/ablation" / "04_similarity" / "external_eval_tiger_vs_evo",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    exp_03 = args.exp_dir_03 if args.exp_dir_03.is_absolute() else ROOT / args.exp_dir_03
    exp_07 = args.exp_dir_07 if args.exp_dir_07.is_absolute() else ROOT / args.exp_dir_07
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wanted = {x.strip().upper() for x in args.datasets.split(",") if x.strip()}

    all_rows: list[dict] = []
    summaries: dict = {
        "moe_tau": args.moe_tau,
        "tiger_expert_03": str(exp_03),
        "tiger_expert_07": str(exp_07),
        "methods": list(METHOD_ORDER),
        "metrics": ["PCC", "KCC", "MAE", "RSE"],
        "note": "MAE = log2MAE on pair Δlog2; hard MoE only (no soft blend)",
        "datasets": {},
    }

    dataset_specs = []
    if "LL37" in wanted:
        p = args.ll37_pair_csv if args.ll37_pair_csv.is_absolute() else ROOT / args.ll37_pair_csv
        dataset_specs.append(("LL37", *load_dataset("LL37", p)))
    if "APEXGO" in wanted:
        p = (
            args.apexgo_pair_csv
            if args.apexgo_pair_csv.is_absolute()
            else ROOT / args.apexgo_pair_csv
        )
        dataset_specs.append(("APEXGO", *load_dataset("APEXGO", p)))

    for name, pairs, pdb_bases in dataset_specs:
        rows, summary = evaluate_dataset(
            name,
            pairs,
            pdb_bases,
            out_dir / name.lower(),
            exp_03,
            exp_07,
            args.moe_tau,
            device,
        )
        all_rows.extend(rows)
        summaries["datasets"][name] = summary

    # Combined ALL (both families) overall metrics from aligned CSVs if both present
    combined_preds: dict[str, list[pd.DataFrame]] = {m: [] for m in METHOD_ORDER}
    for name, _, _ in dataset_specs:
        d = out_dir / name.lower()
        hard = pd.read_csv(d / "tiger_moe_hard_predictions.csv")
        p03 = pd.read_csv(d / "tiger_0p3_predictions.csv")
        p07 = pd.read_csv(d / "tiger_0p7_predictions.csv")
        evo = pd.read_csv(d / "evo" / "evo_pair_predictions.csv")
        combined_preds["TIGER MoE"].append(hard)
        combined_preds["TIGER 0.3"].append(p03)
        combined_preds["TIGER 0.7"].append(p07)
        combined_preds["EvoGradient"].append(evo)

    if len(dataset_specs) > 1:
        for method in METHOD_ORDER:
            cat = pd.concat(combined_preds[method], ignore_index=True)
            all_rows.append(_row("ALL (LL37+APEXGO)", method, _metrics(cat), dataset="ALL"))

    leaderboard = pd.DataFrame(all_rows)
    # Stable method order within each group
    leaderboard["method"] = pd.Categorical(
        leaderboard["method"], categories=list(METHOD_ORDER), ordered=True
    )
    leaderboard = leaderboard.sort_values(["dataset", "peptide_group", "method"]).reset_index(
        drop=True
    )
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)

    # Compact view: primary metrics only
    compact = leaderboard[
        ["dataset", "peptide_group", "method", "n", "PCC", "KCC", "MAE", "RSE"]
    ].copy()
    compact.to_csv(out_dir / "leaderboard_metrics.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))

    print("\n=== TIGER (hard MoE / 0.3 / 0.7) vs EvoGradient ===", flush=True)
    print(compact.to_string(index=False), flush=True)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
