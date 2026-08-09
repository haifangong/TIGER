#!/usr/bin/env python3
"""Nature-style one-page similarity figure: APEXGO, LL37, and internal_toxin_cohort.

Layout (3×2):
  a  max similarity to MIC holdout train set (histogram)
  b  cumulative max similarity to MIC train (ECDF)
  c  within-test pair similarity (histogram; APEXGO geo3 + LL37 neighbor-509)
  d  cumulative within-test pair similarity (ECDF)
  e  internal_toxin_cohort max similarity to toxin train set (histogram)
  f  internal_toxin_cohort cumulative max similarity to toxin train (ECDF)

Outputs (this directory):
  - pair_similarity_apexgo_vs_ll37.pdf
  - similarity_summary.csv
  - test_vs_train_max_similarity.csv
  - test_vs_train_similarity_summary.csv
  - internal_toxin_cohort_vs_toxin_max_similarity.csv
  - internal_toxin_cohort_vs_toxin_similarity_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # TIGER/  (…/data/test_external/<this>)
EXT = ROOT / "data/test_external"

APEX_PAIR_CSV = EXT / "test_activity_apexgo/pairs/pairs_template_centric_alldelta_geo3.csv"
APEX_PEPTIDE_CSV = EXT / "test_activity_apexgo/labels/apexgo_peptides_geo3.csv"
LL37_PAIR_CSV = EXT / "test_activity_ll37/ll37_pairs_neighbor.csv"
LL37_SEQ_CSV = EXT / "test_activity_ll37/ll37_sequences_mic.csv"
TRAIN_CSV = ROOT / "metadata/train_val_by_cfu_group_ug_per_mL.csv"
REMOVED_SIM_CSV = ROOT / "metadata/removed_similar_to_LL37_seq.csv"
INTERNAL_TOXIN_COHORT_CSV = EXT / "test_toxin_internal_toxin_cohort/internal_toxin_cohort_hemolysis_active_micmin_le128.csv"
TOXIN_TRAIN_CSV = (
    ROOT / "outputs" / "outputs_toxin_filter_thr512_both" / "toxicity_labeled_dataset.csv"
)

# Nature SI-friendly, colourblind-aware palette
C_APEX = "#0072B2"
C_LL37 = "#D55E00"
C_ITC = "#009E73"
C_THRESH = "#666666"
TRAIN_THRESH = 0.30
MIN_LEN, MAX_LEN = 6, 50


def setup_nature_style() -> None:
    # Ensure system Arial (msttcorefonts) is registered; matplotlib cache often misses it.
    arial_files = [
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/ariali.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/arialbi.ttf"),
    ]
    for fp in arial_files:
        if fp.exists():
            try:
                font_manager.fontManager.addfont(str(fp))
            except (RuntimeError, OSError, ValueError):
                pass

    arial_ok = any(f.name == "Arial" for f in font_manager.fontManager.ttflist)
    if not arial_ok:
        raise RuntimeError(
            "Arial font not available. Install msttcorefonts or place Arial.ttf on the system."
        )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,  # TrueType embed (editable text)
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.dpi": 150,
        }
    )


def setup_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    return aligner


def holdout_similarity(aligner: PairwiseAligner, a: str, b: str) -> float:
    """Holdout-style contract: score(a,b) / max(len(a), len(b))."""
    if not a or not b:
        return 0.0
    return round(float(aligner.score(a, b) / max(len(a), len(b))), 10)


def summarize(x: np.ndarray, name: str) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "dataset": name,
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "max": float(np.max(x)),
        "frac_ge_0.3": float((x >= 0.3).mean()),
        "frac_gt_0.3": float((x > 0.3).mean()),
        "frac_ge_0.5": float((x >= 0.5).mean()),
        "frac_ge_0.7": float((x >= 0.7).mean()),
    }


def load_unique_seqs(csv_path: Path, col: str = "sequence") -> list[str]:
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        for alt in ("Seq", "sequence", "SEQUENCE"):
            if alt in df.columns:
                col = alt
                break
    seqs = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .loc[lambda s: s.str.len().between(MIN_LEN, MAX_LEN) & s.str.isalpha()]
        .drop_duplicates()
        .tolist()
    )
    return sorted(seqs)


def max_sim_to_train(query_seqs: list[str], train_seqs: list[str]) -> pd.DataFrame:
    aligner = setup_aligner()
    train_set = set(train_seqs)
    rows = []
    for q in query_seqs:
        best = -1.0
        best_t = ""
        for t in train_seqs:
            sim = holdout_similarity(aligner, q, t)
            if sim > best:
                best = sim
                best_t = t
        rows.append(
            {
                "sequence": q,
                "len": len(q),
                "max_sim_to_train": best,
                "nearest_train_sequence": best_t,
                "exact_in_train": q in train_set,
            }
        )
    return pd.DataFrame(rows)


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def _style_ax(ax) -> None:
    ax.tick_params(length=2.5, pad=1.5)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)


def _hist_pair(ax, vals_a, vals_b, bins) -> None:
    ax.hist(
        vals_b,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.32,
        color=C_LL37,
        edgecolor=C_LL37,
        linewidth=0.7,
    )
    ax.hist(
        vals_a,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.32,
        color=C_APEX,
        edgecolor=C_APEX,
        linewidth=0.7,
    )
    ax.hist(vals_b, bins=bins, density=True, histtype="step", color=C_LL37, linewidth=0.85)
    ax.hist(vals_a, bins=bins, density=True, histtype="step", color=C_APEX, linewidth=0.85)


def _hist_one(ax, vals, bins, color) -> None:
    ax.hist(
        vals,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color=color,
        edgecolor=color,
        linewidth=0.7,
    )
    ax.hist(vals, bins=bins, density=True, histtype="step", color=color, linewidth=0.85)


def _ecdf(ax, vals, color) -> None:
    xs = np.sort(np.asarray(vals, dtype=float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color=color, lw=1.25, solid_capstyle="round")


def _annotate_n(ax, n_apex: int, n_ll37: int, unit: str) -> None:
    ax.text(
        0.98,
        0.96,
        f"APEXGO {unit} n={n_apex}\nLL-37 {unit} n={n_ll37}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color="#333333",
        linespacing=1.35,
    )


def _annotate_itc(ax, n_itc: int, n_toxin_train: int) -> None:
    ax.text(
        0.98,
        0.96,
        f"internal_toxin_cohort sequences n={n_itc}\ntoxin train n={n_toxin_train}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color="#333333",
        linespacing=1.35,
    )


def plot_nature_one_page(
    out_pdf: Path,
    sa_pair: np.ndarray,
    sl_pair: np.ndarray,
    sa_train: np.ndarray,
    sl_train: np.ndarray,
    sq_toxin: np.ndarray,
    n_toxin_train: int,
) -> None:
    """Single-page 3×2 figure."""
    setup_nature_style()

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.2, 7.8),
        gridspec_kw={"wspace": 0.30, "hspace": 0.38},
    )
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]
    ax_e, ax_f = axes[2, 0], axes[2, 1]

    n_apex_pair, n_ll37_pair = len(sa_pair), len(sl_pair)
    n_apex_seq, n_ll37_seq = len(sa_train), len(sl_train)
    n_itc = len(sq_toxin)

    # a — max sim to MIC train (histogram); no threshold line
    bins_train = np.linspace(0.15, 0.55, 21)
    _hist_pair(ax_a, sa_train, sl_train, bins_train)
    ax_a.set_xlim(0.15, 0.55)
    ax_a.set_xlabel("Max similarity to MIC training set")
    ax_a.set_ylabel("Density")
    _annotate_n(ax_a, n_apex_seq, n_ll37_seq, "sequences")
    _panel_label(ax_a, "a")
    _style_ax(ax_a)

    # b — max sim to MIC train (ECDF)
    _ecdf(ax_b, sa_train, C_APEX)
    _ecdf(ax_b, sl_train, C_LL37)
    ax_b.axvline(TRAIN_THRESH, color=C_THRESH, ls="--", lw=0.8, zorder=3)
    ax_b.set_xlim(0.15, 0.55)
    ax_b.set_ylim(0.0, 1.02)
    ax_b.set_xlabel("Max similarity to MIC training set")
    ax_b.set_ylabel("Cumulative fraction")
    _panel_label(ax_b, "b")
    _style_ax(ax_b)

    # c — pair similarity (histogram); no threshold
    bins_pair = np.linspace(0.0, 1.0, 29)
    _hist_pair(ax_c, sa_pair, sl_pair, bins_pair)
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_xlabel("Pair similarity")
    ax_c.set_ylabel("Density")
    _annotate_n(ax_c, n_apex_pair, n_ll37_pair, "pairs")
    _panel_label(ax_c, "c")
    _style_ax(ax_c)

    # d — pair similarity (ECDF); no threshold
    _ecdf(ax_d, sa_pair, C_APEX)
    _ecdf(ax_d, sl_pair, C_LL37)
    ax_d.set_xlim(0.0, 1.0)
    ax_d.set_ylim(0.0, 1.02)
    ax_d.set_xlabel("Pair similarity")
    ax_d.set_ylabel("Cumulative fraction")
    _panel_label(ax_d, "d")
    _style_ax(ax_d)

    # e — internal_toxin_cohort vs toxin train (histogram); no threshold line
    x_min = max(0.0, float(np.min(sq_toxin)) - 0.05)
    x_max = min(1.0, float(np.max(sq_toxin)) + 0.05)
    bins_itc = np.linspace(x_min, x_max, 21)
    _hist_one(ax_e, sq_toxin, bins_itc, C_ITC)
    ax_e.set_xlim(x_min, x_max)
    ax_e.set_xlabel("Max similarity to toxin training set")
    ax_e.set_ylabel("Density")
    _annotate_itc(ax_e, n_itc, n_toxin_train)
    _panel_label(ax_e, "e")
    _style_ax(ax_e)

    # f — internal_toxin_cohort vs toxin train (ECDF); no threshold line
    _ecdf(ax_f, sq_toxin, C_ITC)
    ax_f.set_xlim(x_min, x_max)
    ax_f.set_ylim(0.0, 1.02)
    ax_f.set_xlabel("Max similarity to toxin training set")
    ax_f.set_ylabel("Cumulative fraction")
    _panel_label(ax_f, "f")
    _style_ax(ax_f)

    handles = [
        Patch(facecolor=C_APEX, edgecolor=C_APEX, alpha=0.35, linewidth=0.8, label="APEXGO"),
        Patch(facecolor=C_LL37, edgecolor=C_LL37, alpha=0.35, linewidth=0.8, label="LL-37"),
        Patch(facecolor=C_ITC, edgecolor=C_ITC, alpha=0.35, linewidth=0.8, label="internal_toxin_cohort"),
        Line2D(
            [0],
            [0],
            color=C_THRESH,
            ls="--",
            lw=0.8,
            label="Threshold (0.30; b only)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.5,
        columnspacing=1.1,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.06, top=0.93)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _load_or_compute_max_sim(
    cache: Path,
    dataset: str,
    query_seqs: list[str],
    train_seqs: list[str],
) -> pd.DataFrame:
    if cache.exists():
        prev = pd.read_csv(cache)
        if "dataset" in prev.columns:
            sub = prev[prev["dataset"] == dataset]
            have = set(sub["sequence"].astype(str).str.upper())
            if set(query_seqs) <= have:
                print(f"[cache] reusing {cache.name} ({dataset})")
                return (
                    sub.set_index(sub["sequence"].str.upper())
                    .loc[query_seqs]
                    .reset_index(drop=True)
                )
    vs = max_sim_to_train(query_seqs, train_seqs)
    vs.insert(0, "dataset", dataset)
    return vs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=HERE / "pair_similarity_apexgo_vs_ll37.pdf",
        help="Output PDF path",
    )
    args = parser.parse_args()
    out_dir = args.out_pdf.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    apex = pd.read_csv(APEX_PAIR_CSV)
    ll37 = pd.read_csv(LL37_PAIR_CSV)
    sa_pair = apex["similarity"].astype(float).to_numpy()
    sl_pair = ll37["similarity"].astype(float).to_numpy()

    st_pair = pd.DataFrame([summarize(sa_pair, "APEXGO"), summarize(sl_pair, "LL37")])
    st_pair.to_csv(out_dir / "similarity_summary.csv", index=False)
    print(st_pair.to_string(index=False))

    train_seqs = load_unique_seqs(TRAIN_CSV, "sequence")
    if REMOVED_SIM_CSV.exists():
        removed_keys = set(
            pd.read_csv(REMOVED_SIM_CSV)["sequence"].astype(str).str.strip().str.upper()
        )
        overlap = removed_keys & set(train_seqs)
        print(
            f"[train] unique seqs={len(train_seqs)}; "
            f"removed-sim seqs={len(removed_keys)}; still-in-train={len(overlap)}"
        )
        if overlap:
            raise RuntimeError(
                f"Holdout removed sequences still present in train CSV: {len(overlap)}"
            )

    apex_seqs = load_unique_seqs(APEX_PEPTIDE_CSV, "sequence")
    ll37_seqs = load_unique_seqs(LL37_SEQ_CSV, "sequence")
    print(f"[apex] unique seqs={len(apex_seqs)}  [ll37] unique seqs={len(ll37_seqs)}")

    cache = out_dir / "test_vs_train_max_similarity.csv"
    apex_vs = _load_or_compute_max_sim(cache, "APEXGO", apex_seqs, train_seqs)
    ll37_vs = _load_or_compute_max_sim(cache, "LL37", ll37_seqs, train_seqs)
    vs_all = pd.concat([apex_vs, ll37_vs], ignore_index=True)
    vs_all.to_csv(cache, index=False)

    sa_train = apex_vs["max_sim_to_train"].to_numpy(dtype=float)
    sl_train = ll37_vs["max_sim_to_train"].to_numpy(dtype=float)
    st_vs = pd.DataFrame(
        [
            summarize(sa_train, "APEXGO_vs_train"),
            summarize(sl_train, "LL37_vs_train"),
        ]
    )
    st_vs.to_csv(out_dir / "test_vs_train_similarity_summary.csv", index=False)
    print(st_vs.to_string(index=False))

    # internal_toxin_cohort (88) vs toxin training set (1596 under thr=512)
    itc_seqs = load_unique_seqs(INTERNAL_TOXIN_COHORT_CSV, "sequence")
    toxin_seqs = load_unique_seqs(TOXIN_TRAIN_CSV, "sequence")
    print(f"[internal_toxin_cohort] unique seqs={len(itc_seqs)}  [toxin train] unique seqs={len(toxin_seqs)}")
    if len(itc_seqs) != 88:
        print(f"[warn] expected 88 internal_toxin_cohort sequences, got {len(itc_seqs)}")

    itc_cache = out_dir / "internal_toxin_cohort_vs_toxin_max_similarity.csv"
    itc_vs = _load_or_compute_max_sim(itc_cache, "internal_toxin_cohort", itc_seqs, toxin_seqs)
    itc_vs.to_csv(itc_cache, index=False)
    sq = itc_vs["max_sim_to_train"].to_numpy(dtype=float)
    st_itc = pd.DataFrame([summarize(sq, "internal_toxin_cohort_vs_toxin_train")])
    st_itc.to_csv(out_dir / "internal_toxin_cohort_vs_toxin_similarity_summary.csv", index=False)
    print(st_itc.to_string(index=False))
    print(f"[exact in toxin train] internal_toxin_cohort={int(itc_vs['exact_in_train'].sum())}")

    plot_nature_one_page(
        args.out_pdf,
        sa_pair,
        sl_pair,
        sa_train,
        sl_train,
        sq,
        n_toxin_train=len(toxin_seqs),
    )
    print(f"wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
