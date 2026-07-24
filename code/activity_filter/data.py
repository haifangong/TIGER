"""Load MIC activity labels with inequality-aware filtering.

Binary activity-filter rule (threshold T, default 128 ug/mL):
  label = 1 (inactive / filter-out) if MIC is determinably >= T
  label = 0 (active / keep)         if MIC is determinably <  T

Per-assay bound rules (after unit conversion to ug/mL):
  - exact value x:        inactive if x >= T else active
  - lower-censored >c:    inactive only if c >= T; else AMBIGUOUS
  - upper-censored <c:    active only if c <= T; else AMBIGUOUS
  - range [a,b]:          inactive if a >= T; active if b < T; else AMBIGUOUS

Aggregation across bacterial MIC assays for one peptide:
  1. If any exact MIC values exist, use mic_min = min(exact) for the label.
  2. Else use inequality-only evidence: active if any assay is determinably
     active; inactive if every determinable assay is inactive; else drop.
  3. Conflicting hard labels (active vs inactive) → drop.

Only monomer peptides with standard amino acids are retained.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from code.toxin_filter.data import (
    STANDARD_AA,
    nested_name,
    normalize_unit,
    parse_concentration_bound,
    resolve_default_json_dir,
    sequence_mw,
    to_ug_per_ml,
)

MIC_MEASURE_RE = re.compile(r"^\s*MIC(\d+)?\s*$", re.I)


def is_mic_like(activity: dict[str, Any]) -> bool:
    """Keep MIC / MIC50 / MIC90 style measures (not MBC/IC50/etc.)."""
    group = nested_name(activity.get("activityMeasureGroup"))
    value = str(activity.get("activityMeasureValue") or "")
    text = f"{group} {value}".strip()
    if not text:
        return False
    # Accept MIC, MIC50, MIC90; reject MBC/MIC-like non-MIC strings.
    if MIC_MEASURE_RE.match(group) or MIC_MEASURE_RE.match(value):
        return True
    return bool(re.search(r"\bMIC(\d+)?\b", text, re.I)) and "MBC" not in text.upper()


def label_from_mic_bound(
    bound: dict[str, Any],
    *,
    unit: str,
    mw: float | None,
    threshold: float,
) -> tuple[int | None, str, float | None]:
    """Return (label or None, reason, exact_ug_or_None). label: 1=inactive, 0=active."""
    kind = bound["kind"]
    if kind == "exact":
        x = to_ug_per_ml(bound["value"], unit, mw)
        if x is None or not math.isfinite(x) or x <= 0:
            return None, "bad_exact", None
        return (1 if x >= threshold else 0), "exact", float(x)
    if kind == "gt":
        c = to_ug_per_ml(bound["value"], unit, mw)
        if c is None or not math.isfinite(c) or c < 0:
            return None, "unit_convert_fail", None
        # MIC > c. Determinable inactive only if lower bound already >= T.
        if c >= threshold:
            return 1, "gt_deterministic", None
        return None, "gt_ambiguous", None
    if kind == "lt":
        c = to_ug_per_ml(bound["value"], unit, mw)
        if c is None or not math.isfinite(c) or c <= 0:
            return None, "unit_convert_fail", None
        # MIC < c. Determinable active only if upper bound itself is <= T
        # (strictly: MIC < c <= T ⇒ MIC < T). Use c <= T for inclusive threshold.
        if c <= threshold:
            return 0, "lt_deterministic", None
        return None, "lt_ambiguous", None
    if kind == "range":
        lo = to_ug_per_ml(bound["low"], unit, mw)
        hi = to_ug_per_ml(bound["high"], unit, mw)
        if lo is None or hi is None:
            return None, "unit_convert_fail", None
        if lo >= threshold:
            return 1, "range_inactive", None
        if hi < threshold:
            return 0, "range_active", None
        return None, "range_ambiguous", None
    return None, "unknown_bound", None


def load_activity_table(
    *,
    json_dir: str | Path | None = None,
    csv_path: str | Path | None = None,
    threshold: float = 128.0,
    min_len: int = 6,
    max_len: int = 50,
    source: str = "json",
) -> pd.DataFrame:
    """Build inequality-aware binary MIC activity labels.

    Parameters
    ----------
    source:
      - ``json`` (default): parse DBAASP ``targetActivities`` MIC assays
      - ``csv``: legacy numeric ug/mL MIC table (inequalities already stripped)
    """
    if source == "csv" or csv_path is not None:
        return _load_from_numeric_csv(
            csv_path,
            threshold=threshold,
            min_len=min_len,
            max_len=max_len,
        )
    return _load_from_json(
        json_dir,
        threshold=threshold,
        min_len=min_len,
        max_len=max_len,
    )


def _load_from_json(
    json_dir: str | Path | None,
    *,
    threshold: float,
    min_len: int,
    max_len: int,
) -> pd.DataFrame:
    root = Path(json_dir) if json_dir else resolve_default_json_dir()
    per_seq: dict[str, dict[str, Any]] = {}
    stats: dict[str, int] = defaultdict(int)

    for path in sorted(root.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if nested_name(data.get("complexity")) != "Monomer":
            continue
        seq = str(data.get("sequence") or "").upper().strip()
        if not STANDARD_AA.match(seq) or not (min_len <= len(seq) <= max_len):
            continue
        n_term = nested_name(data.get("nTerminus"))
        c_term = nested_name(data.get("cTerminus"))
        mw = sequence_mw(seq, n_term, c_term)

        hard_labels: list[int] = []
        exact_vals: list[float] = []
        reasons: list[str] = []
        raws: list[str] = []

        for activity in data.get("targetActivities") or []:
            if not is_mic_like(activity):
                continue
            stats["n_mic_activities"] += 1
            unit = normalize_unit(nested_name(activity.get("unit")))
            if not unit:
                stats["drop_bad_unit"] += 1
                continue
            bound = parse_concentration_bound(activity.get("concentration"))
            if bound is None:
                stats["drop_bad_conc"] += 1
                continue
            label, reason, exact = label_from_mic_bound(
                bound, unit=unit, mw=mw, threshold=threshold
            )
            reasons.append(reason)
            raws.append(f"{bound['kind']}:{bound.get('raw')}")
            if label is None:
                stats[f"ambiguous_{reason}"] += 1
                continue
            stats[f"keep_{reason}"] += 1
            hard_labels.append(int(label))
            if exact is not None:
                exact_vals.append(float(exact))

        if not hard_labels and not exact_vals:
            if reasons:
                stats["peptides_all_ambiguous"] += 1
            continue

        # Prefer min exact MIC when available.
        if exact_vals:
            mic_min = min(exact_vals)
            label = 1 if mic_min >= threshold else 0
        else:
            uniq = set(hard_labels)
            if len(uniq) > 1:
                stats["peptides_conflict"] += 1
                continue
            label = hard_labels[0]
            mic_min = float("nan")

        entry = per_seq.get(seq)
        if entry is None:
            per_seq[seq] = {
                "sequence": seq,
                "n_terminus": n_term,
                "c_terminus": c_term,
                "label": int(label),
                "exact_vals": list(exact_vals),
                "n_determinable": len(hard_labels),
                "evidence": list(raws),
            }
            stats["peptides_kept"] += 1
        else:
            # Merge: combine exact MICs; recompute label from pooled exact min.
            entry["exact_vals"].extend(exact_vals)
            entry["n_determinable"] += len(hard_labels)
            entry["evidence"].extend(raws)
            if entry["exact_vals"]:
                mic_min = min(entry["exact_vals"])
                new_label = 1 if mic_min >= threshold else 0
            else:
                new_label = int(label)
            if entry["label"] != new_label and not entry["exact_vals"]:
                # inequality-only conflict across JSON records
                stats["peptides_conflict"] += 1
                del per_seq[seq]
                continue
            entry["label"] = new_label

    rows = []
    for seq, entry in per_seq.items():
        mic = mean(entry["exact_vals"]) if entry["exact_vals"] else float("nan")
        mic_min = min(entry["exact_vals"]) if entry["exact_vals"] else float("nan")
        rows.append(
            {
                "sequence": seq,
                "n_terminus": entry["n_terminus"],
                "c_terminus": entry["c_terminus"],
                "label": int(entry["label"]),
                "mic_mean_exact": mic,
                "mic_min_exact": mic_min,
                "n_determinable_assays": int(entry["n_determinable"]),
                "evidence": ";".join(entry["evidence"][:12]),
            }
        )
    df = pd.DataFrame(rows).reset_index(drop=True)
    if df.empty or df["label"].nunique() < 2:
        raise ValueError(
            f"Need both classes after inequality-aware MIC filtering. stats={dict(stats)}"
        )
    df.attrs["filter_stats"] = dict(stats)
    df.attrs["threshold"] = float(threshold)
    df.attrs["json_dir"] = str(root)
    df.attrs["label_positive"] = "inactive_MIC_ge_threshold"
    return df


def _load_from_numeric_csv(
    csv_path: str | Path | None,
    *,
    threshold: float,
    min_len: int,
    max_len: int,
) -> pd.DataFrame:
    """Legacy path: numeric ug/mL MIC columns (inequalities already stripped)."""
    candidates = [
        Path(csv_path) if csv_path else None,
        Path(__file__).resolve().parents[3] / "newdata" / "dbaasp_amp_training_ug_per_mL.csv",
        Path(__file__).resolve().parents[2] / "metadata" / "train_val_ug_per_mL.csv",
    ]
    path = next((p for p in candidates if p is not None and p.exists()), None)
    if path is None:
        raise FileNotFoundError("No numeric MIC CSV found.")

    df = pd.read_csv(path, encoding="latin1")
    mic_cols = [
        c
        for c in df.columns
        if c.startswith("MIC_")
        and not any(s in c for s in ("_media", "_cfu", "_cfu_groups"))
    ]
    if not mic_cols:
        raise ValueError(f"No MIC_* columns in {path}")

    work = df.copy()
    work["sequence"] = work["sequence"].astype(str).str.upper().str.strip()
    if "n_terminus" not in work.columns:
        work["n_terminus"] = ""
    if "c_terminus" not in work.columns:
        work["c_terminus"] = ""
    for c in mic_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    def row_min(row) -> float:
        vals = [float(row[c]) for c in mic_cols if pd.notna(row[c]) and float(row[c]) > 0]
        return min(vals) if vals else float("nan")

    work["mic_min_exact"] = work.apply(row_min, axis=1)
    work = work.dropna(subset=["mic_min_exact"])
    work = work[work["sequence"].str.len().between(min_len, max_len)]
    work = work[work["sequence"].map(lambda s: bool(STANDARD_AA.match(s)))]

    agg = (
        work.groupby("sequence", as_index=False)
        .agg(
            mic_min_exact=("mic_min_exact", "min"),
            n_terminus=("n_terminus", "first"),
            c_terminus=("c_terminus", "first"),
        )
        .reset_index(drop=True)
    )
    agg["label"] = (agg["mic_min_exact"] >= float(threshold)).astype(int)
    agg["mic_mean_exact"] = agg["mic_min_exact"]
    agg["n_determinable_assays"] = 1
    agg["evidence"] = "csv_min_mic"
    agg.attrs["threshold"] = float(threshold)
    agg.attrs["csv_path"] = str(path)
    agg.attrs["label_positive"] = "inactive_MIC_ge_threshold"
    return agg


def summarize_label_balance(df: pd.DataFrame) -> dict:
    y = df["label"].to_numpy(dtype=int)
    out = {
        "n": int(len(y)),
        "n_inactive": int((y == 1).sum()),
        "n_active": int((y == 0).sum()),
        "inactive_frac": float(y.mean()) if len(y) else float("nan"),
        "seq_len_mean": float(df["sequence"].str.len().mean()) if len(df) else float("nan"),
        "seq_len_max": int(df["sequence"].str.len().max()) if len(df) else 0,
        "threshold": float(df.attrs.get("threshold", float("nan"))),
        "label_positive": df.attrs.get("label_positive", "inactive_MIC_ge_threshold"),
    }
    if "filter_stats" in df.attrs:
        out["filter_stats"] = df.attrs["filter_stats"]
    return out
