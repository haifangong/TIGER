"""Load HC50 hemolysis labels with inequality-aware filtering.

Binary main-task rule (threshold T, default 512 ug/mL):
  - exact value x:        toxin if x <= T else non-toxin
  - lower-censored >c:    non-toxin only if c >= T; else AMBIGUOUS (drop)
  - upper-censored <c:    toxin only if c <= T; else AMBIGUOUS (drop)
  - range [a,b]:          toxin if b <= T; non-toxin if a > T; else AMBIGUOUS

Only peptides with at least one determinable assay and no conflicting hard
labels are retained for classification.
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

STANDARD_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

DEFAULT_JSON_DIR = Path("/data4T/ubuntu/wangyue/postdoc_2025/POAP/Data/dbaasp_jsons")

RESIDUE_MASS = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "E": 129.1155, "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}
WATER_MASS = 18.01528


def nested_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return "" if value is None else str(value)


def sequence_mw(sequence: str, n_terminus: str = "", c_terminus: str = "") -> float | None:
    sequence = re.sub(r"[^A-Za-z]", "", sequence or "").upper()
    if not sequence or any(aa not in RESIDUE_MASS for aa in sequence):
        return None
    mw = sum(RESIDUE_MASS[aa] for aa in sequence) + WATER_MASS
    n = (n_terminus or "").lower()
    c = (c_terminus or "").lower()
    if "ace" in n or "acetyl" in n:
        mw += 42.0367
    if "amd" in c or "amid" in c or "nh2" in c:
        mw -= 0.9840
    return mw


def normalize_unit(unit: str) -> str | None:
    unit = (unit or "").strip().lower().replace("μ", "µ").replace(" ", "")
    if unit in {"µm", "um", "μm"}:
        return "uM"
    if unit in {"µg/ml", "ug/ml", "µg/ml".lower(), "ug/ml"}:
        return "ug_per_mL"
    return None


def to_ug_per_ml(value: float, unit: str, mw: float | None) -> float | None:
    if unit == "ug_per_mL":
        return value
    if unit == "uM":
        if not mw or mw <= 0:
            return None
        return value * mw / 1000.0
    return None


def is_hc50_like(activity: dict[str, Any]) -> bool:
    cell = nested_name(activity.get("targetCell")).lower()
    if "erythrocyte" not in cell or "human" not in cell:
        return False
    text = " ".join(
        [
            nested_name(activity.get("activityMeasureForLysisGroup")),
            str(activity.get("activityMeasureForLysisValue") or ""),
        ]
    )
    return bool(re.search(r"\bHC50\b|50\s*%\s*Hemolysis|50-60%\s*Hemolysis", text, re.I))


def parse_concentration_bound(concentration: Any) -> dict[str, Any] | None:
    """Parse DBAASP concentration into exact / gt / lt / range bound."""
    text = str(concentration or "").replace(",", ".")
    if not text.strip() or text.strip().upper() in {"NA", "N/A", "NONE", "-"}:
        return None
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    compact = re.sub(r"\s+", "", text)

    # >= or >  (lower censoring: true value is above the reported number)
    m = re.match(rf"^(>=|>|≥)\s*({NUMBER})$", compact)
    if m:
        return {"kind": "gt", "value": float(m.group(2)), "raw": text}

    # <= or <  (upper censoring)
    m = re.match(rf"^(<=|<|≤)\s*({NUMBER})$", compact)
    if m:
        return {"kind": "lt", "value": float(m.group(2)), "raw": text}

    # range a-b
    m = re.search(rf"(?<![A-Za-z0-9.])({NUMBER})-({NUMBER})(?![A-Za-z0-9.])", compact)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return {"kind": "range", "low": lo, "high": hi, "raw": text}

    # exact (ignore trailing uncertainty markers by taking first number)
    if compact[0:1] in {"<", ">", "≥", "≤"}:
        return None
    m = re.search(rf"{NUMBER}", compact)
    if not m:
        return None
    value = float(m.group())
    if not math.isfinite(value) or value < 0:
        return None
    return {"kind": "exact", "value": value, "raw": text}


def label_from_bound(
    bound: dict[str, Any],
    *,
    unit: str,
    mw: float | None,
    threshold: float,
) -> tuple[int | None, str]:
    """Return (label or None, reason). label: 1=toxin, 0=non-toxin."""
    kind = bound["kind"]
    if kind == "exact":
        x = to_ug_per_ml(bound["value"], unit, mw)
        if x is None:
            return None, "unit_convert_fail"
        return (1 if x <= threshold else 0), "exact"
    if kind == "gt":
        c = to_ug_per_ml(bound["value"], unit, mw)
        if c is None:
            return None, "unit_convert_fail"
        # HC50 > c. Determinable non-toxin only if even the lower bound is > T,
        # i.e. c >= T (conservative: >T also implies non-toxin when c >= T).
        if c >= threshold:
            return 0, "gt_deterministic"
        return None, "gt_ambiguous"
    if kind == "lt":
        c = to_ug_per_ml(bound["value"], unit, mw)
        if c is None:
            return None, "unit_convert_fail"
        # HC50 < c. Determinable toxin only if upper bound itself is <= T.
        if c <= threshold:
            return 1, "lt_deterministic"
        return None, "lt_ambiguous"
    if kind == "range":
        lo = to_ug_per_ml(bound["low"], unit, mw)
        hi = to_ug_per_ml(bound["high"], unit, mw)
        if lo is None or hi is None:
            return None, "unit_convert_fail"
        if hi <= threshold:
            return 1, "range_toxin"
        if lo > threshold:
            return 0, "range_nontoxin"
        return None, "range_ambiguous"
    return None, "unknown_bound"


def resolve_default_json_dir() -> Path:
    candidates = [
        DEFAULT_JSON_DIR,
        Path(__file__).resolve().parents[3] / "POAP" / "Data" / "dbaasp_jsons",
        Path("/data4T/ubuntu/wangyue/postdoc_2025/POAP/Data/dbaasp_jsons"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "DBAASP JSON dir not found. Pass --json-dir. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_toxicity_table(
    csv_path: str | Path | None = None,
    *,
    json_dir: str | Path | None = None,
    threshold: float = 512.0,
    min_len: int = 6,
    max_len: int = 50,
    source: str = "json",
) -> pd.DataFrame:
    """Build inequality-aware binary toxicity labels.

    Parameters
    ----------
    source:
      - ``json`` (default): parse raw DBAASP exports with censoring rules
      - ``csv``: legacy numeric CSV (inequalities already stripped; not recommended)
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
    stats = defaultdict(int)

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

        for activity in data.get("hemoliticCytotoxicActivities") or []:
            if not is_hc50_like(activity):
                continue
            stats["n_hc50_activities"] += 1
            unit = normalize_unit(nested_name(activity.get("unit")))
            if not unit:
                stats["drop_bad_unit"] += 1
                continue
            bound = parse_concentration_bound(activity.get("concentration"))
            if bound is None:
                stats["drop_bad_conc"] += 1
                continue
            label, reason = label_from_bound(bound, unit=unit, mw=mw, threshold=threshold)
            reasons.append(reason)
            raws.append(f"{bound['kind']}:{bound.get('raw')}")
            if label is None:
                stats[f"ambiguous_{reason}"] += 1
                continue
            stats[f"keep_{reason}"] += 1
            hard_labels.append(int(label))
            if bound["kind"] == "exact":
                x = to_ug_per_ml(bound["value"], unit, mw)
                if x is not None:
                    exact_vals.append(x)

        if not hard_labels:
            if reasons:
                stats["peptides_all_ambiguous"] += 1
            continue
        if len(set(hard_labels)) > 1:
            stats["peptides_conflict"] += 1
            continue

        label = hard_labels[0]
        entry = per_seq.get(seq)
        if entry is None:
            per_seq[seq] = {
                "sequence": seq,
                "n_terminus": n_term,
                "c_terminus": c_term,
                "label": label,
                "exact_vals": list(exact_vals),
                "n_determinable": len(hard_labels),
                "evidence": list(raws),
            }
            stats["peptides_kept"] += 1
        else:
            if entry["label"] != label:
                stats["peptides_conflict"] += 1
                del per_seq[seq]
                continue
            entry["exact_vals"].extend(exact_vals)
            entry["n_determinable"] += len(hard_labels)
            entry["evidence"].extend(raws)

    rows = []
    for seq, entry in per_seq.items():
        hc50 = mean(entry["exact_vals"]) if entry["exact_vals"] else float("nan")
        rows.append(
            {
                "sequence": seq,
                "n_terminus": entry["n_terminus"],
                "c_terminus": entry["c_terminus"],
                "label": int(entry["label"]),
                "hc50": hc50,
                "n_determinable_assays": int(entry["n_determinable"]),
                "evidence": ";".join(entry["evidence"][:12]),
            }
        )
    df = pd.DataFrame(rows).reset_index(drop=True)
    if df.empty or df["label"].nunique() < 2:
        raise ValueError(
            f"Need both classes after inequality-aware filtering. stats={dict(stats)}"
        )
    df.attrs["filter_stats"] = dict(stats)
    df.attrs["threshold"] = float(threshold)
    df.attrs["json_dir"] = str(root)
    return df


def _load_from_numeric_csv(
    csv_path: str | Path | None,
    *,
    threshold: float,
    min_len: int,
    max_len: int,
    hc50_col: str = "HC50_Human_erythrocytes",
) -> pd.DataFrame:
    """Legacy path: numeric CSV where inequalities were already stripped."""
    candidates = [
        Path(csv_path) if csv_path else None,
        Path(__file__).resolve().parents[2] / "metadata" / "train_val_ug_per_mL.csv",
        Path(__file__).resolve().parents[3] / "newdata" / "dbaasp_amp_training_ug_per_mL.csv",
    ]
    path = next((p for p in candidates if p is not None and p.exists()), None)
    if path is None:
        raise FileNotFoundError("No numeric HC50 CSV found.")
    df = pd.read_csv(path, encoding="latin1")
    work = df[["sequence", hc50_col]].copy()
    work["n_terminus"] = df["n_terminus"] if "n_terminus" in df.columns else ""
    work["c_terminus"] = df["c_terminus"] if "c_terminus" in df.columns else ""
    work["sequence"] = work["sequence"].astype(str).str.upper().str.strip()
    work[hc50_col] = pd.to_numeric(work[hc50_col], errors="coerce")
    work = work.dropna(subset=[hc50_col])
    work = work[work[hc50_col] > 0]
    work = work[work["sequence"].str.len().between(min_len, max_len)]
    work = work[work["sequence"].map(lambda s: bool(STANDARD_AA.match(s)))]
    agg = work.groupby("sequence", as_index=False).agg(
        {hc50_col: "mean", "n_terminus": "first", "c_terminus": "first"}
    )
    agg["label"] = (agg[hc50_col] <= float(threshold)).astype(int)
    agg["hc50"] = agg[hc50_col]
    agg = agg.drop(columns=[hc50_col]).reset_index(drop=True)
    return agg


def summarize_label_balance(df: pd.DataFrame) -> dict:
    y = df["label"].to_numpy(dtype=int)
    out = {
        "n": int(len(y)),
        "n_toxin": int((y == 1).sum()),
        "n_non_toxin": int((y == 0).sum()),
        "toxin_frac": float(y.mean()) if len(y) else float("nan"),
        "seq_len_mean": float(df["sequence"].str.len().mean()) if len(df) else float("nan"),
        "seq_len_max": int(df["sequence"].str.len().max()) if len(df) else 0,
        "threshold": float(df.attrs.get("threshold", float("nan"))),
    }
    if "filter_stats" in df.attrs:
        out["filter_stats"] = df.attrs["filter_stats"]
    return out
