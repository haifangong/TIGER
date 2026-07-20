"""Physicochemical / tabular feature computation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from .constants import AA_ORDER, CFU_COL, CFU_LEVELS, PRIMARY_CFU


def normalize_cfu_group(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    import re

    text = re.sub(r"\s+", " ", str(value).strip().upper())
    compact = text.replace(" ", "")
    aliases = {"1E4-1E5": "1E4 - 1E5", "1E5-1E6": "1E5 - 1E6", "1E6-1E7": "1E6 - 1E7"}
    return aliases.get(compact, text if text in CFU_LEVELS else "unknown")


def cfu_one_hot(value: Any) -> list[float]:
    out = [0.0] * len(CFU_LEVELS)
    out[CFU_LEVELS[normalize_cfu_group(value)]] = 1.0
    return out


def normalize_mod(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return "" if s in {"", "nan", "none", "free"} else s


def terminal_charge_adjustment(n_term: Any, c_term: Any) -> float:
    n = normalize_mod(n_term)
    c = normalize_mod(c_term)
    charge = 0.0
    if not any(k in n for k in ["ace", "acetyl", "pyro", "pglu", "formyl"]):
        charge += 1.0
    if not any(k in c for k in ["amd", "amid", "amide", "nh2", "methyl", "ester"]):
        charge -= 1.0
    return charge


def calculate_property(
    seq: str,
    n_term: Any = "",
    c_term: Any = "",
    scaling: str = "zscore",
) -> list[float]:
    analysed_seq = ProteinAnalysis(seq)
    aa_counts = analysed_seq.count_amino_acids()
    aliphatic_index = (aa_counts["A"] + 2.9 * aa_counts["V"] + 3.9 * (aa_counts["I"] + aa_counts["L"])) / len(seq)
    charge = analysed_seq.charge_at_pH(7.0) + terminal_charge_adjustment(n_term, c_term)
    charge_density = charge / len(seq)
    alpha_helix, beta_helix, turn_helix = analysed_seq.secondary_structure_fraction()
    raw = [
        round(analysed_seq.gravy(), 3),
        round(aliphatic_index, 3),
        round(analysed_seq.aromaticity(), 3),
        round(analysed_seq.instability_index(), 3),
        round(alpha_helix, 3),
        round(beta_helix, 3),
        round(turn_helix, 3),
        round(charge, 3),
        round(analysed_seq.isoelectric_point(), 3),
        round(charge_density, 3),
    ]
    if str(scaling).lower() in {"zscore", "raw", "none"}:
        return raw
    return [
        raw[0] * 10,
        raw[1] * 10,
        raw[2] * 10,
        raw[3],
        round(raw[4] * 10, 3),
        round(raw[5] * 10, 3),
        round(raw[6] * 10, 3),
        raw[7],
        raw[8],
        raw[9] * 10,
    ]


def tabular_features(
    df: pd.DataFrame,
    include_cfu: bool = True,
    property_scaling: str = "raw",
) -> np.ndarray:
    rows = []
    for r in df.itertuples():
        props = calculate_property(r.sequence, r.n_terminus, r.c_terminus, scaling=property_scaling)
        counts = ProteinAnalysis(r.sequence).count_amino_acids()
        aa_freq = [counts[a] / len(r.sequence) for a in AA_ORDER]
        cfu = cfu_one_hot(getattr(r, CFU_COL, "unknown")) if include_cfu and CFU_COL in df.columns else []
        rows.append([len(r.sequence)] + props + aa_freq + cfu)
    return np.asarray(rows, dtype=np.float32)
