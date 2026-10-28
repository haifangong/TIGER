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


# Eisenberg consensus hydrophobicity (standard scale for hydrophobic moments).
EISENBERG_HYDRO = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}


def hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> float:
    """Normalized hydrophobic moment (Eisenberg): |sum s_i * e^{i*i*theta}| / n.

    angle 100 deg/residue -> alpha-helical moment (helical-face amphipathicity);
    160 deg -> beta-strand moment. Captures the *distribution pattern* of
    hydrophobic residues on the helix face, which bulk GRAVY cannot represent.
    """
    import math as _math

    theta = _math.radians(angle_deg)
    x = y = 0.0
    for i, aa in enumerate(seq):
        s = EISENBERG_HYDRO.get(aa, 0.0)
        x += s * _math.cos(i * theta)
        y += s * _math.sin(i * theta)
    return _math.hypot(x, y) / max(1, len(seq))


def hydrophobic_moments(seq: str) -> list[float]:
    """[helical(100deg) moment, beta(160deg) moment] — appended to p_f when
    ``Config.p_feature_hmom`` is on (see dataloader.construct_graph)."""
    return [round(hydrophobic_moment(seq, 100.0), 4), round(hydrophobic_moment(seq, 160.0), 4)]


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


def load_aa_prior_table(feature_path: str):
    """Parse the handcrafted per-AA descriptor file -> {letter: [20 floats]}."""
    import torch as _torch

    seen = {}
    with open(feature_path) as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if len(parts[1:]) == 20:
                seen[parts[0]] = [float(v) for v in parts[1:]]
    return seen


def aa_prior_table(feature_path: str):
    """Cached 21x20 float tensor; ids follow the sequential 1..20 alphabet."""
    import torch as _torch
    from functools import lru_cache

    @lru_cache(maxsize=4)
    def _load(path):
        seen = load_aa_prior_table(path)
        # sequential ids A=1..Y=20 alphabetical (same as encode_sequence)
        letters = sorted(seen.keys())
        table = _torch.zeros(21, 20)
        for i, aa in enumerate(letters, start=1):
            table[i] = _torch.tensor(seen[aa])
        return table

    return _load(str(feature_path))


def aa_prior_projected(feature_path: str, emb_dim: int, seed: int = 0):
    """21 x emb_dim prior: 20-dim descriptor through a FIXED random orthonormal
    projection (seeded, distance-preserving); pad row stays zero."""
    import torch as _torch

    base = aa_prior_table(feature_path)  # 21 x 20
    g = _torch.Generator().manual_seed(seed)
    a = _torch.randn(emb_dim, base.shape[1], generator=g)  # emb_dim x 20
    q, _ = _torch.linalg.qr(a)  # emb_dim x 20, orthonormal columns
    return base @ q.t()  # 21 x emb_dim
