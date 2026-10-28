"""Sequence + global physicochemical features (no structure)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from torch.utils.data import Dataset

# Alphabetical 1..20 (TIGER convention); pad=0
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}

# POAP toxicity FusionPeptide one-hot layout (21 channels, last = X)
POAP_AMAS = {
    "G": 20, "A": 1, "V": 2, "L": 3, "I": 4, "P": 5, "F": 6, "Y": 7, "W": 8,
    "S": 9, "T": 10, "C": 11, "M": 12, "N": 13, "Q": 14, "D": 15, "E": 16,
    "K": 17, "R": 18, "H": 19, "X": 21,
}

GLOBAL_FEATURE_NAMES = [
    "gravy",
    "aliphatic_index",
    "aromaticity",
    "instability_index",
    "helix_frac",
    "sheet_frac",
    "turn_frac",
    "charge_pH7",
    "isoelectric_point",
    "charge_density",
]


def _normalize_mod(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return "" if s in {"", "nan", "none", "free"} else s


def terminal_charge_adjustment(n_term: Any, c_term: Any) -> float:
    n = _normalize_mod(n_term)
    c = _normalize_mod(c_term)
    charge = 0.0
    if not any(k in n for k in ["ace", "acetyl", "pyro", "pglu", "formyl"]):
        charge += 1.0
    if not any(k in c for k in ["amd", "amid", "amide", "nh2", "methyl", "ester"]):
        charge -= 1.0
    return charge


def calculate_global_properties(
    seq: str,
    n_term: Any = "",
    c_term: Any = "",
) -> np.ndarray:
    """Raw 10-D physicochemical vector (z-scored later on train folds)."""
    pa = ProteinAnalysis(seq)
    aa_counts = pa.count_amino_acids()
    length = max(len(seq), 1)
    aliphatic = (aa_counts["A"] + 2.9 * aa_counts["V"] + 3.9 * (aa_counts["I"] + aa_counts["L"])) / length
    charge = pa.charge_at_pH(7.0) + terminal_charge_adjustment(n_term, c_term)
    helix, sheet, turn = pa.secondary_structure_fraction()
    return np.asarray(
        [
            pa.gravy(),
            aliphatic,
            pa.aromaticity(),
            pa.instability_index(),
            helix,
            sheet,
            turn,
            charge,
            pa.isoelectric_point(),
            charge / length,
        ],
        dtype=np.float64,
    )


def aa_composition(seq: str) -> np.ndarray:
    counts = ProteinAnalysis(seq).count_amino_acids()
    length = max(len(seq), 1)
    return np.asarray([counts[a] / length for a in AA_ORDER], dtype=np.float64)


def build_tabular_features(df: pd.DataFrame, feature_mode: str = "both") -> np.ndarray:
    """Tabular ML features.

    feature_mode
    ------------
    both     : length + 10 global physicochemical props + 20 AA frequencies
    global   : 10 global physicochemical props only
    sequence : length + 20 AA frequencies only
    """
    mode = str(feature_mode).lower()
    if mode not in {"both", "global", "sequence"}:
        raise ValueError(f"Unsupported feature_mode={feature_mode!r}")
    rows = []
    for r in df.itertuples():
        props = calculate_global_properties(r.sequence, r.n_terminus, r.c_terminus)
        aa = aa_composition(r.sequence)
        length = np.asarray([len(r.sequence)], dtype=np.float64)
        if mode == "global":
            rows.append(props)
        elif mode == "sequence":
            rows.append(np.concatenate([length, aa]))
        else:
            rows.append(np.concatenate([length, props, aa]))
    return np.asarray(rows, dtype=np.float64)


FEATURE_MODE_DIM = {
    "global": 10,
    "sequence": 21,  # length + 20 AA freqs
    "both": 31,      # length + 10 props + 20 AA freqs
}


def encode_sequence_integer(seq: str, max_len: int) -> np.ndarray:
    ids = [AA_TO_IDX[ch] for ch in seq[:max_len]]
    out = np.zeros(max_len, dtype=np.int64)
    out[: len(ids)] = ids
    return out


def encode_sequence_poap_onehot(seq: str, max_len: int) -> np.ndarray:
    emb = np.zeros((max_len, 21), dtype=np.float32)
    for pos, aa in enumerate(seq[:max_len]):
        emb[pos, POAP_AMAS[aa]] = 1.0
    return emb


def fit_zscore(X: np.ndarray) -> dict[str, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std}


def apply_zscore(X: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return (X - stats["mean"]) / stats["std"]


class ToxinDataset(Dataset):
    """Sequence + global-feature dataset for DL toxin classifiers."""

    def __init__(
        self,
        sequences: list[str],
        global_f: np.ndarray,
        labels: np.ndarray,
        max_len: int = 50,
        seq_style: str = "poap",
    ):
        self.sequences = list(sequences)
        self.global_f = np.asarray(global_f, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.max_len = int(max_len)
        self.seq_style = seq_style
        assert len(self.sequences) == len(self.global_f) == len(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        if self.seq_style == "poap":
            seq_t = torch.from_numpy(encode_sequence_poap_onehot(seq, self.max_len))
        else:
            seq_t = torch.from_numpy(encode_sequence_integer(seq, self.max_len))
        return (
            seq_t,
            torch.from_numpy(self.global_f[idx]),
            torch.tensor([self.labels[idx]], dtype=torch.float32),
            seq,
        )
