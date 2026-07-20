#!/usr/bin/env python3
"""Minimal inference dataset for hemolytic / toxicity filtering (mode=101)."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import pandas as pd
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from torch.utils.data import Dataset
from tqdm import tqdm

AMAs = {
    "G": 20, "A": 1, "V": 2, "L": 3, "I": 4, "P": 5, "F": 6, "Y": 7, "W": 8,
    "S": 9, "T": 10, "C": 11, "M": 12, "N": 13, "Q": 14, "D": 15, "E": 16,
    "K": 17, "R": 18, "H": 19, "X": 21,
}


def calculate_property(seq: str) -> list[float]:
    analysed_seq = ProteinAnalysis(seq)
    aa_counts = analysed_seq.count_amino_acids()
    aliphatic_index = (
        (aa_counts["A"] + 2.9 * aa_counts["V"] + 3.9 * (aa_counts["I"] + aa_counts["L"])) / len(seq)
    )
    total_charge = (
        sum(aa_counts.get(aa, 0) for aa in ["R", "K", "H"])
        - sum(aa_counts.get(aa, 0) for aa in ["D", "E"])
    )
    charge_density = total_charge / len(seq)
    alpha_helix, beta_helix, turn_helix = analysed_seq.secondary_structure_fraction()
    return [
        round(analysed_seq.gravy(), 3) * 10,
        round(aliphatic_index, 3) * 10,
        round(analysed_seq.aromaticity(), 3) * 10,
        round(analysed_seq.instability_index(), 3),
        round(alpha_helix * 10, 3),
        round(beta_helix * 10, 3),
        round(turn_helix * 10, 3),
        round(analysed_seq.charge_at_pH(7), 3),
        round(analysed_seq.isoelectric_point(), 3),
        round(charge_density, 3) * 10,
    ]


class PeptideInferenceDataset(Dataset):
    """Load sequences from a headerless CSV: sequence[,score_or_copy]."""

    def __init__(
        self,
        csv: str,
        max_length: int = 30,
        model_mode: str = "101",
        min_length: int = 6,
    ):
        if model_mode[1] != "0":
            raise ValueError(
                "This slim dataset only supports sequence+globf inference "
                "(model_mode[1] must be '0', e.g. '101')."
            )

        all_data = pd.read_csv(csv, encoding="unicode_escape", header=None).values
        idx_list, seq_list, labels = self.data_process(all_data, mode="p123")

        filter_idx_list, seq_new_list, label_list = [], [], []
        invalid = set("XBJZUO")
        for idx in range(len(idx_list)):
            seq = str(seq_list[idx]).upper().strip()
            if any(ch in seq for ch in invalid) or not (min_length <= len(seq) <= max_length):
                continue
            filter_idx_list.append(idx)
            seq_new_list.append(seq)
            label_list.append(labels[idx])

        self.data_list = []
        for i in tqdm(range(len(filter_idx_list)), desc="Featurizing"):
            seq = seq_new_list[i]
            label = label_list[i]
            seq_emb = np.zeros((max_length, 21), dtype=np.float32)
            for pos, aa in enumerate(seq):
                seq_emb[pos, AMAs[aa]] = 1
            globf = calculate_property(seq)
            self.data_list.append((0, seq_emb, globf, label, seq))

    @staticmethod
    def data_process(
        data: np.ndarray,
        mode: Literal["p123"] = "p123",
        threshold_bin: Union[float, tuple] = 0.5,
    ):
        del threshold_bin
        if mode != "p123":
            raise NotImplementedError(mode)
        idx_list = range(len(data))
        seq_list = data[:, 0]
        labels = [1] * len(data)
        return idx_list, seq_list, labels

    def __getitem__(self, idx):
        voxel, seq_emb, globf, gt, seq = self.data_list[idx]
        return (
            torch.tensor(voxel).float(),
            torch.tensor(seq_emb).float(),
            torch.tensor(globf).float(),
            torch.tensor([gt]).float(),
            seq,
        )

    def __len__(self):
        return len(self.data_list)


# Backward-compatible alias used by the original POAP scripts.
MDataset = PeptideInferenceDataset
