"""Shared constants for TIGER core.

Amino-acid integer encoding is sequential 1..20 in alphabetical order
(A=1 … Y=20). Index 0 is reserved for padding.
"""

from __future__ import annotations

import re

# Alphabetical standard amino acids → codes 1..20
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: dict[str, int] = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}
# Explicit table for readability / docs
# A:1 C:2 D:3 E:4 F:5 G:6 H:7 I:8 K:9 L:10
# M:11 N:12 P:13 Q:14 R:15 S:16 T:17 V:18 W:19 Y:20
NUM_AA = 20
PAD_IDX = 0
NUM_AA_TOKENS = NUM_AA + 1  # pad + 20 AAs

VALID_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
TARGET_COL = "MIC_Escherichia_coli"
CFU_COL = "cfu_group"
CFU_LEVELS = {"unknown": 0, "1E4 - 1E5": 1, "1E5 - 1E6": 2, "1E6 - 1E7": 3}
PRIMARY_CFU = "1E5 - 1E6"


def encode_sequence(seq: str, max_len: int) -> list[int]:
    """Pad amino-acid sequence to ``max_len`` with sequential codes 1..20 (0=pad)."""
    ids = [AA_TO_IDX[ch] for ch in seq.upper()]
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids + [PAD_IDX] * (max_len - len(ids))
