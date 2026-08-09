#!/usr/bin/env python3
"""Build LL37-holdout train/val + test metadata under TIGER/metadata."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from Bio.Align import PairwiseAligner

ROOT = Path("/data4T/ubuntu/wangyue/postdoc_2025/TIGER_3D")
NEWDATA = ROOT / "newdata"
LL37_SRC = ROOT / "POAP/4_siamese_mic/metadata/LL37_v0.csv"
OUT = ROOT / "TIGER/metadata"
TRAIN_UG = NEWDATA / "dbaasp_amp_training_ug_per_mL.csv"
SIM_THRESH = 0.30
N_WORKERS = max(1, min(32, (os.cpu_count() or 8)))
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def is_standard_sequence(seq: str) -> bool:
    """Keep only canonical L-amino-acid sequences (20 std AAs, uppercase, no X/O/D-aa/etc.)."""
    s = str(seq).strip()
    return bool(s) and s.isalpha() and s.isupper() and set(s) <= STANDARD_AA


def setup_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    return aligner


def pairwise_similarity(aligner: PairwiseAligner, a: str, b: str) -> float:
    """Length-normalized NW score: score(a,b) / max(len(a), len(b)).

    More conservative than the old asymmetric score/len(query), which inflated
    similarity for short cationic peptides against longer LL-37 relatives.
    """
    if not a or not b:
        return 0.0
    return float(aligner.score(a, b) / max(len(a), len(b)))


def max_sim_to_family(seq: str, family: list[str]) -> float:
    """Max similarity of seq to any LL37-family member."""
    if not seq:
        return 0.0
    aligner = setup_aligner()
    return max(pairwise_similarity(aligner, seq, fam) for fam in family)


def _worker(chunk: list[str], family: list[str]) -> list[tuple[str, float]]:
    return [(s, max_sim_to_family(s, family)) for s in chunk]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    ll37 = pd.read_csv(LL37_SRC, header=None, names=["id", "name", "sequence"])
    ll37["sequence"] = ll37["sequence"].astype(str).str.strip()
    ll37["sequence_key"] = ll37["sequence"].str.upper()
    ll37_unique = ll37.drop_duplicates("sequence_key").copy()
    family = ll37_unique["sequence"].tolist()
    family_keys = set(ll37_unique["sequence_key"])

    shutil.copy2(LL37_SRC, OUT / "LL37_v0.csv")

    train = pd.read_csv(TRAIN_UG)
    train["sequence"] = train["sequence"].astype(str).str.strip()
    train["sequence_key"] = train["sequence"].str.upper()
    train["is_standard"] = train["sequence"].map(is_standard_sequence)

    # One representative per upper-cased sequence: prefer a canonical standard spelling.
    train_sorted = train.sort_values("is_standard", ascending=False)
    uniq = (
        train_sorted.drop_duplicates("sequence_key")[["sequence", "sequence_key", "is_standard"]]
        .copy()
        .reset_index(drop=True)
    )
    uniq["exact_ll37"] = uniq["sequence_key"].isin(family_keys)

    # Train/val never includes non-standard residues; skip them in the similarity scan.
    nonstd_uniq = uniq.loc[~uniq["is_standard"] & ~uniq["exact_ll37"]].copy()
    candidates = uniq.loc[uniq["is_standard"] & ~uniq["exact_ll37"], "sequence"].tolist()
    print(f"LL37 family unique: {len(family_keys)}")
    print(f"Training unique seqs: {uniq['sequence_key'].nunique()}")
    print(f"Exact LL37 overlap to remove: {int(uniq['exact_ll37'].sum())}")
    print(f"Non-standard AA unique (excluded from train/val): {len(nonstd_uniq)}")
    print(f"Candidates for similarity scan: {len(candidates)} (workers={N_WORKERS})")

    chunk_size = max(50, len(candidates) // (N_WORKERS * 4) or 50)
    chunks = [candidates[i : i + chunk_size] for i in range(0, len(candidates), chunk_size)]

    sim_map: dict[str, float] = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_worker, ch, family) for ch in chunks]
        done = 0
        for fut in as_completed(futs):
            for seq, sim in fut.result():
                sim_map[seq] = sim
            done += 1
            if done % 5 == 0 or done == len(futs):
                print(f"  progress chunks {done}/{len(futs)}", flush=True)

    def _sim_for_row(r: pd.Series) -> float:
        if r["exact_ll37"]:
            return 1.0
        if not r["is_standard"]:
            return float("nan")
        return float(sim_map.get(r["sequence"], 0.0))

    uniq["max_sim_to_ll37"] = uniq.apply(_sim_for_row, axis=1)
    # Round to avoid float noise around the 30% boundary (e.g. 0.3000000000000002).
    # User rule: remove only if similarity *strictly greater than* 30%.
    uniq["max_sim_to_ll37"] = uniq["max_sim_to_ll37"].round(10)
    uniq["remove_sim"] = uniq["exact_ll37"] | (
        uniq["is_standard"] & (uniq["max_sim_to_ll37"] > SIM_THRESH)
    )
    uniq["remove_nonstandard"] = ~uniq["is_standard"] & ~uniq["exact_ll37"]
    uniq["keep_train_val"] = uniq["is_standard"] & ~uniq["remove_sim"]

    removed = uniq[uniq["remove_sim"]].copy()
    kept = uniq[uniq["keep_train_val"]].copy()
    print(f"Removed (sim>{SIM_THRESH} or exact): {len(removed)}")
    print(f"Kept for train/val (standard + sim<=30%): {len(kept)}")
    print(f"Removed sim distribution:\n{removed['max_sim_to_ll37'].describe()}")

    keep_keys = set(kept["sequence_key"])
    remove_sim_keys = set(removed["sequence_key"])
    nonstd_keys = set(nonstd_uniq["sequence_key"])

    # Keep only rows whose *literal* sequence is canonical (same upper-key may have
    # D-aa / mixed-case variants that must not enter train/val).
    train_val = train[train["sequence_key"].isin(keep_keys) & train["is_standard"]].drop(
        columns=["sequence_key", "is_standard"]
    ).copy()
    mixed_case_dropped = train[
        train["sequence_key"].isin(keep_keys) & ~train["is_standard"]
    ].drop(columns=["sequence_key", "is_standard"]).copy()

    removed_rows = train[train["sequence_key"].isin(remove_sim_keys)].drop(
        columns=["sequence_key", "is_standard"]
    ).copy()
    nonstd_rows = train[train["sequence_key"].isin(nonstd_keys)].drop(
        columns=["sequence_key", "is_standard"]
    ).copy()
    if len(mixed_case_dropped):
        print(f"Also dropped mixed-case/nonstd row variants under kept keys: {len(mixed_case_dropped)}")
        nonstd_rows = pd.concat([nonstd_rows, mixed_case_dropped], ignore_index=True).drop_duplicates()
    test_rows = train[train["sequence_key"].isin(family_keys)].drop(
        columns=["sequence_key", "is_standard"]
    ).copy()

    assert train_val["sequence"].map(is_standard_sequence).all()

    print(
        f"Dropped non-standard AA rows from train/val: {len(nonstd_rows)} "
        f"(unique {nonstd_rows['sequence'].str.upper().nunique()})"
    )
    print(
        f"Train/val rows: {len(train_val)} unique seqs: "
        f"{train_val['sequence'].str.upper().nunique()}"
    )
    print(
        f"Test ug/ml rows: {len(test_rows)} unique: "
        f"{test_rows['sequence'].str.upper().nunique()}"
    )
    missing_labels = family_keys - set(test_rows["sequence"].str.upper())
    print(f"LL37 family seqs without ug/ml labels: {len(missing_labels)}")

    train_val_seq = (
        train_val[["sequence"]]
        .assign(description=lambda d: d["sequence"])
        .drop_duplicates(subset=["sequence"], keep="first")
        .reset_index(drop=True)
    )
    test_seq_simple = ll37_unique.assign(description=lambda d: d["sequence"])[
        ["sequence", "description"]
    ]

    train_val.to_csv(OUT / "train_val_ug_per_mL.csv", index=False)
    test_rows.to_csv(OUT / "test_LL37_ug_per_mL.csv", index=False)
    train_val_seq.to_csv(OUT / "train_val_seq.csv", index=False)
    test_seq_simple.to_csv(OUT / "test_seq.csv", index=False)
    ll37_unique[["id", "name", "sequence"]].to_csv(OUT / "test_seq_LL37_named.csv", index=False)

    audit = uniq[
        [
            "sequence",
            "sequence_key",
            "is_standard",
            "exact_ll37",
            "max_sim_to_ll37",
            "remove_sim",
            "remove_nonstandard",
            "keep_train_val",
        ]
    ].sort_values("max_sim_to_ll37", ascending=False, na_position="last")
    audit.to_csv(OUT / "similarity_filter_audit.csv", index=False)
    removed[["sequence", "sequence_key", "exact_ll37", "max_sim_to_ll37"]].to_csv(
        OUT / "removed_similar_to_LL37_seq.csv", index=False
    )
    removed_rows.to_csv(OUT / "removed_similar_to_LL37_ug_per_mL.csv", index=False)
    nonstd_rows.to_csv(OUT / "removed_nonstandard_AA_ug_per_mL.csv", index=False)
    (
        nonstd_rows[["sequence"]]
        .drop_duplicates()
        .assign(description=lambda d: d["sequence"])
        .to_csv(OUT / "removed_nonstandard_AA_seq.csv", index=False)
    )

    summary = f"""# LL37-holdout metadata (ug/mL)

## Setting
- **Test family**: sequences from `POAP/4_siamese_mic/metadata/LL37_v0.csv` ({len(family_keys)} unique)
- **Train/val source**: `newdata/dbaasp_amp_training_ug_per_mL.csv`
- **Filter**: remove any train/val sequence with length-normalized NW similarity **> {SIM_THRESH:.0%}** to any LL37 family sequence
- **Standard AA filter (train/val only)**: keep sequences composed solely of the 20 canonical uppercase amino acids `ACDEFGHIKLMNPQRSTVWY` (exclude X/O, D-amino acids/lowercase, digits, etc.)
- **Similarity definition**: Biopython global `PairwiseAligner` (match=1, mismatch=-1, open=-0.5, extend=-0.1); similarity = `score(a,b) / max(len(a), len(b))`. (Replaces the older asymmetric `score/len(query)`, which over-filtered short peptides.)

## Counts
| Split | Rows | Unique sequences |
|---|---:|---:|
| train/val ug/mL | {len(train_val)} | {train_val['sequence'].str.upper().nunique()} |
| test ug/mL (LL37 rows with labels) | {len(test_rows)} | {test_rows['sequence'].str.upper().nunique()} |
| removed by LL37 similarity | {len(removed_rows)} | {len(remove_sim_keys)} |
| removed by non-standard AA | {len(nonstd_rows)} | {nonstd_rows['sequence'].str.upper().nunique() if len(nonstd_rows) else 0} |
| LL37 family (sequence list) | - | {len(family_keys)} |

## Files
- `LL37_v0.csv` — copied test family table
- `test_seq.csv` — LL37 unique sequences (`sequence,description`)
- `test_seq_LL37_named.csv` — LL37 with id/name
- `test_LL37_ug_per_mL.csv` — ug/mL assay rows for LL37 sequences (from training table)
- `train_val_seq.csv` — filtered train/val sequences
- `train_val_ug_per_mL.csv` — filtered train/val ug/mL assays
- `similarity_filter_audit.csv` — per-sequence max similarity to LL37
- `removed_similar_to_LL37_seq.csv` — sequences removed by the similarity filter
- `removed_similar_to_LL37_ug_per_mL.csv` — assay rows removed by the similarity filter
- `removed_nonstandard_AA_seq.csv` — sequences removed for non-standard residues
- `removed_nonstandard_AA_ug_per_mL.csv` — assay rows removed for non-standard residues

Train/val is kept as one pool for GroupKFold CV (as in the TIGER pipeline).
"""
    (OUT / "README.md").write_text(summary)
    print("Wrote files to", OUT)
    print(summary)


if __name__ == "__main__":
    main()
