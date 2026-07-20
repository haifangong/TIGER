#!/usr/bin/env python3
"""
Step 1 — Efficient mutational search + antimicrobial activity pre-filter.

Given a template peptide sequence, enumerate physicochemical-filtered mutants
and classify them with a CatBoost activity model.

Outputs:
  <sequence>_positive_<k>.csv   candidates predicted active
  <sequence>_negative_<k>.csv   candidates predicted inactive
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from itertools import chain, combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path

import joblib
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from joblib import Parallel, delayed

# -----------------------------------------------------------------------------
# Amino-acid / physicochemical constants
# -----------------------------------------------------------------------------
DIWV = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 44.94, -7.49, 1, 1, 1, -7.49, 1, 1, 1, 1, 1, 20.26, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 20.26, 1, 1, 1, 33.6, 1, 1, 20.26, 33.6, 1, 20.26, -6.54, 1, 1, 33.6, -6.54, 24.68, 1],
    [0, 1, 1, 1, 1, -6.54, 1, 1, 1, -7.49, 1, 1, 1, 1, 1, -6.54, 20.26, -14.03, 1, 1, 1],
    [0, 1, 44.94, 20.26, 33.6, 1, 1, -6.54, 20.26, 1, 1, 1, 1, 20.26, 20.26, 1, 20.26, 1, 1, -14.03, 1],
    [0, 1, 1, 13.34, 1, 1, 1, 1, 1, -14.03, 1, 1, 1, 20.26, 1, 1, 1, 1, 1, 1, 33.601],
    [0, -7.49, 1, 1, -6.54, 1, 13.34, 1, -7.49, -7.49, 1, 1, -7.49, 1, 1, 1, 1, -7.49, 1, 13.34, -7.49],
    [0, 1, 1, 1, 1, -9.37, -9.37, 1, 44.94, 24.68, 1, 1, 24.68, -1.88, 1, 1, 1, -6.54, 1, -1.88, 44.94],
    [0, 1, 1, 1, 44.94, 1, 1, 13.34, 1, -7.49, 20.26, 1, 1, -1.88, 1, 1, 1, 1, -7.49, 1, 1],
    [0, 1, 1, 1, 1, 1, -7.49, 1, -7.49, 1, -7.49, 33.6, 1, -6.54, 24.64, 33.6, 1, 1, -7.49, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, -7.49, 1, 1, 1, 20.26, 33.6, 20.26, 1, 1, 1, 24.68, 1],
    [0, 13.34, 1, 1, 1, 1, 1, 58.28, 1, 1, 1, -1.88, 1, 44.94, -6.54, -6.54, 44.94, -1.88, 1, 1, 24.68],
    [0, 1, -1.88, 1, 1, -14.03, -14.03, 1, 44.94, 24.68, 1, 1, 1, -1.88, -6.54, 1, 1, -7.49, 1, -9.37, 1],
    [0, 20.26, -6.54, -6.54, 18.38, 20.26, 1, 1, 1, 1, 1, -6.54, 1, 20.26, 20.26, -6.54, 20.26, 1, 20.26, -1.88, 1],
    [0, 1, -6.54, 20.26, 20.26, -6.54, 1, 1, 1, 1, 1, 1, 1, 20.26, 20.26, 1, 44.94, 1, -6.54, 1, -6.54],
    [0, 1, 1, 1, 1, 1, -7.49, 20.26, 1, 1, 1, 1, 13.34, 20.26, 20.26, 58.28, 44.94, 1, 1, 58.28, -6.54],
    [0, 1, 33.6, 1, 20.26, 1, 1, 1, 1, 1, 1, 1, 1, 44.94, 20.26, 20.26, 20.26, 1, 1, 1, 1],
    [0, 1, 1, 1, 20.26, 13.34, -7.49, 1, 1, 1, 1, 1, -14.03, 1, -6.54, 1, 1, 1, 1, -14.03, 1],
    [0, 1, 1, -14.03, 1, 1, -7.49, 1, 1, -1.88, 1, 1, 1, 20.26, 1, 1, 1, -7.49, 1, 1, -6.54],
    [0, -14.03, 1, 1, 1, 1, -9.37, 24.68, 1, 1, 13.34, 24.68, 13.34, 1, 1, 1, 1, -14.03, -7.49, 1, 1],
    [0, 24.68, 1, 24.68, -6.54, 1, -7.49, 13.34, 1, 1, 1, 44.94, 1, 13.34, 1, -15.91, 1, -7.49, 1, -9.37, 13.34],
])

GRAVY = np.array([
    1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
    1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3,
])

AA_TO_IDX = {
    "": 0, "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8,
    "K": 9, "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16,
    "T": 17, "V": 18, "W": 19, "Y": 20,
}
IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "R_catboost_model.pkl"


def calculate_property(seq: str) -> list[float]:
    """Compute the 10 physicochemical features used by the CatBoost model."""
    pa = ProteinAnalysis(seq)
    aa_counts = pa.count_amino_acids()
    length = len(seq)

    gravy = round(pa.gravy(), 3) * 10
    aliphatic = round(
        (aa_counts["A"] + 2.9 * aa_counts["V"] + 3.9 * (aa_counts["I"] + aa_counts["L"])) / length,
        3,
    ) * 10
    aromaticity = round(pa.aromaticity(), 3) * 10
    instability = round(pa.instability_index(), 3)
    helix, sheet, turn = pa.secondary_structure_fraction()
    charged = sum(aa_counts[a] for a in ["D", "E", "R", "K", "H"])
    charge_density = round(charged / length, 3) * 10

    return [
        gravy,
        aliphatic,
        aromaticity,
        instability,
        round(helix * 10, 3),
        round(sheet * 10, 3),
        round(turn * 10, 3),
        round(pa.charge_at_pH(7), 3),
        round(pa.isoelectric_point(), 3),
        charge_density,
    ]


def calculate_numeric_properties(peptide: np.ndarray) -> tuple[float, float]:
    length = peptide.size
    aliphatic = (
        np.sum(peptide == 1)
        + 2.9 * np.sum(peptide == 18)
        + 3.9 * (np.sum(peptide == 10) + np.sum(peptide == 11))
    ) / length
    aromatic = (np.sum(peptide == 14) + np.sum(peptide == 19) + np.sum(peptide == 20)) / length
    return aliphatic, aromatic


def calculate_charge(peptides: np.ndarray) -> np.ndarray:
    pos = np.isin(peptides, [9, 15, 17]).sum(axis=1)
    neg = np.isin(peptides, [3, 4]).sum(axis=1)
    return pos - neg


def generate_peptides(start_aa: int, peptide_length: int = 3) -> np.ndarray:
    """Enumerate net-positive fill sequences of a fixed mutation length."""
    amino_acids = np.arange(1, 21, dtype=np.int8)
    if peptide_length < 1:
        raise ValueError("peptide_length must be >= 1")
    if peptide_length == 1:
        out = np.array([[start_aa]], dtype=np.int8)
        return out[calculate_charge(out) > 0]

    count = 20 ** (peptide_length - 1)
    out = np.empty((count, peptide_length), dtype=np.int8)
    out[:, 0] = start_aa
    grid = np.array(np.meshgrid(*[amino_acids] * (peptide_length - 1))).T.reshape(-1, peptide_length - 1)
    out[:, 1:] = grid
    return out[calculate_charge(out) > 0]


def generate_mutations(template: np.ndarray, max_changed: int, fill_seqs: np.ndarray):
    """Yield physicochemical-filtered mutants of a template sequence."""
    length = len(template)
    for positions in combinations(range(length), max_changed):
        for fill in fill_seqs:
            if len(fill) != max_changed:
                raise ValueError("Fill sequence length must equal max_changed")
            pep = template.copy()
            pep[list(positions)] = fill

            if np.sum(GRAVY[pep - 1]) / length > 2:
                continue
            aliphatic, aromatic = calculate_numeric_properties(pep)
            if aliphatic > 0.8 or aromatic > 0.4:
                continue
            stability = (10.0 / length) * DIWV[pep[:-1], pep[1:]].sum()
            if stability > 60:
                continue
            yield pep


def collect_for_fill(args):
    template, max_changed, fill_seq = args
    return list(generate_mutations(template, max_changed, [fill_seq]))


def pad_sequence_with_features(seq: np.ndarray, target_length: int = 30) -> np.ndarray:
    padded = np.pad(seq, (0, target_length - len(seq)), mode="constant", constant_values=0)
    aa_seq = "".join(IDX_TO_AA[int(n)] for n in seq)
    feats = calculate_property(aa_seq)
    return np.concatenate([padded, np.array(feats, dtype=float)]).reshape(1, -1)


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _concat_batches(batches):
    if not batches:
        return np.array([])
    try:
        return np.concatenate([np.asarray(b) for b in batches])
    except Exception:
        return np.array(list(chain.from_iterable(batches)))


def estimate_predict_time(clf, padded_batches, sample_count=800, n_jobs=-1, batch_size=10):
    all_samples = _concat_batches(padded_batches)
    total_samples = len(all_samples)
    if total_samples == 0:
        return 0.0, 0.0, 0

    tester_n = min(sample_count, total_samples)
    tester_samples = all_samples[:tester_n]
    tester_batches = [tester_samples[i:i + batch_size] for i in range(0, tester_n, batch_size)]

    try:
        if tester_batches:
            clf.predict(tester_batches[0])
    except Exception:
        pass

    start = time.perf_counter()
    Parallel(n_jobs=n_jobs, backend="loky", max_nbytes="9999M", batch_size=batch_size)(
        delayed(clf.predict)(batch) for batch in tester_batches
    )
    tester_time = time.perf_counter() - start
    estimated_total = tester_time if tester_n < sample_count else (total_samples / float(sample_count)) * tester_time
    return tester_time, estimated_total, total_samples


def run_search(
    sequence: str,
    search_length: int,
    output_dir: str | Path,
    model_path: str | Path = DEFAULT_MODEL,
) -> tuple[Path, Path]:
    """Run mutational search and write positive/negative CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"CatBoost model not found: {model_path}")

    sequence = sequence.strip().upper()
    template = np.array([AA_TO_IDX[a] for a in sequence], dtype=np.int8)
    k = search_length
    start = time.time()

    with Pool(cpu_count()) as pool:
        chunks = pool.starmap(generate_peptides, [(aa, k) for aa in range(1, 21)])
    fills = np.vstack(chunks)
    print(f"[Generation] {time.time() - start:.1f}s  fills={len(fills)}")

    args_list = [(template, k, fills[i]) for i in range(len(fills))]
    with Pool() as pool:
        all_lists = pool.map(collect_for_fill, args_list)
    unique = list({tuple(x) for sub in all_lists for x in sub})
    mutations = [np.array(u, dtype=np.int8) for u in unique]
    print(f"[Mutations] {time.time() - start:.1f}s  unique={len(mutations)}")

    with Pool() as pool:
        padded = pool.starmap(pad_sequence_with_features, [(m, 30) for m in mutations])
    print(f"[Featurize] {time.time() - start:.1f}s")

    clf = joblib.load(model_path)
    tester_time, estimated_total, total_samples = estimate_predict_time(clf, padded)
    print(
        f"[Estimate] tested={min(800, total_samples)}  "
        f"tester={tester_time:.3f}s  estimated_full={format_time(estimated_total)}"
    )

    pred_start = time.perf_counter()
    preds = Parallel(n_jobs=-1, backend="loky", max_nbytes="9999M", batch_size=10)(
        delayed(clf.predict)(batch) for batch in padded
    )
    print(f"[Predict] actual={format_time(time.perf_counter() - pred_start)}")

    pos_path = output_dir / f"{sequence}_positive_{k}.csv"
    neg_path = output_dir / f"{sequence}_negative_{k}.csv"
    n_pos = 0
    with pos_path.open("w", newline="") as fp, neg_path.open("w", newline="") as fn:
        wpos, wneg = csv.writer(fp), csv.writer(fn)
        for i, arr in enumerate(padded):
            pred = preds[i][0]
            nums = arr[0, :30].astype(int)
            seq = "".join(IDX_TO_AA[n] for n in nums if n != 0)
            if pred >= 1:
                wpos.writerow([seq, seq])
                n_pos += 1
            else:
                wneg.writerow([i, seq])

    print(f"[Done] positives={n_pos}  outdir={output_dir}  elapsed={time.time() - start:.1f}s")
    return pos_path, neg_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate peptide mutants and filter by CatBoost antimicrobial activity."
    )
    parser.add_argument("-s", "--sequence", default="KSMLKSMPMTLK", help="Template peptide sequence")
    parser.add_argument("-k", "--search_length", type=int, default=3, help="Number of mutated positions")
    parser.add_argument("-o", "--output_dir", default="outputs", help="Output directory for CSVs")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to CatBoost model pickle (default: models/R_catboost_model.pkl)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_search(args.sequence, args.search_length, args.output_dir, args.model)


if __name__ == "__main__":
    main()
