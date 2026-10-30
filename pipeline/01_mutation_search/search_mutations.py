#!/usr/bin/env python3
"""
Step 1 — Efficient mutational search + antimicrobial activity pre-filter.

Given a template peptide sequence, enumerate physicochemical-filtered mutants
and classify them with a calibrated E. coli MIC <128 µg/mL model.

Outputs:
  <sequence>_positive_<k>.csv   candidates predicted active
  <sequence>_negative_<k>.csv   candidates predicted inactive
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import sys
import numpy as np

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
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from amp_classification.predict import Predictor
from amp_classification.features import validate
DEFAULT_MODEL = REPO / "amp_classification/checkpoints/CatBoost-Tuned/seed123"


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


def run_search(
    sequence: str,
    search_length: int,
    output_dir: str | Path,
    model_path: str | Path = DEFAULT_MODEL,
    operating_point: str = 'recall95',
    threads: int = 4,
) -> tuple[Path, Path]:
    """Keep the historical mutant generator; screen with the current MIC128 model."""
    import json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)
    if not (model_path / 'release.json').is_file():
        raise FileNotFoundError(f'Expected a released MIC128 checkpoint directory: {model_path}')
    identity = json.loads((model_path / 'release.json').read_text())
    classifier = Predictor(identity['model'], identity['seed'], model_path, threads=threads)
    sequence = sequence.strip().upper()
    validate([sequence])
    if not 1 <= search_length <= len(sequence):
        raise ValueError('search_length must lie between 1 and the template length')
    if threads < 1:
        raise ValueError('threads must be positive')
    template = np.array([AA_TO_IDX[a] for a in sequence], dtype=np.int8)
    k = search_length
    start = time.time()
    with Pool(threads) as pool:
        chunks = pool.starmap(generate_peptides, [(aa, k) for aa in range(1, 21)])
    fills = np.vstack(chunks)
    print(f'[Generation] {time.time()-start:.1f}s  fills={len(fills)}')
    args_list = [(template, k, fills[i]) for i in range(len(fills))]
    with Pool(threads) as pool:
        all_lists = pool.map(collect_for_fill, args_list)
    unique = list({tuple(x) for sub in all_lists for x in sub})
    sequences = [''.join(IDX_TO_AA[n] for n in u) for u in unique]
    pos_path = output_dir / f'{sequence}_positive_{k}.csv'
    neg_path = output_dir / f'{sequence}_negative_{k}.csv'
    scores_path = output_dir / f'{sequence}_MIC128_scores_{k}.csv'
    cutoff = classifier.cut[operating_point]
    n_pos = 0
    with pos_path.open('w', newline='') as fp, neg_path.open('w', newline='') as fn, scores_path.open('w', newline='') as fs:
        wpos, wneg, ws = csv.writer(fp), csv.writer(fn), csv.writer(fs)
        ws.writerow(['sequence', 'probability_AMP_balanced_cohort', 'predicted_AMP', 'operating_point', 'threshold', 'model', 'seed'])
        for offset in range(0, len(sequences), 4096):
            batch = sequences[offset:offset+4096]
            probabilities = classifier.predict_proba(batch)
            for i, (seq, prob) in enumerate(zip(batch, probabilities), offset):
                active = int(prob >= cutoff)
                ws.writerow([seq, prob, active, operating_point, cutoff, identity['model'], identity['seed']])
                if active:
                    wpos.writerow([seq, seq]); n_pos += 1
                else:
                    wneg.writerow([i, seq])
    print(f'[Done] {identity["model"]} seed{identity["seed"]}; {operating_point}; positives={n_pos}/{len(sequences)}; elapsed={time.time()-start:.1f}s')
    return pos_path, neg_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mutants and screen E. coli MIC <128 µg/mL with a calibrated classifier."
    )
    parser.add_argument("-s", "--sequence", default="KSMLKSMPMTLK", help="Template peptide sequence")
    parser.add_argument("-k", "--search_length", type=int, default=3, help="Number of mutated positions")
    parser.add_argument("-o", "--output_dir", default="outputs", help="Output directory for CSVs")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Released MIC128 checkpoint directory (default: CatBoost-Tuned seed123)",
    )
    parser.add_argument("--operating-point", choices=["balanced", "default", "recall90", "recall95", "recall98"], default="recall95")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    run_search(args.sequence, args.search_length, args.output_dir, args.model, args.operating_point, args.threads)


if __name__ == "__main__":
    main()
