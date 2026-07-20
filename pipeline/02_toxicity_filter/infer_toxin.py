#!/usr/bin/env python3
"""
Step 2 — Hemolytic / host-toxicity filtering.

Reads candidate sequences from Step 1 and splits them into toxin / non-toxin
subsets using a pretrained FusionPeptide classifier (mode=101).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PeptideInferenceDataset
from network import FusionPeptide

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "checkpoints" / "model_1.pth"
DEFAULT_CONFIG = HERE / "checkpoints" / "config.json"


def load_model(args, device: torch.device) -> FusionPeptide:
    model = FusionPeptide(
        classes=1,
        q_encoder=args.q_encoder,
        v_encoder=args.v_encoder,
        channels=args.channels,
        mode=args.mode,
    ).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(args) -> tuple[Path, Path, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = load_model(args, device)
    dataset = PeptideInferenceDataset(
        csv=args.csv,
        max_length=args.max_len,
        model_mode=args.mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    toxins, non_toxins = [], []
    t0 = time.time()
    with torch.no_grad():
        for voxel, seq, globf, _gt, seq_str in tqdm(loader, desc="Toxicity inference"):
            probs = torch.sigmoid(model((voxel.to(device), seq.to(device), globf.to(device))))
            for sequence, ptox, props in zip(seq_str, probs, globf):
                row = {
                    "sequence": sequence,
                    "tox_prob": float(ptox),
                    "hydrophobic": props[0].item(),
                    "aliphatic_index": props[1].item(),
                    "aromaticity": props[2].item(),
                    "instability_index": props[3].item(),
                    "alpha_helix": props[4].item(),
                    "beta_helix": props[5].item(),
                    "turn_helix": props[6].item(),
                    "charge": props[7].item(),
                    "isoelectric_pt": props[8].item(),
                    "charge_density": props[9].item(),
                }
                if float(ptox) >= args.threshold:
                    toxins.append(row)
                else:
                    non_toxins.append(row)

    elapsed = time.time() - t0
    stem = Path(args.csv).stem
    # Prefer a readable stem: drop trailing "_positive_k" if present.
    save_name = stem.split("_positive")[0] if "_positive" in stem else stem.split("_")[0]

    toxin_path = out_dir / f"{save_name}_toxins.csv"
    non_toxin_path = out_dir / f"{save_name}_non_toxins.csv"
    all_path = out_dir / f"{save_name}_all.csv"

    pd.DataFrame(toxins).to_csv(toxin_path, index=False)
    pd.DataFrame(non_toxins).to_csv(non_toxin_path, index=False)
    pd.DataFrame(toxins + non_toxins).to_csv(all_path, index=False)

    total = len(dataset)
    rate = total / elapsed if elapsed > 0 else float("inf")
    print(f"Processed {total} sequences in {elapsed:.2f}s ({rate:.2f} seq/s)")
    print(f"toxins={len(toxins)}  non_toxins={len(non_toxins)}")
    print(f"Wrote:\n  {toxin_path}\n  {non_toxin_path}\n  {all_path}")
    return toxin_path, non_toxin_path, all_path


def parse_args():
    defaults = {}
    if DEFAULT_CONFIG.is_file():
        defaults = json.loads(DEFAULT_CONFIG.read_text())

    parser = argparse.ArgumentParser(description="Filter hemolytic / toxic peptide candidates.")
    parser.add_argument("--csv", required=True, help="Input CSV from Step 1 (sequence[,...])")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL), help="Checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Toxicity probability cutoff")
    parser.add_argument("--max-len", type=int, default=defaults.get("max_length", 30))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--q-encoder", default=defaults.get("q_encoder", "gru"))
    parser.add_argument("--v-encoder", default=defaults.get("model", "resnet34"))
    parser.add_argument("--channels", type=int, default=defaults.get("channels", 32))
    parser.add_argument("--mode", default=defaults.get("mode", "101"))
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
