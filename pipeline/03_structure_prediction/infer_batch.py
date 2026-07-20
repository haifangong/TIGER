#!/usr/bin/env python3
"""
Step 3a — Batch HelixFold-Single structure prediction.

Reads a CSV with a `sequence` column (or headerless sequence[,...]) and writes
one unrelaxed PDB per sequence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import ml_collections
import numpy as np
import paddle
import pandas as pd

from alphafold_paddle.common import protein, residue_constants
from alphafold_paddle.data.data_utils import single_sequence_to_features
from alphafold_paddle.model import config, features, utils
from utils.model_tape import RunTapeModel
from utils.utils import get_model_parameter_size, tree_map

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "weights" / "helixfold-single.pdparams"


def read_sequences(csv_file: str | Path) -> tuple[list[str], list[str]]:
    csv_file = Path(csv_file)
    # Prefer pandas when a header exists.
    with csv_file.open("r", encoding="utf-8-sig") as handle:
        first = handle.readline().strip().lower()
    has_header = "sequence" in first or first.startswith("seq")

    if has_header:
        df = pd.read_csv(csv_file)
        col = "sequence" if "sequence" in df.columns else df.columns[0]
        sequences = [str(s).strip().upper() for s in df[col].tolist() if str(s).strip()]
    else:
        lines = csv_file.read_text(encoding="utf-8-sig").splitlines()
        sequences = []
        for line in lines:
            if not line.strip():
                continue
            seq = line.strip().split(",")[0].strip().upper()
            if seq and seq.lower() != "sequence":
                sequences.append(seq)

    descriptions = list(sequences)
    return sequences, descriptions


def sequence_to_batch(sequence: str, description: str, model_config):
    raw_features = single_sequence_to_features(sequence, ">" + description)
    feat = features.np_example_to_features(np_example=raw_features, config=model_config)
    return {
        "name": [">" + description],
        "feat": tree_map(lambda v: paddle.to_tensor(v[None, ...]), feat),
        "label": {},
    }


def postprocess(description: str, batch: dict, results: dict, output_dir: Path):
    batch["feat"] = tree_map(lambda x: x[0].numpy(), batch["feat"])
    results = tree_map(lambda x: x[0].numpy(), results)
    results.update(utils.get_confidence_metrics(results))
    plddt = results["plddt"]
    plddt_b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)
    prot = protein.from_prediction(batch["feat"], results, b_factors=plddt_b_factors)
    pdb_path = output_dir / f"{description.replace(' ', '_')}.pdb"
    pdb_path.write_text(protein.to_pdb(prot))
    return pdb_path


def build_model(init_model: Path):
    model_config = ml_collections.ConfigDict(
        json.loads((HERE / "model_configs" / "tape-lnw4.json").read_text())
    )
    tape_model_config = ml_collections.ConfigDict(
        json.loads((HERE / "tape" / "configs" / "deberta_1B_bs_cp.json").read_text())
    )
    af2_model_config = config.model_config("seq512_pair64_l24_vio0")
    model = RunTapeModel(None, model_config, tape_model_config, af2_model_config)
    print("model size:", get_model_parameter_size(model))
    model.load_params(str(init_model))
    af2_model_config.data.eval.delete_msa_block = False
    return model, af2_model_config


def run_batch(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_model = Path(args.init_model)
    if not init_model.is_file():
        raise FileNotFoundError(
            f"HelixFold weights not found: {init_model}\n"
            "Download helixfold-single.pdparams or update --init_model / weights symlink."
        )

    model, af2_model_config = build_model(init_model)
    sequences, descriptions = read_sequences(args.csv_file)
    print(f"Predicting {len(sequences)} sequences -> {output_dir}")

    for sequence, description in zip(sequences, descriptions):
        out_pdb = output_dir / f"{description.replace(' ', '_')}.pdb"
        if args.skip_exist and out_pdb.is_file():
            print(f"[Skip] {out_pdb.name}")
            continue
        print(f"[Predict] {sequence}")
        batch = sequence_to_batch(sequence, description, af2_model_config)
        model.eval()
        with paddle.no_grad():
            results = model(batch, compute_loss=False)
        postprocess(description, batch, results, output_dir)

    print(f"Done. Outputs written to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch HelixFold-Single inference")
    parser.add_argument("--csv_file", "--csv-file", required=True, help="Input CSV of sequences")
    parser.add_argument("--output_dir", "--output-dir", default="outputs/pdb", help="PDB output dir")
    parser.add_argument(
        "--init_model",
        "--init-model",
        default=str(DEFAULT_WEIGHTS),
        help="Path to helixfold-single.pdparams",
    )
    parser.add_argument(
        "--skip_exist",
        "--skip-exist",
        action="store_true",
        default=True,
        help="Skip sequences whose PDB already exists (default: True)",
    )
    parser.add_argument(
        "--no-skip-exist",
        dest="skip_exist",
        action="store_false",
        help="Recompute even if PDB exists",
    )
    return parser.parse_args()


def main():
    # Ensure local package imports work when launched from another cwd.
    os.chdir(HERE)
    args = parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
