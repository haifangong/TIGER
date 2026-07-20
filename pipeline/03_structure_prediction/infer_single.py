#!/usr/bin/env python3
"""
Step 3a (single sequence) — HelixFold-Single structure prediction.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import ml_collections
import numpy as np
import paddle

from alphafold_paddle.common import protein, residue_constants
from alphafold_paddle.data.data_utils import single_sequence_to_features
from alphafold_paddle.model import config, features, utils
from utils.model_tape import RunTapeModel
from utils.utils import get_model_parameter_size, tree_map

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "weights" / "helixfold-single.pdparams"


def postprocess(desc: str, batch: dict, results: dict, outdir: Path):
    batch["feat"] = tree_map(lambda x: x[0].numpy(), batch["feat"])
    results = tree_map(lambda x: x[0].numpy(), results)
    results.update(utils.get_confidence_metrics(results))
    plddt = results["plddt"]
    bf = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)
    prot = protein.from_prediction(batch["feat"], results, b_factors=bf)
    out_path = outdir / f"{desc.replace(' ', '_')}.pdb"
    out_path.write_text(protein.to_pdb(prot))
    print(f"[Done] wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Single-sequence HelixFold-Single inference")
    parser.add_argument("--init_model", "--init-model", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--seq", required=True, help="Amino-acid sequence")
    parser.add_argument("--name", default=None, help="Output PDB stem (default: sequence)")
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.chdir(HERE)
    seq = args.seq.strip().upper()
    desc = args.name.strip() if args.name else seq
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f"{desc.replace(' ', '_')}.pdb"
    if outp.is_file() and not args.overwrite:
        print(f"[Skip] {outp} already exists (pass --overwrite to recompute)")
        sys.exit(0)

    init_model = Path(args.init_model)
    if not init_model.is_file():
        raise FileNotFoundError(f"HelixFold weights not found: {init_model}")

    cfg1 = ml_collections.ConfigDict(json.loads((HERE / "model_configs/tape-lnw4.json").read_text()))
    cfg2 = ml_collections.ConfigDict(
        json.loads((HERE / "tape/configs/deberta_1B_bs_cp.json").read_text())
    )
    af2_cfg = config.model_config("seq512_pair64_l24_vio0")
    model = RunTapeModel(None, cfg1, cfg2, af2_cfg)
    print("[Model] size:", get_model_parameter_size(model))
    model.load_params(str(init_model))
    af2_cfg.data.eval.delete_msa_block = False

    rawf = single_sequence_to_features(seq, ">" + desc)
    feat = features.np_example_to_features(np_example=rawf, config=af2_cfg)
    batch = {
        "name": [">" + desc],
        "feat": tree_map(lambda v: paddle.to_tensor(v[None, ...]), feat),
        "label": {},
    }

    model.eval()
    with paddle.no_grad():
        res = model(batch, compute_loss=False)
    postprocess(desc, batch, res, outdir)

    try:
        paddle.device.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    main()
