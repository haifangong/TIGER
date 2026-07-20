#!/usr/bin/env python3
"""
Step 3b — Rosetta / PyRosetta FastRelax for predicted PDBs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyrosetta


def relax_pdb(pdb_file: str, input_dir: str, output_dir: str) -> str:
    pyrosetta.init(extra_options="-mute all")
    scorefxn = pyrosetta.get_fa_scorefxn()
    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)

    pose = pyrosetta.pose_from_pdb(str(Path(input_dir) / pdb_file))
    relax.apply(pose)

    output_path = Path(output_dir) / pdb_file
    pose.dump_pdb(str(output_path))
    print(f"Relaxed structure saved to {output_path}")
    return str(output_path)


def run_relax(input_dir: str | Path, output_dir: str | Path, workers: int | None = None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(f.name for f in input_dir.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {input_dir}")

    print(f"Relaxing {len(pdb_files)} PDBs: {input_dir} -> {output_dir}")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(
            executor.map(
                relax_pdb,
                pdb_files,
                [str(input_dir)] * len(pdb_files),
                [str(output_dir)] * len(pdb_files),
            )
        )
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="PyRosetta FastRelax for HelixFold PDBs")
    parser.add_argument("--input_dir", "--input-dir", required=True)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=None, help="ProcessPool workers")
    return parser.parse_args()


def main():
    args = parse_args()
    run_relax(args.input_dir, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
