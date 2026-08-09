#!/usr/bin/env python3
"""Sequence-encoding ablation under the locked paper MIC pair-delta recipe.

Locked
------
  structure_features=s, include_node_coords=false
  feature_modalities=gsh
  fusion=attention, fusion_attn_mode=cross_qs
  similarity_threshold=0.3
  use_signed_sampling=false, delta_bin_width=1.0
  pair_balance_num=10000
  lr=1e-3, wd=0, cosine, zscore

Sweep ``seq_encoding``
----------------------
  integer   — Linear(max_len) over AA codes 1..20 (paper default / baseline)
  embedding — nn.Embedding(21, emb_dim) + positional embedding + masked mean
  onehot    — one-hot(21) → Linear(21, emb_dim) + positional embedding + masked mean

Selection (lower better): log2MAE + RSE - PCC - KCC
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "code/configs/gsh_struct_s_base.json"
DEFAULT_SEED = 1

# Categorical AA methods vs legacy integer codes.
SEQ_ENCODING_GRID = ["integer", "embedding", "onehot"]


def build_specs():
    out = []
    for enc in SEQ_ENCODING_GRID:
        name = f"seq_{enc}"
        out.append(
            {
                "id": f"seq_encoding_grid__{name}",
                "name": name,
                "seq_encoding": enc,
            }
        )
    return out


def build_command(item, out_dir, gpu, seed=DEFAULT_SEED):
    return [
        sys.executable,
        "-m",
        "code.main",
        "train",
        "--config",
        str(BASE_CONFIG),
        "--out-dir",
        str(out_dir),
        "--gpu",
        str(gpu),
        "--name",
        item["name"],
        "--structure-features",
        "s",
        "--feature-modalities",
        "gsh",
        "--fusion",
        "attention",
        "--fusion-attn-mode",
        "cross_qs",
        "--seq-encoding",
        item["seq_encoding"],
        "--no-include-node-coords",
        "--similarity-threshold",
        "0.3",
        "--delta-bin-width",
        "1.0",
        "--pair-balance-num",
        "10000",
        "--no-use-signed-sampling",
        "--seed",
        str(seed),
    ]


def _row_sort_key(r: dict):
    score = r.get("cv_selection_score")
    return (
        score is None or not isinstance(score, (int, float)),
        score if isinstance(score, (int, float)) else 1e9,
        r.get("cv_log2MAE") if r.get("cv_log2MAE") is not None else 1e9,
        -(r.get("cv_PCC") or -1.0),
    )


def write_leaderboard(out_root: Path, specs: list[dict]) -> list[dict]:
    rows = []
    for item in specs:
        summary_path = out_root / item["id"] / "results" / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        cv = summary.get("cv", {})
        rows.append(
            {
                "id": item["id"],
                "name": item["name"],
                "seq_encoding": item["seq_encoding"],
                "cv_n": cv.get("n"),
                "cv_log10MAE": cv.get("log10MAE"),
                "cv_log2MAE": cv.get("log2MAE", cv.get("mae")),
                "cv_RSE": cv.get("RSE", cv.get("rse")),
                "cv_PCC": cv.get("PCC", cv.get("pearson")),
                "cv_KCC": cv.get("KCC", cv.get("kendall")),
                "cv_selection_score": cv.get("selection_score"),
                "runtime_sec": summary.get("runtime_sec") or summary.get("runtime_seconds"),
            }
        )
    rows.sort(key=_row_sort_key)
    (out_root / "cv_leaderboard.json").write_text(json.dumps(rows, indent=2))
    if rows:
        (out_root / "best_config.json").write_text(json.dumps(rows[0], indent=2))
        best = rows[0]
        print(
            f"[best] {best['name']} enc={best['seq_encoding']} "
            f"score={best['cv_selection_score']:.4f} "
            f"log2MAE={best['cv_log2MAE']:.4f} PCC={best['cv_PCC']:.4f}",
            flush=True,
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs" / "outputs_code_struct_s_seq_encoding_grid",
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--leaderboard-only", action="store_true")
    args = parser.parse_args()
    if args.slots_per_gpu < 1:
        parser.error("--slots-per-gpu must be at least 1")

    specs = build_specs()
    if args.only:
        wanted = set(args.only)
        specs = [x for x in specs if x["id"] in wanted or x["name"] in wanted]

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "locked": {
                    "structure_features": "s",
                    "feature_modalities": "gsh",
                    "fusion": "attention",
                    "fusion_attn_mode": "cross_qs",
                    "similarity_threshold": 0.3,
                    "delta_bin_width": 1.0,
                    "pair_balance_num": 10000,
                    "use_signed_sampling": False,
                    "include_node_coords": False,
                },
                "seq_encoding_grid": SEQ_ENCODING_GRID,
                "seq_encoding_legend": {
                    "integer": "Linear(max_len) over AA codes 1..20 (legacy / paper default)",
                    "embedding": "nn.Embedding(21, emb_dim) + pos embedding + masked mean",
                    "onehot": "one-hot(21) → Linear(21, emb_dim) + pos embedding + masked mean",
                },
                "experiments": build_specs(),
            },
            indent=2,
        )
    )
    logs = args.out_root / "suite_logs"
    logs.mkdir(exist_ok=True)

    if args.leaderboard_only:
        write_leaderboard(args.out_root, build_specs())
        return

    pending = [x for x in specs if not (args.out_root / x["id"] / "results/summary.json").exists()]
    active, failures = [], []
    print(
        f"[seq-encoding-grid] total={len(specs)} pending={len(pending)} "
        f"gpus={args.gpus} slots={args.slots_per_gpu}",
        flush=True,
    )
    print(f"[seq-encoding-grid] encodings={SEQ_ENCODING_GRID}", flush=True)

    while pending or active:
        occupancy = {gpu: sum(j["gpu"] == gpu for j in active) for gpu in args.gpus}
        for gpu in args.gpus:
            while pending and occupancy[gpu] < args.slots_per_gpu:
                item = pending.pop(0)
                out_dir = args.out_root / item["id"]
                out_dir.mkdir(parents=True, exist_ok=True)
                handle = (logs / f"{item['id']}.log").open("w")
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                proc = subprocess.Popen(
                    build_command(item, out_dir, gpu, seed=args.seed),
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                active.append({"item": item, "gpu": gpu, "proc": proc, "handle": handle, "started": time.time()})
                occupancy[gpu] += 1
                print(f"[launch] {item['id']} gpu={gpu} enc={item['seq_encoding']}", flush=True)
        time.sleep(5)
        keep = []
        for job in active:
            code = job["proc"].poll()
            if code is None:
                keep.append(job)
                continue
            job["handle"].close()
            print(
                f"[done] {job['item']['id']} gpu={job['gpu']} code={code} "
                f"elapsed={time.time() - job['started']:.1f}s",
                flush=True,
            )
            if code:
                failures.append({"id": job["item"]["id"], "code": code})
            write_leaderboard(args.out_root, build_specs())
        active = keep
        (args.out_root / "suite_state.json").write_text(
            json.dumps(
                {
                    "pending": [x["id"] for x in pending],
                    "active": [j["item"]["id"] for j in active],
                    "failures": failures,
                },
                indent=2,
            )
        )

    rows = write_leaderboard(args.out_root, build_specs())
    (args.out_root / "suite_done.json").write_text(
        json.dumps({"experiments": len(specs), "completed": len(rows), "failures": failures}, indent=2)
    )
    if failures:
        raise SystemExit(f"Failures: {failures}")


if __name__ == "__main__":
    main()
