"""Run an archived CV/final recipe with its original numerical worker."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True, help='A new/empty directory; frozen release data cannot be overwritten')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='TinyCNN only; original training used CUDA')
    p.add_argument('--threads', type=int, default=4, help='v3 worker CPU threads; original v2 workers use four, TinyCNN two')
    a = p.parse_args(argv)
    if a.threads < 1:
        raise ValueError('threads must be positive')
    record = json.loads(a.config.read_text()); job = record['job']; version = record['backend']
    methods = json.loads((ROOT / 'methods.json').read_text())['methods']
    if job['model'] not in methods or version not in ['v2', 'v3'] or job['stage'] not in ['cv', 'final']:
        raise ValueError('Not a released classification recipe')
    out = a.out_dir.resolve()
    if ROOT == out or ROOT in out.parents:
        raise ValueError('Training outputs must be outside the frozen classification package')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new/empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    raw = out / 'job_input.json'; raw.write_text(json.dumps(job, indent=2)+'\n')
    env = dict(os.environ, TIGER_AMP_RUNS=str(out), TIGER_AMP_DEVICE=a.device,
               OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    cmd = [sys.executable, str(ROOT / 'core' / version / 'worker.py'), str(raw)]
    if version == 'v3': cmd += ['--threads', str(a.threads)]
    subprocess.run(cmd, env=env, check=True)
    destination = out / job['id']
    if job['stage'] == 'final':
        import pandas as pd
        from .metrics import thresholds
        frame = pd.read_csv(destination / 'Vop_predictions.csv', float_precision='round_trip')
        cuts, _ = thresholds(frame.y, frame.probability)
        old = json.loads((destination / 'result.json').read_text())['thresholds']
        # Preserve the original v2 threshold choices; add the v3 recall95/98 operating points.
        if version == 'v2': cuts.update(old)
        (destination / 'release.json').write_text(json.dumps({'model': job['model'], 'seed': job['seed'], 'backend': version, 'thresholds': cuts}, indent=2)+'\n')
    print(f'Completed {job["id"]}: {destination}')

if __name__ == '__main__':
    main()
