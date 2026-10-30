"""Repeat the archived CPU workload protocol; write new timings separately."""
from pathlib import Path
import argparse, json, os, time
ROOT = Path(__file__).resolve().parent

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 200, 1000, 10000])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.threads < 1 or any(n < 1 for n in a.batches): raise ValueError('Use positive thread and batch counts')
    if ROOT == a.output.resolve() or ROOT in a.output.resolve().parents: raise ValueError('Do not overwrite archived timings')
    os.environ.update(OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    import numpy as np, pandas as pd
    from threadpoolctl import threadpool_limits
    from .predict import Predictor
    seqs = pd.read_csv(ROOT / 'data/Vop.csv').sort_values('sample_id').sequence.tolist()
    one = sorted(seqs, key=lambda s: (len(s), s))[len(seqs)//2]
    rows = []
    with threadpool_limits(limits=a.threads):
        model = Predictor(a.model, threads=a.threads)
        # Thread context and model loading are outside the measured region,
        # matching the original timer, rather than timing the convenience API.
        def predict(seq):
            return model.calibrator.predict_proba(model.raw_score(model.encode(seq))[:, None])[:, 1]
        for n in a.batches:
            seq = [one] if n == 1 else (seqs * (n//len(seqs)+1))[:n]
            for _ in range(3): predict(seq)
            elapsed = []
            for _ in range(30):
                start = time.perf_counter_ns(); values = predict(seq)
                elapsed.append((time.perf_counter_ns()-start)/1e6)
            if len(values) != n or not np.isfinite(values).all(): raise ValueError('Invalid timing endpoint')
            rows.append(dict(n=n, median_ms=float(np.median(elapsed)), p95_ms=float(np.quantile(elapsed, .95)), measurements_ms=elapsed))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(model=a.model, seed=123, threads=a.threads, warmup=3, repeats=30,
        endpoint='Raw Vop sequences -> calibrated AMP probability', results=rows), indent=2)+'\n')
    print(a.output)

if __name__ == '__main__': main()
