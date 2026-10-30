"""Portable raw-sequence inference with the released calibrated classifiers."""
from pathlib import Path
import importlib.util
import json
from functools import lru_cache
import numpy as np
import joblib
from scipy.special import logit
from threadpoolctl import threadpool_limits
from .features import compact, tokens, validate

ROOT = Path(__file__).resolve().parent
METHODS = json.loads((ROOT / 'methods.json').read_text())['methods']


@lru_cache(maxsize=1)
def _amp553_library():
    path = ROOT / 'third_party/AMPActiPred/predict_lib.py'
    spec = importlib.util.spec_from_file_location('tiger_amp553', path)
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    return lib


def amp553(sequences):
    """The exact AMPActiPred descriptor implementation used by Cascade."""
    import pandas as pd
    lib = _amp553_library()
    data = pd.DataFrame({'Id': [str(i) for i in range(len(sequences))], 'Sequence': sequences})
    result = lib.construct_features(data).drop(columns=['Id', 'Sequence'])
    expected = json.loads((ROOT / 'data/AMP553_schema.json').read_text())['columns']
    if result.columns.tolist() != expected:
        raise ValueError('Cascade descriptor schema changed')
    return result.to_numpy(dtype='float32')


class Predictor:
    def __init__(self, model='CatBoost-Tuned', seed=123, model_dir=None, threads=1):
        if model not in METHODS:
            raise ValueError(f'Choose one of the 15 released methods: {METHODS}')
        if threads < 1:
            raise ValueError('threads must be positive')
        self.path = Path(model_dir) if model_dir else ROOT / 'checkpoints' / model / f'seed{seed}'
        if not (self.path / 'job.json').is_file():
            raise FileNotFoundError(f'{self.path}: seed123 is bundled; train other seeds with the released configs')
        self.job = json.loads((self.path / 'job.json').read_text())
        self.name = self.job['model']
        if self.name != model or self.job['seed'] != seed:
            raise ValueError('Requested model/seed does not match the checkpoint identity')
        release = self.path / 'release.json'
        self.backend = json.loads(release.read_text())['backend'] if release.exists() else ('v3' if model in ['CatBoost-Tuned', 'XGBoost', 'HistGradientBoosting', 'RBFSVM', 'AdaBoost', 'GBDT', 'EBM'] else 'v2')
        self.cut = json.loads(release.read_text())['thresholds'] if release.exists() else json.loads((self.path / 'result.json').read_text())['thresholds']
        self.threads = threads
        self.calibrator = joblib.load(self.path / 'calibrator.joblib')
        if model == 'TinyCNN':
            import torch
            from .networks import SequenceModel
            torch.set_num_threads(threads)
            self.model = SequenceModel(model, self.job['config'])
            state = torch.load(self.path / 'model.pt', map_location='cpu')
            self.model.load_state_dict(state['state_dict'])
            self.model.eval()
        elif model == 'Cascade':
            from deepforest import CascadeForestClassifier
            self.model = CascadeForestClassifier(n_jobs=threads, verbose=0)
            self.model.load(str(self.path / 'cascade'))
            self._cascade_threads(self.model, set())
        else:
            self.model = joblib.load(self.path / 'model.joblib')
            if model in ['RF', 'ExtraTrees', 'LightGBM', 'XGBoost', 'EBM']:
                self.model.set_params(n_jobs=threads)

    def _cascade_threads(self, obj, seen):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, 'n_jobs'):
            obj.n_jobs = self.threads
        values = (obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple))
                  else obj.__dict__.values() if hasattr(obj, '__dict__') else [])
        for value in values:
            if isinstance(value, (dict, list, tuple)) or hasattr(value, 'n_jobs'):
                self._cascade_threads(value, seen)

    def encode(self, sequences):
        sequences = list(sequences)
        if not sequences:
            raise ValueError('At least one peptide is required')
        if self.backend == 'v2':
            validate(sequences)
        feature = self.job['config']['feature']
        if feature == 'tokens':
            return tokens(sequences)
        if feature == 'AMP553':
            return amp553(sequences)
        return compact(sequences, feature)

    def raw_score(self, x):
        if self.name == 'TinyCNN':
            import torch
            with torch.inference_mode():
                score = np.concatenate([self.model(torch.as_tensor(x[i:i + 256])).numpy() for i in range(0, len(x), 256)])
        elif self.name == 'CatBoost-Tuned':
            score = logit(np.clip(self.model.predict_proba(x, thread_count=self.threads)[:, 1], 1e-7, 1-1e-7))
        elif hasattr(self.model, 'decision_function'):
            score = self.model.decision_function(x)
        else:
            score = logit(np.clip(self.model.predict_proba(x)[:, 1], 1e-7, 1-1e-7))
        score = np.asarray(score, dtype=float if self.backend == 'v3' else None).reshape(-1)
        if len(score) != len(x) or not np.isfinite(score).all():
            raise ValueError('Invalid model scores')
        return score

    def predict_proba(self, sequences):
        with threadpool_limits(limits=self.threads):
            score = self.raw_score(self.encode(sequences))
            return self.calibrator.predict_proba(score[:, None])[:, 1]

    def predict(self, sequences, operating_point='recall95'):
        return (self.predict_proba(sequences) >= self.cut[operating_point]).astype(int)


def main(argv=None):
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='CSV with a sequence column, or FASTA')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', choices=METHODS, default='CatBoost-Tuned')
    parser.add_argument('--seed', type=int, choices=range(120, 125), default=123)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--check-reference', action='store_true', help='Require the released 200 Test IDs, order and sequences; compare probabilities')
    a = parser.parse_args(argv)
    if a.output.resolve() == ROOT or ROOT in a.output.resolve().parents:
        raise ValueError('Prediction outputs must be outside the frozen classification package')
    if a.input.suffix.lower() in ['.fa', '.fasta', '.faa']:
        records = []; name = None; sequence = []
        for line in a.input.read_text().splitlines():
            if line.startswith('>'):
                if name is not None: records.append((name, ''.join(sequence)))
                name = line[1:].strip(); sequence = []
            elif line.strip():
                if name is None: raise ValueError('FASTA sequence precedes its identifier')
                sequence.append(line.strip())
        if name is not None: records.append((name, ''.join(sequence)))
        frame = pd.DataFrame(records, columns=['id', 'sequence'])
    else:
        frame = pd.read_csv(a.input)
    if 'sequence' not in frame:
        raise ValueError('Input must have a sequence column')
    model = Predictor(a.model, a.seed, a.model_dir, a.threads)
    p = model.predict_proba(frame.sequence.tolist())
    if a.check_reference:
        ref = pd.read_csv(ROOT / f'results/predictions/{a.model}_seed{a.seed}.csv', float_precision='round_trip')
        if len(frame) != 200 or frame.sequence.tolist() != ref.sequence.tolist() or frame.sample_id.tolist() != ref.sample_id.tolist():
            raise ValueError('Reference comparison requires the exact Test200 IDs and sequence order')
        np.testing.assert_allclose(p, ref.probability, atol=2e-6, rtol=0)
        print(json.dumps({'model': a.model, 'seed': a.seed, 'n': len(p), 'max_absolute_probability_error': float(np.max(np.abs(p-ref.probability))), 'status': 'PASS'}))
    out = frame.copy(); out['probability_AMP_balanced_cohort'] = p
    for point, cut in model.cut.items(): out['pred_' + point] = (p >= cut).astype(int)
    a.output.parent.mkdir(parents=True, exist_ok=True); out.to_csv(a.output, index=False)

if __name__ == '__main__':
    main()
