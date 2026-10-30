"""Verify release bytes, split identities, labels and CV separation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent


def main(argv=None):
    if argv: raise ValueError('verify takes no arguments')
    manifest = json.loads((ROOT / 'provenance/manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Payload checksum mismatch: {name}')
    full = pd.read_csv(ROOT / 'data/samples.csv')
    if not full.sample_id.is_unique or not full.sequence.is_unique or len(full) != 10962:
        raise ValueError('Cohort identity/count mismatch')
    if full.y.value_counts().to_dict() != {0: 5481, 1: 5481}: raise ValueError('Cohort class balance differs')
    measured = full.label_evidence.eq('measured_MIC')
    np.testing.assert_array_equal(full.loc[measured, 'y'], (full.loc[measured, 'log2_MIC'] < 7).astype(int))
    if not full.loc[~measured, 'MIC_ug_mL'].isna().all() or not full.loc[~measured, 'y'].eq(0).all():
        raise ValueError('Background negative label semantics changed')
    partitions = {}
    for name, count in [('T', 8610), ('Vcal', 1076), ('Vop', 1076), ('Test', 200)]:
        df = pd.read_csv(ROOT / f'data/{name}.csv')
        expect = full.loc[full.split_primary.eq(name)]
        if len(df) != count or df.sample_id.tolist() != expect.sample_id.tolist(): raise ValueError(f'{name} membership differs')
        partitions[name] = set(df.sample_id)
    train = partitions['T']; scored = []
    for fold in json.loads((ROOT / 'data/folds_primary.json').read_text()):
        fit, score, ef, es = [set(fold[k]) for k in ['fit_ids', 'score_ids', 'early_fit_ids', 'early_stop_ids']]
        if fit & score or ef & es or (fit | score) != train or (ef | es) != fit:
            raise ValueError('CV fit/score or inner early-stopping separation failed')
        scored.extend(score)
    if len(scored) != len(train) or set(scored) != train: raise ValueError('CV does not score every training peptide once')
    print(json.dumps({'status': 'PASS', 'payload_files': len(manifest['sha256']), 'cohort': 10962, 'test': 200, 'folds': 5}))

if __name__ == '__main__': main()
