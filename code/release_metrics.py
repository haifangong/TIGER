"""Identity-aligned family metrics, shared with the V17 benchmark release."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
KEYS=["family","anchor_sequence","query_sequence"]
METRICS=["PCC","KCC","log10MAE","RSE"]

def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned_prediction(frame, pairs):
    """Validate identities and labels, then align by identity (never row number)."""
    require(set(KEYS+['y_true', 'y_pred']).issubset(frame), 'Missing pair columns')
    require(not frame.duplicated(KEYS).any(), 'Duplicate pair identities')
    require(set(map(tuple, frame[KEYS].to_numpy())) == set(map(tuple, pairs[KEYS].to_numpy())), 'Incomplete or unexpected endpoint')
    aligned = pairs[KEYS].merge(frame, on=KEYS, validate='one_to_one', sort=False)
    require(np.allclose(aligned.y_true, pairs.y_true, rtol=0, atol=1e-10), 'Pair labels differ from canonical labels')
    pred = aligned.y_pred.to_numpy(float)
    require(np.isfinite(pred).all(), 'Nonfinite predictions')
    return pred


def family_metrics(truth, pred):
    """Primary policy: undefined correlations propagate; never nanmean."""
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    require(truth.shape == pred.shape and np.isfinite(truth).all() and np.isfinite(pred).all(), 'Invalid metric inputs')
    sst = np.square(truth-truth.mean()).sum()
    require(sst > 0, 'Constant family target: RSE undefined')
    constant = np.ptp(pred) == 0
    error = pred-truth
    return {
        'PCC': np.nan if constant else float(stats.pearsonr(truth, pred).statistic),
        'KCC': np.nan if constant else float(stats.kendalltau(truth, pred, variant='b').statistic),
        'log10MAE': float(np.abs(error).mean()*np.log10(2)),
        'RSE': float(np.square(error).sum()/sst),
    }


def holm(p):
    p = np.asarray(p, float)
    result = np.full(len(p), np.nan)
    ix = np.flatnonzero(np.isfinite(p))
    order = ix[np.argsort(p[ix], kind='stable')]
    result[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(order)))))
    return result


def compare(actual, reference, keys, columns, atol=1e-9):
    require(not actual.duplicated(keys).any() and not reference.duplicated(keys).any(), f'Duplicate comparison keys: {keys}')
    a, b = actual.set_index(keys).sort_index(), reference.set_index(keys).sort_index()
    require(a.index.equals(b.index), f'Reference identities differ: {keys}')
    for col in columns:
        require(np.allclose(a[col], b[col], atol=atol, rtol=0, equal_nan=True), f'Reference mismatch: {col}')
    return len(a)*len(columns)
