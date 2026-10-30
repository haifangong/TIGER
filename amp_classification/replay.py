"""Independently reproduce the archived 15-method, five-seed Test200 tables."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
METRICS = ['AUROC', 'AUPRC', 'Precision', 'Recall', 'F1_score']


def independent_metrics(y, probability, predicted, label):
    # Complement hard predictions as well as labels; -p preserves ordering without
    # creating extra floating-point ties through subtraction from one.
    if label == 'non-AMP': y, probability, predicted = 1-y, -probability, 1-predicted
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(AUROC=float(roc_auc_score(y, probability)), AUPRC=float(average_precision_score(y, probability)),
                Precision=tp/(tp+fp) if tp+fp else np.nan, Recall=tp/(tp+fn),
                F1_score=2*tp/(2*tp+fp+fn), TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))


def replay():
    names = json.loads((ROOT / 'methods.json').read_text())['methods']
    registry = json.loads((ROOT / 'results/registry.json').read_text())
    expected = {(n, s) for n in names for s in range(120, 125)}
    if {(r['model'], r['seed']) for r in registry} != expected or len(registry) != 75:
        raise ValueError('The 15 × 5 model/seed registry is incomplete or duplicated')
    test = pd.read_csv(ROOT / 'data/Test.csv')
    if len(test) != 200 or test.y.value_counts().to_dict() != {0: 100, 1: 100}:
        raise ValueError('Unexpected Test200 membership')
    rows = []; threshold_checks = 0
    for record in registry:
        name, seed = record['model'], record['seed']
        pred = pd.read_csv(ROOT / f'results/predictions/{name}_seed{seed}.csv', float_precision='round_trip')
        if pred.sample_id.tolist() != test.sample_id.tolist() or pred.sequence.tolist() != test.sequence.tolist():
            raise ValueError(f'Prediction identity/order mismatch: {name}, {seed}')
        np.testing.assert_array_equal(pred.y, test.y)
        prob = pred.probability.to_numpy(); y = pred.y.to_numpy(dtype=int)
        if not (np.isfinite(prob).all() and np.all((prob >= 0) & (prob <= 1))):
            raise ValueError('Invalid probabilities')
        from .metrics import thresholds
        vop = pd.read_csv(ROOT / f'results/validation/{name}/seed{seed}/Vop_predictions.csv', float_precision='round_trip')
        target = pd.read_csv(ROOT / 'data/Vop.csv')
        if vop.sample_id.tolist() != target.sample_id.tolist():
            raise ValueError('Operating-point validation identities differ')
        np.testing.assert_array_equal(vop.y, target.y)
        cuts, _ = thresholds(vop.y, vop.probability)
        for point, cutoff in record['thresholds'].items():
            np.testing.assert_allclose(cutoff, cuts[point], atol=1e-12, rtol=0)
            threshold_checks += 1
            hard = (prob >= cutoff).astype(int)
            np.testing.assert_array_equal(hard, pred['pred_' + point])
            for label in ['AMP', 'non-AMP']:
                rows.append(dict(model=name, seed=seed, operating_point=point, set='Test200', n=200, **{'class': label},
                                 **independent_metrics(y, prob, hard, label)))
    table = pd.DataFrame(rows)
    keys = ['model', 'seed', 'operating_point', 'class']
    numeric = METRICS + ['TP', 'FP', 'FN', 'TN']
    ref = pd.read_csv(ROOT / 'results/tables/test_classwise_per_seed.csv', float_precision='round_trip')
    if len(ref) != len(table): raise ValueError('Reference row count differs')
    pd.testing.assert_frame_equal(table.set_index(keys)[numeric].sort_index(), ref.set_index(keys)[numeric].sort_index(),
                                  check_dtype=False, atol=1e-12, rtol=0)
    grouped = table.groupby(['model', 'operating_point', 'class'])[METRICS].agg(['mean', 'std', 'count'])
    grouped.columns = ['_'.join(c) for c in grouped.columns]
    grouped = grouped.reset_index()
    ref = pd.read_csv(ROOT / 'results/tables/test_classwise_summary.csv', float_precision='round_trip')
    keys = ['model', 'operating_point', 'class']
    pd.testing.assert_frame_equal(grouped.set_index(keys).sort_index(), ref.set_index(keys).sort_index(),
                                  check_dtype=False, atol=1e-12, rtol=0)
    check = {'status': 'PASS', 'methods': 15, 'seeds': 5, 'predictions_checked': 75,
             'test_peptides': 200, 'thresholds_checked_on_Vop': threshold_checks,
             'classwise_rows': len(table), 'metric_and_count_comparisons': len(table)*len(numeric)+len(grouped)*(len(grouped.columns)-3),
             'undefined_precision_preserved_as_NA': True, 'reference_data_unchanged': True}
    return table, grouped, check


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT.parent / 'outputs/amp_replay')
    a = p.parse_args(argv); output = a.output.resolve()
    if output == ROOT or ROOT in output.parents: raise ValueError('Use an output directory outside the frozen classification package')
    table, summary, check = replay()
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / 'test_classwise_per_seed.csv', index=False)
    summary.to_csv(output / 'test_classwise_summary.csv', index=False)
    summary.query('operating_point=="balanced" and `class`=="AMP"').sort_values('AUROC_mean', ascending=False).to_csv(output / 'AUROC_ranked_AMP.csv', index=False)
    (output / 'validation.json').write_text(json.dumps(check, indent=2)+'\n')
    print(json.dumps(check, indent=2))

if __name__ == '__main__': main()
