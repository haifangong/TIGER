"""Boundary, feature-identity and deployment-contract checks for the release."""
from pathlib import Path
import json
import unittest
import numpy as np
import pandas as pd
from amp_classification.features import compact, tokens, validate
from amp_classification.replay import independent_metrics
from amp_classification.predict import Predictor, main as predict_cli
from amp_classification.metrics import thresholds

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_input_rejects_invalid_and_preserves_long_sequences(self):
        for sequence in ['AAA', 'A'*51, 'ACDXEF', 'acdefg', 'ACD-EFG']:
            with self.subTest(sequence=sequence), self.assertRaises(ValueError): validate([sequence])
        encoded = tokens(['A'*30+'K'*20])
        self.assertEqual(encoded.shape, (1, 50))
        self.assertTrue(np.all(encoded[0, 30:] != 0))

    def test_frozen_features(self):
        test = pd.read_csv(ROOT / 'data/Test.csv')
        for feature in ['F1', 'F2']:
            np.testing.assert_array_equal(compact(test.sequence.tolist(), feature), np.load(ROOT / f'data/Test_{feature}.npy'))

    def test_no_positive_predictions_remain_undefined_precision(self):
        y = np.array([0, 0, 1, 1]); prob = np.full(4, .5)
        result = independent_metrics(y, prob, np.zeros(4, int), 'AMP')
        self.assertTrue(np.isnan(result['Precision']))
        self.assertEqual(result['Recall'], 0.)
        self.assertEqual(result['AUPRC'], .5)
        self.assertEqual(result['AUROC'], .5)

    def test_dummy_threshold_ties(self):
        cuts, _ = thresholds(np.array([0, 1, 0, 1]), np.full(4, .5))
        self.assertGreater(cuts['balanced'], 1.)
        self.assertEqual(cuts['recall95'], 0.)
        self.assertEqual(cuts['recall98'], 0.)

    def test_complemented_class_confusion(self):
        y = np.array([0, 0, 1, 1]); p = np.array([.1, .8, .7, .9]); hard = (p >= .5).astype(int)
        a = independent_metrics(y, p, hard, 'AMP'); b = independent_metrics(y, p, hard, 'non-AMP')
        self.assertEqual(a['AUROC'], b['AUROC'])
        self.assertEqual([a[k] for k in ['TP', 'FP', 'FN', 'TN']], [2, 1, 0, 1])
        self.assertEqual([b[k] for k in ['TP', 'FP', 'FN', 'TN']], [1, 0, 1, 2])

    def test_registry_contains_exact_requested_methods(self):
        methods = json.loads((ROOT / 'methods.json').read_text())['methods']
        self.assertEqual(len(set(methods)), 15)
        self.assertFalse(set(methods) & {'DeepAMP', 'CatBoost-Recall', 'Das', 'CatBoost', 'APEX', 'Logistic'})

    def test_checkpoint_identity_cannot_be_relabelled(self):
        with self.assertRaises(ValueError):
            Predictor('CatBoost-Tuned', seed=120, model_dir=ROOT / 'checkpoints/CatBoost-Tuned/seed123')

    def test_frozen_output_guard(self):
        with self.assertRaises(ValueError):
            predict_cli(['--input', str(ROOT/'data/Test.csv'), '--output', str(ROOT/'data/Test.csv')])


if __name__ == '__main__': unittest.main()
