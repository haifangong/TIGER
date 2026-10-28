"""Scientific failure modes that a result-replay gate must reject."""
import unittest
import numpy as np
import pandas as pd
from code.release_metrics import aligned_prediction,family_metrics,holm

class ReleaseMetrics(unittest.TestCase):
    def setUp(self):
        self.pairs=pd.DataFrame({'family':['F']*3,'anchor_sequence':['AA','AA','BB'],
            'query_sequence':['BB','CC','CC'],'y_true':[1.,2.,1.]})
        self.frame=self.pairs.assign(y_pred=[.7,1.9,.9])
    def test_reordered_rows_align_by_identity(self):
        np.testing.assert_array_equal(aligned_prediction(self.frame.iloc[::-1],self.pairs),[.7,1.9,.9])
    def test_wrong_unit_rejected(self):
        with self.assertRaises(ValueError):
            aligned_prediction(self.frame.assign(y_true=self.frame.y_true*np.log10(2)),self.pairs)
    def test_missing_and_duplicate_pairs_rejected(self):
        for bad in [self.frame.iloc[:-1],pd.concat([self.frame.iloc[:2],self.frame.iloc[:1]])]:
            with self.assertRaises(ValueError):aligned_prediction(bad,self.pairs)
    def test_nonfinite_prediction_rejected(self):
        bad=self.frame.copy();bad.loc[0,'y_pred']=np.nan
        with self.assertRaises(ValueError):aligned_prediction(bad,self.pairs)
    def test_affine_pcc_does_not_validate_amplitudes(self):
        y=np.array([-2.,0.,1.,4.]);m=family_metrics(y,3*y+2)
        self.assertAlmostEqual(m['PCC'],1.)
        self.assertGreater(m['RSE'],1.)
    def test_constant_prediction_propagates_undefined(self):
        m=family_metrics(np.array([1.,2.,3.]),np.ones(3))
        self.assertTrue(np.isnan(m['PCC']));self.assertTrue(np.isnan(m['KCC']))
    def test_holm_joint_164_tests(self):
        self.assertAlmostEqual(holm(np.full(164,0.001953125)).min(),0.3203125)

if __name__=='__main__':unittest.main()
