"""Numerical checks for the paper's new branch plots and figure source data.

Run: python -m unittest discover -s tests -p test_figure_repro.py
"""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MOD=ROOT/'scripts'/'figures'/'make_branch_figures.py'
spec=importlib.util.spec_from_file_location('make_branch_figures',MOD)
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

class FigureData(unittest.TestCase):
    def test_original_case_counts(self):
        final=pd.read_csv(ROOT/'results'/'final'/'final_results.csv')
        cand=pd.read_csv(ROOT/'results'/'final'/'candidate_results.csv')
        self.assertEqual(len(final),100)
        self.assertEqual(len(cand),500)
        self.assertTrue(final['state'].eq('success').all())
        self.assertTrue(final['n_candidates'].eq(5).all())
        self.assertEqual(len(final[['lambda','seed','method']].drop_duplicates()),100)
        self.assertEqual(len(cand.loc[cand['selected']==1]),100)
    def test_bifurcation_math(self):
        for lam in b.LAMBDAS:
            lo,hi,critical,lamcrit=b.roots(lam)
            self.assertLess(lo,critical)
            self.assertGreater(hi,critical)
            self.assertAlmostEqual(float(b.lambda_at_a(lo)),lam,places=10)
            self.assertAlmostEqual(float(b.lambda_at_a(hi)),lam,places=10)
            self.assertAlmostEqual(b.profile(lo,0.),0.,places=10)
            self.assertAlmostEqual(b.profile(hi,1.),0.,places=10)
        self.assertAlmostEqual(lamcrit,3.513830719,places=8)
    def test_example_prediction_branch_distances(self):
        lo,hi,_,_=b.roots(3.4)
        z=b.selected_file(ROOT/'results','ddqn',3.4,456)
        x=z['x'].ravel();y=z['y_pred'].ravel()
        rms_l=float(np.sqrt(np.mean((y-b.profile(lo,x))**2)))
        rms_h=float(np.sqrt(np.mean((y-b.profile(hi,x))**2)))
        self.assertAlmostEqual(rms_l,.0603,delta=.0001)
        self.assertAlmostEqual(rms_h,.4789,delta=.0001)
    def test_all_overlays_available(self):
        for lam in b.LAMBDAS:
            for m in b.METHODS:
                with b.selected_file(ROOT/'results',m,lam,123) as z:
                    self.assertEqual(z['x'].size,2000)
                    self.assertEqual(z['y_pred'].size,2000)
    def test_all_figures_exist(self):
        output=ROOT/'build'/'figures'
        for i in range(1,10):
            self.assertGreater((output/f'Fig{i}.pdf').stat().st_size,1000)

if __name__=='__main__':unittest.main()
