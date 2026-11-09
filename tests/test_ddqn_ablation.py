import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np
from utils.ddqn_ablation_support import normalized_action, decode_array_index


class DDQNNormalizedTests(unittest.TestCase):
    def setUp(self):
        self.m=np.asarray([0.8,0.9,1.,1.1,1.2], dtype=np.float32)

    def test_sum_and_floor(self):
        for alpha,beta in ((1,1), (0.01,1.99),(1.99,0.01),(1.5,0.5)):
            for action in range(25):
                a,b=normalized_action(alpha,beta,action,self.m,0.01)
                self.assertAlmostEqual(a+b,2.0,places=12)
                self.assertGreaterEqual(a,0.01-1e-12)
                self.assertGreaterEqual(b,0.01-1e-12)
                self.assertLessEqual(a,1.99+1e-12)
                self.assertLessEqual(b,1.99+1e-12)

    def test_equal_multiplier_is_noop(self):
        a,b=normalized_action(0.6,1.4,12,self.m)
        self.assertAlmostEqual(a,0.6,places=6)
        self.assertAlmostEqual(b,1.4,places=6)

    def test_action_preserves_ratio_without_clamp(self):
        a,b=normalized_action(1.,1.,4,self.m)
        self.assertAlmostEqual(a/b, float(self.m[0])/float(self.m[4]),places=6)

    def test_array_mapping(self):
        self.assertEqual(decode_array_index(0),("original_control",1.0,42))
        self.assertEqual(decode_array_index(19),("original_control",3.4,1000))
        self.assertEqual(decode_array_index(20),("normalized",1.0,42))
        self.assertEqual(decode_array_index(39),("normalized",3.4,1000))
        for i in (-1,40):
            with self.assertRaises(ValueError):
                decode_array_index(i)


if __name__ == '__main__':
    unittest.main()
