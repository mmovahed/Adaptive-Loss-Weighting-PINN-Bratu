"""Self-contained checks for the validation-only static protocol."""
import math
import unittest
from unittest import mock

from utils import static_validation_baseline as p
from experiments import bratu_static_validation_tune as tune
from problems.bratu import BratuProblem


class StaticValidationTests(unittest.TestCase):
    def test_grid_is_complete_and_ordered(self):
        cases = p.tune_cases()
        self.assertEqual(len(cases), 60)
        self.assertEqual(len(p.final_cases()), 20)
        self.assertEqual(cases[0], (0, 1.0, 0.1, 1.0, 2027))
        self.assertEqual(cases[-1], (59, 1.0, 10.0, 3.4, 2039))
        self.assertEqual(p.final_cases()[0], (0, 1.0, 42))
        self.assertEqual(p.final_cases()[-1], (19, 3.4, 1000))
        self.assertTrue(set(p.DEV_SEEDS).isdisjoint(p.FINAL_SEEDS))

    def test_score_aggregation_and_tie_order(self):
        rows = []
        for alpha, beta in p.WEIGHTS:
            for lam in p.LAMBDAS:
                for seed in p.DEV_SEEDS:
                    rows.append({"alpha": alpha, "beta": beta,
                                 "validation_score": 0.5 if beta == 3 else 1.0})
        winner, summaries = p.summarize_validation_trials(rows)
        self.assertEqual((winner["alpha"], winner["beta"]), (1.0, 3.0))
        self.assertAlmostEqual(winner["mean_log_validation_score"], math.log(0.5))
        for row in rows:
            row["validation_score"] = 1.0
        winner, _ = p.summarize_validation_trials(rows)
        self.assertEqual((winner["alpha"], winner["beta"]), p.WEIGHTS[0])

    def test_tune_no_reference_access(self):
        # Tiny smoke: errors must stay unavailable during all tuning operations.
        with mock.patch.object(BratuProblem, "reference_solution", side_effect=AssertionError("reference leaked")):
            with mock.patch.object(tune.cfg, "TOTAL_PINN_STEPS", 1), \
                 mock.patch.object(tune.cfg, "N_COLLOCATION", 8), \
                 mock.patch.object(tune.cfg, "N_VALIDATION", 8):
                result = tune.train_one(1.0, 1.0, 1.0, 2027, "cpu")
        self.assertGreater(result["validation_score"], 0.0)

    def test_production_config_guard(self):
        with mock.patch.object(p.cfg, "TOTAL_PINN_STEPS", 1):
            with self.assertRaisesRegex(ValueError, "Production protocol mismatch"):
                p.validate_config(smoke=False)
            self.assertEqual(p.validate_config(smoke=True)["total_pinn_steps"], 1)


if __name__ == "__main__":
    unittest.main()
