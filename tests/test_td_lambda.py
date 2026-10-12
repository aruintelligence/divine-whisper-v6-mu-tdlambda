import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from divine_whisper_v6_multimodal_mu_tdlambda import (
    MuField,
    TDLambdaValue,
    benchmark,
    run_demo,
)


class TemporalCreditTests(unittest.TestCase):
    def test_lambda_propagates_delayed_reward_to_previous_state(self):
        previous = np.array([1.0, 0.0])
        final = np.array([0.0, 1.0])
        td0 = TDLambdaValue(2, alpha=0.1, lambda_=0.0)
        td8 = TDLambdaValue(2, alpha=0.1, lambda_=0.8)
        for learner in (td0, td8):
            learner.update(previous, 0.0, final)
            learner.update(final, 1.0, None)
            self.assertTrue(np.array_equal(learner.eligibility, np.zeros(2)))
        self.assertAlmostEqual(td0.weights[0], 0.0)
        self.assertAlmostEqual(td8.weights[0], 0.08)
        self.assertAlmostEqual(td0.weights[1], 0.1)
        self.assertAlmostEqual(td8.weights[1], 0.1)

    def test_terminal_bootstrap_is_zero_and_trace_resets(self):
        learner = TDLambdaValue(1, alpha=0.5, gamma=0.9, lambda_=1.0)
        result = learner.update(np.array([1.0]), 1.0, None)
        self.assertAlmostEqual(result["td_error"], 1.0)
        self.assertAlmostEqual(learner.weights[0], 0.5)
        self.assertAlmostEqual(learner.eligibility[0], 0.0)
        learner.update(np.array([1.0]), 0.0, None)
        self.assertAlmostEqual(learner.weights[0], 0.25)

    def test_benchmark_is_paired_and_learns_known_values(self):
        report = benchmark(seed=42, episodes=200, replicates=8, lambda_=0.9)
        self.assertEqual(report, benchmark(seed=42, episodes=200, replicates=8, lambda_=0.9))
        self.assertEqual(report["true_values"], [i / 6 for i in range(1, 6)])
        initial = report["methods"]["td0"]["mean_rmse_at_sample"][0]
        for method in report["methods"].values():
            self.assertLess(np.mean(method["final_rmse_by_replicate"]), initial)
            self.assertEqual(len(method["final_rmse_by_replicate"]), 8)

    def test_demo_reproducible_and_lambda_changes_credit(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one = run_demo(42, 12, 0.9, Path(first))
            two = run_demo(42, 12, 0.9, Path(second))
            self.assertEqual(one, two)
            for filename in ("heaven_ledger.jsonl", "coherence_trajectory.json"):
                self.assertEqual(
                    (Path(first) / filename).read_bytes(),
                    (Path(second) / filename).read_bytes(),
                )
            td0 = run_demo(42, 12, 0.0, Path(second))
            self.assertEqual(one["coherence_history"], td0["coherence_history"])
            self.assertNotEqual(one["value_weights"], td0["value_weights"])
            self.assertEqual(sum(row["reward"] for row in one["steps"]), 1.0)
            self.assertEqual(
                json.loads((Path(first) / "coherence_trajectory.json").read_text()),
                one,
            )

    def test_field_reports_concentration_not_learning(self):
        field = MuField.new(dim=4)
        self.assertAlmostEqual(field.coherence, 0.0)
        field.update(np.array([1.0, 0.0, 0.0, 0.0]), alpha=1.0)
        self.assertGreater(field.coherence, 0.99)
        with self.assertRaises(ValueError):
            field.update(np.ones(3), alpha=1.0)


if __name__ == "__main__":
    unittest.main()
