from __future__ import annotations

import unittest

import numpy as np

from brian2026.phase31_experiment import (
    POST_DIAGNOSTIC_DECLARATION,
    SCORE_QUANTILES,
    CAPITAL_FRACTIONS,
    _candidate,
    _split_validation,
)


class Phase31ExperimentTests(unittest.TestCase):
    def test_validation_split_is_chronological_disjoint_and_embargoed(self):
        rows = np.arange(1000, dtype=int)
        calibration, policy = _split_validation(rows)
        self.assertGreater(len(calibration), 0)
        self.assertGreater(len(policy), 0)
        self.assertLess(int(calibration[-1]), int(policy[0]))
        self.assertGreater(int(policy[0]) - int(calibration[-1]), 12)

    def test_preregistered_search_spaces_are_fixed_and_conservative(self):
        self.assertEqual(SCORE_QUANTILES, (0.75, 0.85, 0.90, 0.95, 0.975))
        self.assertEqual(CAPITAL_FRACTIONS, (0.05, 0.10, 0.15, 0.20, 0.25))
        self.assertIn("NOT PRISTINE OOS", POST_DIAGNOSTIC_DECLARATION)

    def test_candidate_rejects_zero_edge_and_zero_capital_deployment(self):
        fold = {
            "cost_sensitivity": {
                "BASE": {"entries": 0, "expectancy": 0.0, "profit_factor": None, "net_pnl": 0.0},
                "STRESS": {"net_pnl": 0.0},
            },
            "capital_policy": {"deployable": False, "ruin_probability": 0.0},
        }
        result = _candidate([fold, fold, fold])
        self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")
        self.assertFalse(result["all_required_gates_passed"])
        self.assertIn("capital deployment validation gate failed", result["reasons"])

    def test_candidate_is_shadow_only_even_when_synthetic_gates_pass(self):
        fold = {
            "cost_sensitivity": {
                "BASE": {"entries": 60, "expectancy": 1.0, "profit_factor": 1.2, "net_pnl": 10.0},
                "STRESS": {"net_pnl": 5.0},
            },
            "capital_policy": {"deployable": True, "ruin_probability": 0.01},
        }
        result = _candidate([fold, fold, fold])
        self.assertEqual(result["status"], "SHADOW_CANDIDATE")
        self.assertTrue(result["all_required_gates_passed"])
        self.assertTrue(result["shadow_only"])
        self.assertFalse(result["final_champion"])


if __name__ == "__main__":
    unittest.main()
