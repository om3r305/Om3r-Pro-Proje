from __future__ import annotations

import math
import unittest

from brian2026.adaptive_quant import ChallengerVote, DriftAssessment, FamiliarityAssessment
from brian2026.capital_efficiency import (
    apply_score_threshold,
    bootstrap_risk_of_ruin,
    choose_capital_policy,
    score_opportunity,
    select_score_threshold,
)
from brian2026.portfolio import DEVELOPMENT_CUTOFF


class CapitalEfficiencyTests(unittest.TestCase):
    def fam(self, *, ood=False):
        return FamiliarityAssessment(0.5, 2.0, 0.9, 1.0, ood)

    def drift(self, *, drifted=False, hard=False):
        return DriftAssessment(0.2, 1.0, drifted, hard, 96)

    def votes(self):
        return [
            ChallengerVote("a", "BUY", 0.40, 0.70, 0.40, "a"),
            ChallengerVote("b", "SELL", -0.10, 0.60, 0.20, "b"),
            ChallengerVote("c", "BUY", 0.25, 0.65, 0.40, "c"),
        ]

    def test_soft_disagreement_penalizes_but_does_not_hard_veto(self):
        row = score_opportunity(1_700_000_000.0, self.votes(), self.fam(), self.drift())
        self.assertEqual(row.proposed_action, "BUY")
        self.assertGreater(row.edge, 0)
        self.assertGreater(row.directional_support, 0.5)
        self.assertGreater(row.dispersion, 0)
        self.assertGreater(row.opportunity_score, 0)
        self.assertFalse(row.veto_reasons)

    def test_ood_and_hard_drift_remain_hard_vetoes(self):
        ood = score_opportunity(1_700_000_000.0, self.votes(), self.fam(ood=True), self.drift())
        hard = score_opportunity(1_700_000_000.0, self.votes(), self.fam(), self.drift(drifted=True, hard=True))
        self.assertEqual(ood.proposed_action, "WAIT")
        self.assertEqual(hard.proposed_action, "WAIT")
        self.assertTrue(ood.veto_reasons)
        self.assertTrue(hard.veto_reasons)

    def test_2026_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, "INVALID_CONTAMINATED"):
            score_opportunity(DEVELOPMENT_CUTOFF, self.votes(), self.fam(), self.drift())

    def test_threshold_uses_only_non_wait_candidates(self):
        receipt = select_score_threshold([0.1, 0.2, 0.9, 0.8], ["WAIT", "BUY", "SELL", "BUY"], 0.5)
        self.assertGreaterEqual(receipt.threshold, 0.2)
        self.assertLessEqual(receipt.threshold, 0.9)
        self.assertEqual(receipt.selection_partition, "policy_validation_only")

    def test_apply_threshold_never_turns_wait_into_trade(self):
        rows = [
            score_opportunity(1_700_000_000.0 + i, self.votes(), self.fam(), self.drift())
            for i in range(2)
        ]
        actions = apply_score_threshold(rows, rows[0].opportunity_score + 1e-6)
        self.assertEqual(actions, ["WAIT", "WAIT"])

    def test_bootstrap_ruin_is_deterministic_and_zero_when_flat(self):
        returns = [0.02, -0.01, 0.015, -0.005]
        first = bootstrap_risk_of_ruin(returns, 0.25, seed=123, trials=200, horizon_trades=50)
        second = bootstrap_risk_of_ruin(returns, 0.25, seed=123, trials=200, horizon_trades=50)
        self.assertEqual(first, second)
        self.assertEqual(bootstrap_risk_of_ruin(returns, 0.0, seed=123), 0.0)

    def test_capital_stays_flat_without_validation_edge(self):
        policy = choose_capital_policy([
            {"fraction": 0.25, "ruin_probability": 0.0, "net_pnl": -1.0,
             "profit_factor": 2.0, "entries": 100, "max_drawdown_pct": 1.0}
        ])
        self.assertFalse(policy.deployable)
        self.assertEqual(policy.equity_fraction, 0.0)

    def test_capital_policy_selects_only_gate_passing_candidate(self):
        policy = choose_capital_policy([
            {"fraction": 0.10, "ruin_probability": 0.01, "net_pnl": 4.0,
             "profit_factor": 1.20, "entries": 30, "max_drawdown_pct": 3.0},
            {"fraction": 0.25, "ruin_probability": 0.20, "net_pnl": 20.0,
             "profit_factor": 1.50, "entries": 30, "max_drawdown_pct": 4.0},
        ])
        self.assertTrue(policy.deployable)
        self.assertEqual(policy.equity_fraction, 0.10)
        self.assertEqual(policy.starting_equity, 500.0)


if __name__ == "__main__":
    unittest.main()
