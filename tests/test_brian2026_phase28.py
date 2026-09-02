import unittest

from brian2026.expert_reasoner import ExpertReasonerConfig, reason_market
from brian2026.portfolio import DEVELOPMENT_CUTOFF


def bullish_snapshot():
    return {
        "structure_state": 1.0,
        "structure_15m": 1.0,
        "structure_1h": 1.0,
        "ema_slope": 0.2,
        "bullish_bos": 1.0,
        "bearish_bos": 0.0,
        "bullish_choch": 0.0,
        "bearish_choch": 0.0,
        "bullish_sweep": 1.0,
        "bearish_sweep": 0.0,
        "failed_breakdown": 1.0,
        "failed_breakout": 0.0,
        "bullish_breakout_retest": 1.0,
        "bearish_breakout_retest": 0.0,
        "bullish_rsi_divergence": 1.0,
        "bearish_rsi_divergence": 0.0,
        "support_distance_atr": 0.2,
        "resistance_distance_atr": 2.0,
        "nearest_support": 98.0,
        "nearest_resistance": 110.0,
        "dip_score": 0.90,
        "rally_score": 0.10,
        "rsi": 60.0,
        "acceleration": 0.1,
        "return_1": 0.2,
        "relative_volume": 2.0,
        "volume_zscore": 2.0,
        "pullback_volume_contraction": 1.0,
        "selling_exhaustion_proxy": 1.0,
        "buying_exhaustion_proxy": 0.0,
        "zscore": -1.0,
        "bb_position": 0.2,
        "range_expansion": 1.2,
        "inside_support_zone": 1.0,
        "inside_resistance_zone": 0.0,
        "lower_wick_ratio": 0.55,
        "upper_wick_ratio": 0.10,
    }


def bearish_snapshot():
    row = bullish_snapshot()
    swaps = {
        "structure_state": -1.0,
        "structure_15m": -1.0,
        "structure_1h": -1.0,
        "ema_slope": -0.2,
        "bullish_bos": 0.0,
        "bearish_bos": 1.0,
        "bullish_sweep": 0.0,
        "bearish_sweep": 1.0,
        "failed_breakdown": 0.0,
        "failed_breakout": 1.0,
        "bullish_breakout_retest": 0.0,
        "bearish_breakout_retest": 1.0,
        "bullish_rsi_divergence": 0.0,
        "bearish_rsi_divergence": 1.0,
        "support_distance_atr": 2.0,
        "resistance_distance_atr": 0.2,
        "nearest_support": 90.0,
        "nearest_resistance": 102.0,
        "dip_score": 0.10,
        "rally_score": 0.90,
        "rsi": 40.0,
        "acceleration": -0.1,
        "return_1": -0.2,
        "selling_exhaustion_proxy": 0.0,
        "buying_exhaustion_proxy": 1.0,
        "zscore": 1.0,
        "bb_position": 0.8,
        "inside_support_zone": 0.0,
        "inside_resistance_zone": 1.0,
        "lower_wick_ratio": 0.10,
        "upper_wick_ratio": 0.55,
    }
    row.update(swaps)
    return row


class Phase28ExpertReasonerTests(unittest.TestCase):
    def test_strong_bullish_evidence_requires_objective_setup_and_buys(self):
        decision = reason_market(bullish_snapshot(), timestamp=1_700_000_000.0)
        self.assertEqual(decision.action, "BUY")
        self.assertEqual(decision.setup, "FAILED_BREAK_REVERSAL")
        self.assertGreaterEqual(decision.agreement, ExpertReasonerConfig().min_agreement)
        self.assertEqual(decision.invalidation_level, 98.0)
        self.assertTrue(decision.shadow_only)

    def test_strong_bearish_evidence_sells_with_resistance_invalidation(self):
        decision = reason_market(bearish_snapshot(), timestamp=1_700_000_000.0)
        self.assertEqual(decision.action, "SELL")
        self.assertEqual(decision.setup, "FAILED_BREAK_REVERSAL")
        self.assertEqual(decision.invalidation_level, 102.0)

    def test_missing_completed_higher_timeframes_is_risk_vetoed(self):
        row = bullish_snapshot()
        row["structure_15m"] = float("nan")
        row["structure_1h"] = float("nan")
        decision = reason_market(row, timestamp=1_700_000_000.0)
        self.assertEqual(decision.action, "WAIT")
        critic = [x for x in decision.experts if x.name == "risk_critic"][0]
        self.assertTrue(critic.veto)

    def test_no_clear_setup_abstains_even_when_some_indicators_are_positive(self):
        row = bullish_snapshot()
        for key in (
            "bullish_bos", "bullish_sweep", "failed_breakdown",
            "bullish_breakout_retest", "bullish_rsi_divergence",
            "inside_support_zone", "selling_exhaustion_proxy",
        ):
            row[key] = 0.0
        row["dip_score"] = 0.50
        decision = reason_market(row, timestamp=1_700_000_000.0)
        self.assertEqual(decision.setup, "NO_CLEAR_SETUP")
        self.assertEqual(decision.action, "WAIT")

    def test_unknown_future_outcome_fields_cannot_change_decision(self):
        row = bullish_snapshot()
        left = reason_market({**row, "future_return": 99.0, "future_label": 1.0}, timestamp=1_700_000_000.0)
        right = reason_market({**row, "future_return": -99.0, "future_label": -1.0}, timestamp=1_700_000_000.0)
        self.assertEqual(left, right)

    def test_scenario_strengths_are_bounded_and_not_presented_as_probabilities(self):
        decision = reason_market(bullish_snapshot(), timestamp=1_700_000_000.0)
        for scenario in (decision.bull_case, decision.bear_case, decision.no_trade_case):
            self.assertGreaterEqual(scenario.strength, 0.0)
            self.assertLessEqual(scenario.strength, 1.0)
        self.assertNotIn("probability", decision.manifest())

    def test_2026_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "INVALID_CONTAMINATED"):
            reason_market(bullish_snapshot(), timestamp=DEVELOPMENT_CUTOFF)

    def test_risk_ablation_is_explicit_not_silent(self):
        row = bullish_snapshot()
        row["structure_15m"] = float("nan")
        row["structure_1h"] = float("nan")
        full = reason_market(row, timestamp=1_700_000_000.0)
        ablated = reason_market(row, timestamp=1_700_000_000.0, use_risk_critic=False)
        self.assertEqual(full.action, "WAIT")
        self.assertFalse(any(x.name == "risk_critic" for x in ablated.experts))


if __name__ == "__main__":
    unittest.main()
