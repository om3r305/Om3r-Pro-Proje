from __future__ import annotations

import unittest

from brian2026.intelligence_fabric import WhaleObservation
from brian2026.smart_money import smart_money_consensus


T0 = 1_700_000_000.0


class SmartMoneyTests(unittest.TestCase):
    def whale(self, entity: str, **overrides):
        values = dict(
            asset="ABC", observed_at=T0, entity_id=entity, label_source="provider",
            label_trust="verified_provider", label_confidence=0.95, flow="dex_buy",
            usd_value=500_000.0, tx_hash=f"0x{entity}", historical_timestamp_verified=True,
        )
        values.update(overrides)
        return WhaleObservation(**values)

    def test_single_huge_whale_does_not_become_strong_consensus(self):
        result = smart_money_consensus([
            self.whale("one", usd_value=20_000_000.0),
        ])
        self.assertEqual(result.unique_entities, 1)
        self.assertTrue(result.concentrated)
        self.assertIn("insufficient independent entity breadth", result.veto_reasons)
        self.assertIn("flow is dominated by one entity", result.veto_reasons)
        self.assertLess(result.confidence, 0.20)

    def test_three_verified_independent_dex_buys_form_breadth(self):
        result = smart_money_consensus([
            self.whale("a", usd_value=500_000),
            self.whale("b", usd_value=600_000),
            self.whale("c", usd_value=700_000),
        ])
        self.assertEqual(result.unique_entities, 3)
        self.assertEqual(result.bullish_entities, 3)
        self.assertFalse(result.concentrated)
        self.assertGreater(result.direction, 0.80)
        self.assertGreater(result.accumulation_score, 0.70)
        self.assertFalse(result.veto_reasons)

    def test_verified_sellers_produce_distribution_context(self):
        result = smart_money_consensus([
            self.whale("a", flow="dex_sell"),
            self.whale("b", flow="dex_sell"),
            self.whale("c", flow="dex_sell"),
        ])
        self.assertLess(result.direction, -0.80)
        self.assertGreater(result.distribution_score, 0.70)
        self.assertEqual(result.accumulation_score, 0.0)

    def test_internal_and_unknown_flows_do_not_create_direction(self):
        result = smart_money_consensus([
            self.whale("a", flow="internal_transfer"),
            self.whale("b", flow="unknown"),
            self.whale("c", flow="internal_transfer"),
        ])
        self.assertEqual(result.qualified_observations, 0)
        self.assertEqual(result.direction, 0.0)
        self.assertIn("no verified directional smart-money observations", result.veto_reasons)

    def test_user_generated_labels_do_not_qualify(self):
        result = smart_money_consensus([
            self.whale("a", label_trust="user_generated", label_confidence=1.0),
            self.whale("b", label_trust="user_generated", label_confidence=1.0),
            self.whale("c", label_trust="user_generated", label_confidence=1.0),
        ])
        self.assertEqual(result.qualified_observations, 0)
        self.assertEqual(result.unresolved_observations, 3)
        self.assertTrue(result.veto_reasons)

    def test_mixed_entities_reduce_directional_consensus(self):
        result = smart_money_consensus([
            self.whale("a", flow="dex_buy", usd_value=500_000),
            self.whale("b", flow="dex_buy", usd_value=500_000),
            self.whale("c", flow="dex_sell", usd_value=500_000),
        ])
        self.assertEqual(result.bullish_entities, 2)
        self.assertEqual(result.bearish_entities, 1)
        self.assertGreater(result.direction, 0)
        self.assertLess(result.direction, 0.50)

    def test_cross_asset_mix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one asset"):
            smart_money_consensus([
                self.whale("a"), self.whale("b", asset="XYZ")
            ])


if __name__ == "__main__":
    unittest.main()
