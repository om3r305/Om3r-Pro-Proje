from __future__ import annotations

import unittest

from brian2026.entity_graph import EntityGraph, EntityLabel, TransferEdge


T0 = 1_700_000_000.0


class EntityGraphTests(unittest.TestCase):
    def label(self, **overrides):
        values = dict(
            address="0xabc", entity_id="fund-a", role="fund", provider="arkham",
            trust_class="verified_provider", confidence=0.95, observed_at=T0,
            label="Fund A", historical_timestamp_verified=True,
        )
        values.update(overrides)
        return EntityLabel(**values)

    def test_user_label_cannot_override_verified_provider_label(self):
        graph = EntityGraph([
            self.label(),
            self.label(entity_id="random-user-guess", role="whale", provider="user",
                       trust_class="user_generated", confidence=1.0, observed_at=T0 + 1),
        ])
        resolved = graph.resolve("0xabc", as_of=T0 + 10)
        self.assertEqual(resolved.entity_id, "fund-a")
        self.assertEqual(resolved.provider, "arkham")
        self.assertFalse(resolved.conflicting_labels)

    def test_future_label_does_not_rewrite_past(self):
        graph = EntityGraph([
            self.label(observed_at=T0 + 100),
        ])
        past = graph.resolve("0xabc", as_of=T0 + 50)
        future = graph.resolve("0xabc", as_of=T0 + 150)
        self.assertIsNone(past.entity_id)
        self.assertEqual(future.entity_id, "fund-a")

    def test_conflicting_high_authority_labels_resolve_to_unknown(self):
        graph = EntityGraph([
            self.label(provider="arkham", entity_id="fund-a", confidence=0.95),
            self.label(provider="nansen", entity_id="fund-b", confidence=0.93),
        ])
        resolved = graph.resolve("0xabc", as_of=T0 + 1)
        self.assertTrue(resolved.conflicting_labels)
        self.assertIsNone(resolved.entity_id)
        self.assertEqual(resolved.role, "unknown")

    def test_same_entity_transfer_is_internal(self):
        graph = EntityGraph([
            self.label(address="0x1", entity_id="fund-a"),
            self.label(address="0x2", entity_id="fund-a"),
        ])
        edge = TransferEdge("0xtx", "ABC", "0x1", "0x2", 2_000_000, T0 + 1, "ethereum")
        result = graph.interpret_transfer(edge)
        self.assertEqual(result.flow, "internal_transfer")
        self.assertFalse(result.economically_directional)

    def test_transfer_to_exchange_is_deposit_not_automatic_sell(self):
        graph = EntityGraph([
            self.label(address="0xwallet", entity_id="fund-a", role="fund"),
            self.label(address="0xexchange", entity_id="binance", role="exchange",
                       provider="provider", confidence=0.99),
        ])
        edge = TransferEdge("0xtx2", "ABC", "0xwallet", "0xexchange", 3_000_000, T0 + 1, "ethereum")
        result = graph.interpret_transfer(edge)
        self.assertEqual(result.flow, "exchange_deposit")
        self.assertTrue(result.economically_directional)
        self.assertNotEqual(result.flow, "dex_sell")

    def test_exchange_withdrawal_is_classified(self):
        graph = EntityGraph([
            self.label(address="0xexchange", entity_id="binance", role="exchange", confidence=0.99),
            self.label(address="0xwallet", entity_id="fund-a", role="fund"),
        ])
        edge = TransferEdge("0xtx3", "ABC", "0xexchange", "0xwallet", 1_000_000, T0 + 1, "ethereum")
        result = graph.interpret_transfer(edge)
        self.assertEqual(result.flow, "exchange_withdrawal")

    def test_dex_transfer_without_swap_semantics_stays_unknown(self):
        graph = EntityGraph([
            self.label(address="0xwallet", entity_id="fund-a", role="fund"),
            self.label(address="0xdex", entity_id="uniswap", role="dex", provider="provider", confidence=0.99),
        ])
        edge = TransferEdge("0xtx4", "ABC", "0xwallet", "0xdex", 500_000, T0 + 1, "ethereum")
        result = graph.interpret_transfer(edge)
        self.assertEqual(result.flow, "unknown")
        self.assertFalse(result.economically_directional)
        self.assertIn("swap-leg semantics", result.reasons[0])

    def test_historical_replay_rejects_unverified_label_availability(self):
        graph = EntityGraph([
            self.label(historical_timestamp_verified=False),
        ])
        with self.assertRaisesRegex(ValueError, "point-in-time availability"):
            graph.assert_historical_labels_safe(as_of=T0 + 1)


if __name__ == "__main__":
    unittest.main()
