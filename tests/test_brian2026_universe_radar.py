from __future__ import annotations

import unittest

from brian2026.universe_radar import (
    MarketUniverseRow,
    UniverseConfig,
    build_universe_snapshot,
    compare_universe,
)


class UniverseRadarTests(unittest.TestCase):
    def row(self, symbol: str, base: str, **overrides):
        values = dict(
            symbol=symbol, base_asset=base, quote_asset="USDT", last_price=10.0,
            quote_volume=20_000_000.0, trades_24h=50_000, price_change_pct=2.0,
            high_price=11.0, low_price=9.0, bid_price=9.99, ask_price=10.01,
            spot_trading_allowed=True,
        )
        values.update(overrides)
        return MarketUniverseRow(**values)

    def test_filters_non_liquid_non_spot_and_stable_assets(self):
        rows = [
            self.row("AAAUSDT", "AAA"),
            self.row("LOWUSDT", "LOW", quote_volume=100.0),
            self.row("OFFUSDT", "OFF", spot_trading_allowed=False),
            self.row("USDCUSDT", "USDC"),
        ]
        snap = build_universe_snapshot(rows, observed_at=100.0)
        self.assertEqual(snap.eligible_symbols, ("AAAUSDT",))
        self.assertEqual(len(snap.candidates), 1)
        self.assertEqual(snap.rejected_count, 3)

    def test_missing_spread_is_neutral_not_perfect(self):
        rows = [
            self.row("AAAUSDT", "AAA", bid_price=None, ask_price=None, quote_volume=30_000_000),
            self.row("BBBUSDT", "BBB", bid_price=9.999, ask_price=10.001, quote_volume=20_000_000),
        ]
        snap = build_universe_snapshot(rows, observed_at=100.0)
        by_symbol = {row.symbol: row for row in snap.candidates}
        self.assertEqual(by_symbol["AAAUSDT"].spread_quality, 0.50)
        self.assertGreater(by_symbol["BBBUSDT"].spread_quality, 0.50)

    def test_radar_ranks_research_attention_without_action(self):
        rows = [
            self.row("AAAUSDT", "AAA", quote_volume=10_000_000, trades_24h=2_000,
                     price_change_pct=1.0, high_price=10.2, low_price=9.8),
            self.row("FASTUSDT", "FAST", quote_volume=300_000_000, trades_24h=500_000,
                     price_change_pct=25.0, high_price=14.0, low_price=8.0,
                     bid_price=9.999, ask_price=10.001),
        ]
        snap = build_universe_snapshot(rows, observed_at=100.0)
        self.assertEqual(snap.candidates[0].symbol, "FASTUSDT")
        self.assertFalse(hasattr(snap.candidates[0], "action"))
        self.assertFalse(hasattr(snap.candidates[0], "order"))

    def test_first_snapshot_is_baseline_not_fake_listing_alert(self):
        snap = build_universe_snapshot([self.row("AAAUSDT", "AAA")], observed_at=100.0)
        delta = compare_universe(None, snap)
        self.assertFalse(delta.comparable)
        self.assertEqual(delta.newly_observed_symbols, ())

    def test_only_chronological_snapshot_diff_can_mark_new_symbol(self):
        first = build_universe_snapshot([self.row("AAAUSDT", "AAA")], observed_at=100.0)
        second = build_universe_snapshot([
            self.row("AAAUSDT", "AAA"), self.row("NEWUSDT", "NEW")
        ], observed_at=200.0)
        delta = compare_universe(first, second)
        self.assertTrue(delta.comparable)
        self.assertEqual(delta.newly_observed_symbols, ("NEWUSDT",))

    def test_snapshot_comparison_rejects_time_reversal(self):
        first = build_universe_snapshot([self.row("AAAUSDT", "AAA")], observed_at=200.0)
        second = build_universe_snapshot([self.row("AAAUSDT", "AAA")], observed_at=100.0)
        with self.assertRaisesRegex(ValueError, "chronologically"):
            compare_universe(first, second)

    def test_top_n_limits_deep_research_candidates_not_eligible_memory(self):
        rows = [self.row(f"A{i}USDT", f"A{i}", quote_volume=10_000_000 + i * 1_000_000)
                for i in range(10)]
        snap = build_universe_snapshot(rows, observed_at=100.0, config=UniverseConfig(top_n=3))
        self.assertEqual(len(snap.candidates), 3)
        self.assertEqual(len(snap.eligible_symbols), 10)


if __name__ == "__main__":
    unittest.main()
