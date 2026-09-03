from __future__ import annotations

import tempfile
import unittest

from brian2026.intelligence_store import IntelligenceStore
from brian2026.prospective_collectors import BinancePublicUniverseCollector
from brian2026.universe_radar import UniverseConfig


class ProspectiveCollectorTests(unittest.TestCase):
    def payloads(self):
        exchange = {
            "symbols": [
                {"symbol": "AAAUSDT", "baseAsset": "AAA", "quoteAsset": "USDT",
                 "status": "TRADING", "isSpotTradingAllowed": True},
                {"symbol": "BBBUSDT", "baseAsset": "BBB", "quoteAsset": "USDT",
                 "status": "TRADING", "isSpotTradingAllowed": True},
            ]
        }
        tickers = [
            {"symbol": "AAAUSDT", "lastPrice": "10", "quoteVolume": "20000000",
             "count": 50000, "priceChangePercent": "3", "highPrice": "11", "lowPrice": "9"},
            {"symbol": "BBBUSDT", "lastPrice": "5", "quoteVolume": "30000000",
             "count": 60000, "priceChangePercent": "5", "highPrice": "6", "lowPrice": "4"},
        ]
        books = [
            {"symbol": "AAAUSDT", "bidPrice": "9.99", "askPrice": "10.01"},
            {"symbol": "BBBUSDT", "bidPrice": "4.99", "askPrice": "5.01"},
        ]
        return exchange, tickers, books

    def getter(self, fail_book=False):
        exchange, tickers, books = self.payloads()
        def get(url, timeout):
            self.assertGreater(timeout, 0)
            if url.endswith("exchangeInfo"):
                return exchange
            if url.endswith("ticker/24hr"):
                return tickers
            if url.endswith("ticker/bookTicker"):
                if fail_book:
                    raise TimeoutError("optional book source unavailable")
                return books
            raise AssertionError(url)
        return get

    def clock(self):
        values = iter([100.0, 101.0, 102.0, 103.0, 104.0])
        return lambda: next(values)

    def test_collects_public_universe_without_secret_arguments(self):
        collector = BinancePublicUniverseCollector(
            getter=self.getter(), clock=self.clock(),
            config=UniverseConfig(min_quote_volume=1_000_000, min_trades_24h=100),
        )
        cycle = collector.collect()
        self.assertEqual(cycle.snapshot.eligible_symbols, ("AAAUSDT", "BBBUSDT"))
        self.assertFalse(cycle.delta.comparable)
        self.assertTrue(cycle.shadow_only)
        self.assertFalse(hasattr(collector, "api_key"))
        self.assertFalse(hasattr(collector, "secret"))

    def test_optional_book_failure_degrades_instead_of_fabricating_spread(self):
        collector = BinancePublicUniverseCollector(
            getter=self.getter(fail_book=True), clock=self.clock(),
            config=UniverseConfig(min_quote_volume=1_000_000, min_trades_24h=100),
        )
        cycle = collector.collect()
        self.assertEqual(cycle.degraded_sources, ("book_ticker",))
        self.assertTrue(all(row.spread_bps is None for row in cycle.snapshot.candidates))
        self.assertTrue(all(row.spread_quality == 0.50 for row in cycle.snapshot.candidates))

    def test_raw_public_responses_are_captured_before_ranking_artifact_use(self):
        with tempfile.TemporaryDirectory() as directory:
            store = IntelligenceStore(directory)
            collector = BinancePublicUniverseCollector(
                store=store, getter=self.getter(), clock=self.clock(),
                config=UniverseConfig(min_quote_volume=1_000_000, min_trades_24h=100),
            )
            cycle = collector.collect()
            self.assertEqual(len(cycle.capture_ids), 3)
            for capture_id in cycle.capture_ids:
                self.assertTrue(capture_id)

    def test_previous_snapshot_enables_newly_observed_detection(self):
        first_payloads = self.payloads()
        exchange, tickers, books = first_payloads
        calls = {"round": 0}
        def get(url, timeout):
            if calls["round"] == 0:
                if url.endswith("exchangeInfo"): return {"symbols": exchange["symbols"][:1]}
                if url.endswith("ticker/24hr"): return tickers[:1]
                if url.endswith("ticker/bookTicker"):
                    calls["round"] = 1
                    return books[:1]
            if url.endswith("exchangeInfo"): return exchange
            if url.endswith("ticker/24hr"): return tickers
            if url.endswith("ticker/bookTicker"): return books
            raise AssertionError(url)
        clock_values = iter([100.0, 200.0])
        collector = BinancePublicUniverseCollector(
            getter=get, clock=lambda: next(clock_values),
            config=UniverseConfig(min_quote_volume=1_000_000, min_trades_24h=100),
        )
        first = collector.collect()
        second = collector.collect(first.snapshot)
        self.assertTrue(second.delta.comparable)
        self.assertEqual(second.delta.newly_observed_symbols, ("BBBUSDT",))

    def test_invalid_required_exchange_response_fails_closed(self):
        def get(url, timeout):
            if url.endswith("exchangeInfo"): return []
            if url.endswith("ticker/24hr"): return []
            return []
        collector = BinancePublicUniverseCollector(getter=get, clock=lambda: 100.0)
        with self.assertRaisesRegex(ValueError, "exchangeInfo"):
            collector.collect()


if __name__ == "__main__":
    unittest.main()
