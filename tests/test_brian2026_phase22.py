import inspect
import pathlib
import tempfile
import unittest
from dataclasses import asdict

from brian2026.catalog import catalog_dataset
from brian2026.data import (BinancePublicKlineAdapter, Instrument, RawKline, RawKlineBatch,
                            SimulationSpread, build_feature_snapshots, build_market_dataset,
                            normalize, read_raw, replay_points)
from brian2026.engine import BrianEngine
from brian2026.features import FeatureSnapshot
from brian2026.multitimeframe import join_completed_timeframes


NOW = 2_000_000_000.0
INSTRUMENT = Instrument("binance", "spot", "BTCUSDT", "BTC", "USDT")
ENDPOINT = BinancePublicKlineAdapter.endpoint


def row(open_s, *, close=None, high=None, low=None, close_s=None):
    close = 100.5 if close is None else close
    high = max(101.0, close) if high is None else high
    low = min(99.0, close) if low is None else low
    close_s = open_s + 59.999 if close_s is None else close_s
    return (str(int(open_s * 1000)), "100", str(high), str(low), str(close), "10",
            str(int(close_s * 1000)), "0", "1", "0", "0", "0")


def raw(open_s, **kwargs):
    return RawKline(tuple(row(open_s, **kwargs)), ENDPOINT, NOW + 100)


def batch(records, start=1_000.0, end=2_000.0):
    return RawKlineBatch(INSTRUMENT, "1m", start, end, tuple(records))


class Response:
    def __init__(self, payload, status=200, headers=None):
        self.payload, self.status_code, self.headers = payload, status, headers or {}
    def json(self): return self.payload
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError(self.status_code)


class Session:
    def __init__(self, pages): self.pages, self.calls = list(pages), []
    def get(self, url, params, timeout):
        self.calls.append((url, dict(params), timeout)); return Response(self.pages.pop(0))


class IngestionIntegrityTests(unittest.TestCase):
    def test_duplicate_and_gap_detection(self):
        duplicate = normalize(batch([raw(1020), raw(1020)]), now=NOW)
        self.assertEqual(duplicate.report.duplicate_count, 1)
        self.assertEqual(duplicate.report.status, "REJECTED")
        gap = normalize(batch([raw(1020), raw(1140)]), now=NOW)
        self.assertEqual(gap.report.missing_interval_count, 1)
        self.assertEqual(gap.report.largest_gap_seconds, 120)
        self.assertEqual(gap.report.status, "WARNING")
        with self.assertRaises(ValueError): build_market_dataset(batch([raw(1020), raw(1140)]), gap)

    def test_incomplete_and_future_rejected(self):
        incomplete = normalize(batch([raw(NOW - 59.999, close_s=NOW)] , start=NOW-60, end=NOW+60), now=NOW)
        self.assertEqual(incomplete.report.incomplete_rejected, 1)
        future = normalize(batch([raw(NOW + 60)], start=NOW, end=NOW+1000), now=NOW)
        self.assertEqual(future.report.future_rejected, 1)
        self.assertFalse(future.records)

    def test_invalid_ohlc_rejected(self):
        result = normalize(batch([raw(1020, high=99, low=98, close=100.5)]), now=NOW)
        self.assertEqual(result.report.invalid_ohlc_rejected, 1)
        self.assertEqual(result.report.status, "REJECTED")

    def test_hashes_are_content_addressed_and_raw_is_immutable(self):
        first = batch([raw(1020), raw(1080)])
        same = RawKlineBatch(INSTRUMENT, "1m", 1000, 2000,
                             tuple(RawKline(r.values, r.source_endpoint, NOW + 999) for r in first.records))
        changed = batch([raw(1020), raw(1080, close=100.6)])
        before = asdict(first)
        self.assertEqual(first.raw_hash, same.raw_hash)
        self.assertNotEqual(first.raw_hash, changed.raw_hash)
        normalized = normalize(first, now=NOW)
        self.assertEqual(before, asdict(first))
        dataset1, _ = build_market_dataset(first, normalized)
        dataset2, _ = build_market_dataset(same, normalize(same, now=NOW))
        self.assertEqual(dataset1.dataset_id, dataset2.dataset_id)

    def test_raw_write_read_roundtrip_is_immutable(self):
        original = batch([raw(1020), raw(1080)])
        with tempfile.TemporaryDirectory() as td:
            path = original.write(td)
            loaded = read_raw(path)
            self.assertEqual(original.raw_hash, loaded.raw_hash)
            self.assertEqual(path, loaded.write(td))
            manifests = list((pathlib.Path(td) / "imports" / original.raw_hash).glob("*.json"))
            self.assertTrue(manifests)
    def test_pagination_order_and_explicit_range(self):
        page1 = [row(1020), row(1080)]
        page2 = [row(1140)]
        session = Session([page1, page2])
        adapter = BinancePublicKlineAdapter(session=session, retries=0)
        result = adapter.fetch(INSTRUMENT, "1m", 1020, 1201, now=NOW, limit=2)
        self.assertEqual(len(result.records), 3)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[1][1]["startTime"], 1140000)


class AvailabilityAndProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.raw_batch = batch([raw(1020), raw(1080), raw(1140)])
        self.normalized = normalize(self.raw_batch, now=NOW)
        self.dataset, self.report = build_market_dataset(self.raw_batch, self.normalized)

    def test_unavailable_market_facts_remain_unavailable(self):
        record = self.normalized.records[0]
        self.assertIsNone(record.bid); self.assertIsNone(record.ask)
        self.assertFalse(record.order_book_available); self.assertIsNone(record.funding_rate)
        event = self.dataset.events[0]
        self.assertIsNone(event.bid); self.assertIsNone(event.ask)
        snapshots = build_feature_snapshots(self.dataset)
        self.assertTrue(all(s.spread_bps is None and s.book_imbalance is None for s in snapshots))

    def test_simulated_spread_is_not_observed_truth(self):
        assumption = SimulationSpread(10)
        points = replay_points(self.normalized.records, assumption)
        self.assertEqual(assumption.provenance, "simulation_assumption")
        self.assertGreater(points[0].ask, points[0].bid)
        self.assertIsNone(self.normalized.records[0].bid)

    def test_catalog_records_instrument_and_quality_provenance(self):
        snapshots = build_feature_snapshots(self.dataset)
        entry = catalog_dataset(self.dataset, snapshots)
        self.assertEqual(entry.exchange, "binance")
        self.assertEqual(entry.market_types, ("spot",))
        self.assertEqual(entry.quality_status, "READY")
        self.assertIn(("candles", ENDPOINT), entry.source_provenance)

    def test_higher_timeframe_not_available_before_close_or_observation(self):
        base = [FeatureSnapshot("BTCUSDT", 100, 100, "UNKNOWN", timeframe="1m", candle_timestamp=100)]
        not_closed = [FeatureSnapshot("BTCUSDT", 101, 100, "UNKNOWN", timeframe="15m", candle_timestamp=101, rsi=70)]
        delayed = [FeatureSnapshot("BTCUSDT", 105, 100, "UNKNOWN", timeframe="1h", candle_timestamp=99, rsi=80)]
        joined = join_completed_timeframes(base, not_closed + delayed)[0]
        self.assertIn("15m", joined.missing_timeframes)
        self.assertIn("1h", joined.missing_timeframes)


class AbsoluteSafetyTests(unittest.TestCase):
    def test_shadow_only_and_no_execution_methods(self):
        with tempfile.TemporaryDirectory() as td:
            engine = BrianEngine({"shadow_only": False}, runtime_root=td)
            self.assertTrue(engine.shadow_only)
        source = inspect.getsource(BinancePublicKlineAdapter)
        for forbidden in ("create_order", "place_order", "cancel_order", "api_key", "secret"):
            self.assertNotIn(forbidden, source.lower())


if __name__ == "__main__": unittest.main()