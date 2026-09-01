from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import unittest

from brian2026.counterfactual import compare, standard_variants
from brian2026.dataset import ClosedCandle, MarketDataset, MarketEvent, from_legacy_closed_candles
from brian2026.engine import BrianEngine
from brian2026.equity import ShadowEquityTracker
from brian2026.experiments import ExperimentManifest
from brian2026.features import FeatureSnapshot
from brian2026.metrics import calculate
from brian2026.replay import ReplayPoint, ReplaySettings, replay
from brian2026.splits import WalkForwardSplitter
from brian2026.types import MarketSnapshot


def point(ts, bid, ask, high, low, close, liquidity=1000.0):
    return ReplayPoint(ts, bid, ask, high, low, close, liquidity, liquidity)


class DatasetTests(unittest.TestCase):
    def event(self, close=101.0):
        candle = ClosedCandle(0.0, 60.0, 100.0, 102.0, 99.0, close, 10.0)
        return MarketEvent(
            "BTCUSDT", "1m", 60.0, 61.0, candle, 100.9, 101.1,
            "TREND", 42, 100, (("candles", "binance"), ("book", "binance")),
        )

    def test_dataset_is_immutable_and_hash_changes_with_data(self):
        a = MarketDataset.from_events([self.event(101.0)])
        b = MarketDataset.from_events([self.event(101.1)])
        self.assertNotEqual(a.dataset_id, b.dataset_id)
        with self.assertRaises(FrozenInstanceError):
            a.events[0].symbol = "ETHUSDT"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / f"{a.dataset_id}.json"
            a.write(path)
            self.assertEqual(a.write(path), path)
            with self.assertRaises(FileExistsError):
                b.write(path)

    def test_incomplete_candle_is_rejected(self):
        candle = ClosedCandle(0.0, 61.0, 100.0, 101.0, 99.0, 100.0, 1.0)
        with self.assertRaises(ValueError):
            MarketEvent("X", "1m", 60.0, 62.0, candle)

    def test_future_feature_sources_are_rejected(self):
        with self.assertRaises(ValueError):
            FeatureSnapshot("X", 100.0, 10.0, "TREND", candle_timestamp=101.0)
        with self.assertRaises(ValueError):
            FeatureSnapshot("X", 100.0, 10.0, "TREND", source_timestamps={"book": 100.1})

    def test_legacy_window_produces_dataset_and_feature_provenance(self):
        rows = [(0, 100, 101, 99, 100.5, 10), (60_000, 100.5, 102, 100, 101, 12)]
        dataset = from_legacy_closed_candles(
            symbol="X", timeframe="1m", candles=rows, ingestion_timestamp=121.0,
            regime="TREND", bid=100.9, ask=101.1,
        )
        snapshot = FeatureSnapshot(
            "X", 121.0, 101.0, "TREND", candle_timestamp=120.0,
            source_timestamps={"closed_candle": 120.0, "book": 121.0},
            dataset_id=dataset.dataset_id,
        )
        context = snapshot.to_market().context
        self.assertEqual(context["dataset_id"], dataset.dataset_id)
        self.assertEqual(context["feature_schema_version"], "brian.features.v2")


class ReplayTests(unittest.TestCase):
    def test_replay_is_deterministic(self):
        path = [point(0, 99.9, 100.1, 100.2, 99.8, 100.0),
                point(60, 101.1, 101.3, 101.5, 100.4, 101.2)]
        cfg = ReplaySettings(position_size=1000, tp_pct=1.0, sl_pct=1.0)
        self.assertEqual(replay(path, "LONG", cfg), replay(path, "LONG", cfg))

    def test_long_short_wait(self):
        up = [point(0, 99.9, 100.0, 100.1, 99.8, 100.0),
              point(60, 101.2, 101.3, 101.5, 100.2, 101.2)]
        down = [point(0, 99.9, 100.0, 100.1, 99.8, 100.0),
                point(60, 98.7, 98.8, 99.7, 98.5, 98.8)]
        cfg = ReplaySettings(position_size=1000, tp_pct=1.0, sl_pct=1.0,
                             taker_fee_bps=0)
        self.assertGreater(replay(up, "LONG", cfg).net_pnl, 0)
        self.assertGreater(replay(down, "SHORT", cfg).net_pnl, 0)
        wait = replay(up, "WAIT", cfg)
        self.assertEqual(wait.status, "WAIT")
        self.assertEqual(wait.net_pnl, 0.0)

    def test_fees_and_slippage_reduce_results(self):
        path = [point(0, 99.9, 100.0, 100.1, 99.8, 100.0),
                point(60, 101.2, 101.3, 101.5, 100.2, 101.2)]
        clean = replay(path, "LONG", ReplaySettings(position_size=1000, tp_pct=1, sl_pct=1,
                                                     taker_fee_bps=0, slippage_bps=0))
        costly = replay(path, "LONG", ReplaySettings(position_size=1000, tp_pct=1, sl_pct=1,
                                                      taker_fee_bps=10, slippage_bps=5))
        self.assertGreater(clean.net_pnl, costly.net_pnl)
        self.assertGreater(costly.fees, 0)

    def test_counterfactuals_share_path_and_report_metrics(self):
        path = (point(0, 99.9, 100.0, 100.1, 99.8, 100.0),
                point(60, 101.2, 101.3, 101.5, 100.2, 101.2))
        results = compare(path, standard_variants("LONG", "SHORT", ReplaySettings(position_size=1000)))
        self.assertEqual(set(results), {"actual_legacy", "brian", "wait", "delay_1s",
                                        "tight_tp_sl", "wide_tp_sl", "size_half", "size_larger"})
        self.assertEqual(results["wait"].net_pnl, 0.0)


class AccountingAndExperimentTests(unittest.TestCase):
    def test_drawdown_accounting(self):
        tracker = ShadowEquityTracker(1000.0)
        tracker.open_position("X:pred", "X", "LONG", 100.0, 2.0, 1.0)
        tracker.mark_price("X", 110.0, 2.0)
        self.assertEqual(tracker.equity, 1020.0)
        tracker.mark_price("X", 90.0, 3.0)
        state = tracker.account_state()
        self.assertEqual(state["equity"], 980.0)
        self.assertEqual(state["drawdown"], 40.0)
        self.assertEqual(state["max_drawdown"], 40.0)
        tracker.close_position("X:pred", -20.0, fees=2.0, timestamp=4.0)
        self.assertEqual(tracker.account_state()["equity"], 978.0)

    def test_walk_forward_purge_and_embargo(self):
        folds = WalkForwardSplitter(10, 4, 4, purge=2, embargo=1, max_folds=2).split(40)
        self.assertEqual(len(folds), 2)
        for fold in folds:
            fold.validate()
            self.assertGreaterEqual(fold.validation.start - fold.train.stop, 2)
            self.assertGreaterEqual(fold.test.start - fold.validation.stop, 2)
        self.assertGreaterEqual(folds[1].train.stop - folds[0].test.stop, 1)

    def test_metrics_and_manifest(self):
        metrics = calculate([10.0, -5.0], starting_equity=1000.0,
                            exposure=120.0, decisions=4, waits=2)
        self.assertEqual(metrics.average_win, 10.0)
        self.assertEqual(metrics.average_loss, -5.0)
        self.assertEqual(metrics.abstention_rate, 0.5)
        manifest = ExperimentManifest(
            "abc", "commit:test", {"model": "none"}, ("BTCUSDT",), ("1m",),
            0.0, 100.0, {"fee_bps": 10.0}, {"latency_ms": 0}, metrics.to_dict(),
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / f"{manifest.experiment_id}.json"
            manifest.write(path)
            self.assertTrue(path.exists())

    def test_brian_stays_hard_shadow_only(self):
        with tempfile.TemporaryDirectory() as td:
            engine = BrianEngine({"shadow_only": False, "starting_equity": 1000}, runtime_root=td)
            decision = engine.decide(MarketSnapshot("X", 100.0))
            self.assertTrue(decision.shadow_only)
            self.assertFalse(any(hasattr(engine, name) for name in ("execute", "place_order", "submit_order")))


if __name__ == "__main__":
    unittest.main()
