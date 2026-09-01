import tempfile
import unittest
from dataclasses import replace

from brian2026.catalog import catalog_dataset
from brian2026.dataset import ClosedCandle, MarketDataset, MarketEvent
from brian2026.engine import BrianEngine
from brian2026.evaluation import (ChampionPolicy, LockedFold, champion_candidate,
                                  evaluate_with_replay, fit_fold)
from brian2026.features import FeatureSnapshot
from brian2026.learning import (GradientBoostingBaseline, LogisticRegressionBaseline,
                                ProbabilityPrediction, metadata_for)
from brian2026.multitimeframe import join_completed_timeframes
from brian2026.policy import PolicyThresholds, decide
from brian2026.replay import ReplayPoint, ReplaySettings
from brian2026.samples import SupervisedSample, TargetConfig, build_samples
from brian2026.splits import WalkForwardSplitter


def sample(i, label=None, missing=False):
    label = (-1, 0, 1)[i % 3] if label is None else label
    x = float(i % 3 - 1)
    return SupervisedSample(float(i), "BTC", (("price", 100 + i), ("return_5", None if missing else x),
                                                    ("rsi", 50 + x * 10)), label, x, float(i + 1), "dataset")


def model(kind="logistic"):
    meta = metadata_for(kind, "dataset", "test-code", "features", 0)
    return LogisticRegressionBaseline(meta) if kind == "logistic" else GradientBoostingBaseline(meta)


def path(i=0):
    base = 100.0 + i * 0.01
    return (ReplayPoint(i, base - .05, base + .05, base + .1, base - .1, base),
            ReplayPoint(i + 1, base + .45, base + .55, base + .7, base + .3, base + .5))


class LeakageSafetyTests(unittest.TestCase):
    def test_labels_use_future_but_features_do_not(self):
        rows = [FeatureSnapshot("BTC", i, 100 + i, "TREND", candle_timestamp=i,
                                return_5=float(i), dataset_id="d") for i in range(4)]
        built = build_samples(rows, TargetConfig(horizon=2))
        self.assertEqual(built[0].target_timestamp, 2)
        self.assertEqual(dict(built[0].features)["return_5"], 0)
        self.assertNotIn("future_return", dict(built[0].features))
        with self.assertRaises(ValueError):
            FeatureSnapshot("BTC", 1, 100, "X", source_timestamps={"future": 2})

    def test_incomplete_higher_timeframe_cannot_leak(self):
        base = [FeatureSnapshot("BTC", 100, 100, "X", timeframe="1m", candle_timestamp=100)]
        higher = [FeatureSnapshot("BTC", 101, 100, "X", timeframe="1h", candle_timestamp=99, rsi=90)]
        joined = join_completed_timeframes(base, higher)
        self.assertIn("1h", joined[0].missing_timeframes)
        self.assertNotIn("1h__rsi", dict(joined[0].values))

    def test_dataset_catalog_provenance_and_missingness(self):
        candle = ClosedCandle(0, 60, 100, 101, 99, 100, 5)
        event = MarketEvent("BTC", "1m", 60, 61, candle, sources=(("candles", "fixture"),))
        dataset = MarketDataset.from_events([event])
        snap = FeatureSnapshot("BTC", 60, 100, "X", candle_timestamp=60, dataset_id=dataset.dataset_id)
        entry = catalog_dataset(dataset, [snap])
        self.assertEqual(entry.dataset_id, dataset.dataset_id)
        self.assertEqual(dict(entry.missing_feature_counts)["rsi"], 1)
        self.assertEqual(entry.source_provenance, (("candles", "fixture"),))


class TrainingBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.rows = tuple(sample(i, missing=(i == 0)) for i in range(30))

    def test_preprocessing_fits_train_only_and_test_locked(self):
        fold = WalkForwardSplitter(12, 6, 6, purge=1, max_folds=1).split(len(self.rows))[0]
        locked = LockedFold(fold, self.rows)
        learner = model()
        learner.fit(locked.train(), partition="train")
        stats_before = learner.pipeline.named_steps["imputer"].statistics_.copy()
        learner.calibrate(locked.validation(), partition="validation")
        self.assertEqual(learner.fit_partition, "train")
        self.assertEqual(learner.calibrator.fit_partition, "validation")
        self.assertTrue((stats_before == learner.pipeline.named_steps["imputer"].statistics_).all())
        locked.test()
        with self.assertRaises(RuntimeError):
            locked.test()
        with self.assertRaises(ValueError):
            learner.fit(locked.validation(), partition="validation")

    def test_save_load_identical_and_reproducible(self):
        first, second = model(), model()
        first.fit(self.rows[:18]); second.fit(self.rows[:18])
        p1, p2 = first.predict_probability(self.rows[18:]), second.predict_probability(self.rows[18:])
        self.assertEqual(p1, p2)
        with tempfile.TemporaryDirectory() as td:
            artifact = first.save(td)
            loaded = LogisticRegressionBaseline.load(artifact)
            self.assertEqual(p1, loaded.predict_probability(self.rows[18:]))

    def test_both_real_ml_baselines_fit(self):
        for kind in ("logistic", "gradient"):
            learner = model(kind).fit(self.rows[:21])
            probs = learner.predict_probability(self.rows[21:])
            self.assertEqual(len(probs), 9)
            self.assertAlmostEqual(sum((probs[0].down, probs[0].neutral, probs[0].up)), 1.0)


class PolicyReplayAndChampionTests(unittest.TestCase):
    def test_buy_sell_wait_thresholds(self):
        cfg = PolicyThresholds(.6, .6, .1)
        self.assertEqual(decide(ProbabilityPrediction(.1, .1, .8), cfg), "BUY")
        self.assertEqual(decide(ProbabilityPrediction(.8, .1, .1), cfg), "SELL")
        self.assertEqual(decide(ProbabilityPrediction(.34, .32, .34), cfg), "WAIT")
        weak = [ProbabilityPrediction(.34, .32, .34)] * 20
        self.assertTrue(all(decide(p, cfg) == "WAIT" for p in weak))

    def test_replay_costs_reduce_model_performance(self):
        probabilities = [ProbabilityPrediction(.05, .05, .9)] * 6
        labels = [1] * 6
        paths = [path(i * 2) for i in range(6)]
        free = evaluate_with_replay("model", probabilities, labels, paths, PolicyThresholds(),
                                    ReplaySettings(position_size=1000, taker_fee_bps=0, slippage_bps=0))
        costly = evaluate_with_replay("model", probabilities, labels, paths, PolicyThresholds(),
                                      ReplaySettings(position_size=1000, taker_fee_bps=10, slippage_bps=5))
        self.assertLess(costly.trading.net_pnl, free.trading.net_pnl)
        self.assertGreater(costly.cost_burden, free.cost_burden)

    def test_champion_requires_locked_oos_evidence(self):
        probabilities = [ProbabilityPrediction(.05, .05, .9)] * 6
        result = evaluate_with_replay("challenger", probabilities, [1] * 6, [path(i * 2) for i in range(6)],
                                      PolicyThresholds(), ReplaySettings(taker_fee_bps=0), validated=True)
        rejected = champion_candidate([result], {"wait": []}, ChampionPolicy(min_folds=2))
        self.assertEqual(rejected["status"], "REJECTED")
        wait_probabilities = [ProbabilityPrediction(.05, .9, .05)] * 6
        wait = evaluate_with_replay("wait", wait_probabilities, [1] * 6, [path(i * 2) for i in range(6)],
                                    PolicyThresholds(), ReplaySettings(taker_fee_bps=0), validated=True)
        candidate = champion_candidate([result, result], {"wait": [wait, wait]}, ChampionPolicy(min_folds=2))
        self.assertEqual(candidate["status"], "CHAMPION_CANDIDATE")
        self.assertFalse(candidate["live_applied"])

    def test_hard_shadow_only_remains(self):
        with tempfile.TemporaryDirectory() as td:
            engine = BrianEngine({"shadow_only": False}, runtime_root=td)
            self.assertTrue(engine.shadow_only)
            snapshot = FeatureSnapshot("BTC", 1, 100, "X")
            self.assertTrue(engine.decide(snapshot.to_market()).shadow_only)


if __name__ == "__main__":
    unittest.main()