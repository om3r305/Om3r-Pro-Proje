from dataclasses import replace
import unittest

import numpy as np

from brian2026.adaptive_quant import (
    CausalDriftMonitor,
    ChallengerVote,
    DriftAssessment,
    DriftConfig,
    FamiliarityAssessment,
    FamiliarityConfig,
    FAMILIARITY_FEATURES,
    LeagueConfig,
    MarketFamiliarityModel,
    combine_challengers,
    validation_weight,
)
from brian2026.engine import BrianEngine
from brian2026.portfolio import DEVELOPMENT_CUTOFF


def feature_map(rows=80, *, shift_from=None, shift=0.0):
    out = {}
    for position, name in enumerate(FAMILIARITY_FEATURES):
        values = np.linspace(-0.2, 0.2, rows) + position * 0.01
        if shift_from is not None:
            values = values.copy()
            values[shift_from:] += shift
        out[name] = values
    return out


def snapshot(features, index):
    return {name: float(values[index]) for name, values in features.items()}


class Phase29FamiliarityTests(unittest.TestCase):
    def setUp(self):
        self.timestamps = np.asarray([1_700_000_000 + i * 300 for i in range(80)], dtype=float)

    def test_familiarity_fits_train_only_and_rejects_extreme_ood(self):
        features = feature_map()
        model = MarketFamiliarityModel(config=FamiliarityConfig(quantile=0.95)).fit(
            features, np.arange(40), self.timestamps
        )
        known = model.assess_snapshot(snapshot(features, 20))
        extreme = snapshot(features, 20)
        extreme = {key: value + 100.0 for key, value in extreme.items()}
        unknown = model.assess_snapshot(extreme)
        self.assertFalse(known.out_of_distribution)
        self.assertTrue(unknown.out_of_distribution)
        self.assertLess(unknown.familiarity, known.familiarity)
        self.assertEqual(model.manifest()["fit_partition"], "train_only")

    def test_future_edits_do_not_change_fitted_familiarity_reference(self):
        original = feature_map()
        edited = feature_map(shift_from=40, shift=999.0)
        left = MarketFamiliarityModel().fit(original, np.arange(40), self.timestamps)
        right = MarketFamiliarityModel().fit(edited, np.arange(40), self.timestamps)
        self.assertEqual(left.manifest(), right.manifest())

    def test_missing_current_features_are_unavailable_not_zero(self):
        features = feature_map()
        model = MarketFamiliarityModel().fit(features, np.arange(40), self.timestamps)
        current = snapshot(features, 20)
        for name in FAMILIARITY_FEATURES[:8]:
            current[name] = float("nan")
        assessment = model.assess_snapshot(current)
        self.assertTrue(assessment.out_of_distribution)
        self.assertEqual(assessment.familiarity, 0.0)

    def test_2026_train_reference_is_forbidden(self):
        features = feature_map()
        timestamps = self.timestamps.copy()
        timestamps[10] = DEVELOPMENT_CUTOFF
        with self.assertRaisesRegex(ValueError, "INVALID_CONTAMINATED"):
            MarketFamiliarityModel().fit(features, np.arange(20), timestamps)


class Phase29DriftAndLeagueTests(unittest.TestCase):
    def setUp(self):
        self.timestamps = np.asarray([1_700_000_000 + i * 300 for i in range(120)], dtype=float)
        self.features = feature_map(120)
        self.model = MarketFamiliarityModel(config=FamiliarityConfig(quantile=0.95)).fit(
            self.features, np.arange(50), self.timestamps
        )

    def test_drift_threshold_is_validation_only_and_shift_is_detected_causally(self):
        cfg = DriftConfig(window=12, warmup=4, validation_quantile=0.95, hard_multiplier=1.2)
        monitor = CausalDriftMonitor(self.model, cfg)
        validation = [snapshot(self.features, i) for i in range(50, 65)]
        monitor.calibrate_validation(validation)
        threshold = monitor.threshold
        self.assertIsNotNone(threshold)
        shifted = {key: value + 50.0 for key, value in snapshot(self.features, 65).items()}
        assessments = [monitor.assess(shifted) for _ in range(12)]
        self.assertEqual(monitor.threshold, threshold)
        self.assertTrue(any(row.drifted for row in assessments))
        self.assertTrue(any(row.hard_drift for row in assessments))

    def test_future_shift_cannot_change_first_test_drift_assessment(self):
        cfg = DriftConfig(window=10, warmup=4, validation_quantile=0.95)
        validation = [snapshot(self.features, i) for i in range(50, 65)]
        left = CausalDriftMonitor(self.model, cfg)
        right = CausalDriftMonitor(self.model, cfg)
        left.calibrate_validation(validation)
        right.calibrate_validation(validation)
        first = snapshot(self.features, 65)
        self.assertEqual(left.assess(first), right.assess(first))
        # A later future observation can only affect later calls.
        future = {key: value + 500.0 for key, value in snapshot(self.features, 66).items()}
        right.assess(future)

    def _votes(self):
        return (
            ChallengerVote("expert", "BUY", 0.8, 0.8, 0.6, "validation qualified"),
            ChallengerVote("logistic", "BUY", 0.7, 0.75, 0.3, "validation qualified"),
            ChallengerVote("gb", "SELL", -0.2, 0.6, 0.1, "minority"),
        )

    def test_ood_gate_forces_wait_even_with_bullish_challengers(self):
        familiarity = FamiliarityAssessment(9.0, 1.0, 0.1, 1.0, True)
        drift = DriftAssessment(0.1, 1.0, False, False, 96)
        decision = combine_challengers(1_700_000_000, self._votes(), familiarity, drift)
        self.assertEqual(decision.action, "WAIT")
        self.assertTrue(any("out-of-distribution" in reason for reason in decision.veto_reasons))

    def test_hard_drift_forces_wait(self):
        familiarity = FamiliarityAssessment(0.2, 1.0, 1.0, 1.0, False)
        drift = DriftAssessment(2.0, 1.0, True, True, 96)
        decision = combine_challengers(1_700_000_000, self._votes(), familiarity, drift)
        self.assertEqual(decision.action, "WAIT")
        self.assertTrue(any("hard distribution shift" in reason for reason in decision.veto_reasons))

    def test_validation_weight_is_deterministic(self):
        a = validation_weight(directional_accuracy=0.60, coverage=0.20, directional_samples=100)
        b = validation_weight(directional_accuracy=0.60, coverage=0.20, directional_samples=100)
        self.assertEqual(a, b)
        self.assertGreater(a, validation_weight(directional_accuracy=0.40, coverage=0.20, directional_samples=100))

    def test_cutoff_and_shadow_only_remain_hard(self):
        familiarity = FamiliarityAssessment(0.1, 1.0, 1.0, 1.0, False)
        drift = DriftAssessment(0.1, 1.0, False, False, 96)
        with self.assertRaisesRegex(ValueError, "INVALID_CONTAMINATED"):
            combine_challengers(DEVELOPMENT_CUTOFF, self._votes(), familiarity, drift)
        engine = BrianEngine({"shadow_only": False})
        self.assertTrue(engine.shadow_only)
        self.assertFalse({"create_order", "execute_order", "place_order"}.intersection(dir(engine)))


if __name__ == "__main__":
    unittest.main()
