from datetime import datetime, timezone
import unittest

from brian2026.data import Instrument, RawKline, RawKlineBatch, normalize
from brian2026.phase24_experiment import FOLDS, HOLDOUT, HOLDOUT_STATUS, HORIZONS, SCHEMA, ts


class Phase24ScientificBoundaryTests(unittest.TestCase):
    def test_malformed_duration_candle_is_rejected(self):
        instrument = Instrument("binance", "spot", "BTCUSDT", "BTC", "USDT")
        values = ("0", "100", "101", "99", "100", "1", "54362", "0", "1")
        raw = RawKline(values, "official-test", 1000.0)
        batch = RawKlineBatch(instrument, "1m", 0.0, 60.0, (raw,))
        result = normalize(batch, now=1000.0)
        self.assertEqual(result.records, ())
        self.assertEqual(result.report.incomplete_rejected, 1)

    def test_locked_folds_precede_holdout_and_are_chronological(self):
        self.assertEqual(HORIZONS, (3, 6, 12))
        self.assertTrue(SCHEMA.startswith("brian.phase24-decision."))
        prior_test_end = 0.0
        for start, train_end, validation_end, test_end in FOLDS:
            boundaries = [ts(v) for v in (start, train_end, validation_end, test_end)]
            self.assertEqual(boundaries, sorted(boundaries))
            self.assertGreater(boundaries[-1], prior_test_end)
            prior_test_end = boundaries[-1]
        self.assertLessEqual(prior_test_end, ts(HOLDOUT[0]))

    def test_2026_holdout_is_permanently_invalid(self):
        self.assertEqual(HOLDOUT_STATUS["status"], "INVALID_CONTAMINATED")
        self.assertFalse(HOLDOUT_STATUS["reusable_as_pristine_holdout"])
        self.assertFalse(HOLDOUT_STATUS["evaluation_allowed"])


if __name__ == "__main__":
    unittest.main()
