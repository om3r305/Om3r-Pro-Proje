import unittest

from brian2026.timing import completed_klines, prior_high


class BrianTimingTests(unittest.TestCase):
    def test_incomplete_kline_is_excluded(self):
        closed = [0, "1", "2", "0.5", "1.5", "10", 999]
        incomplete = [1000, "1.5", "3", "1", "2.5", "20", 2001]
        self.assertEqual(completed_klines([closed, incomplete], now_ms=2000), [closed])

    def test_breakout_reference_excludes_observation_candle(self):
        self.assertEqual(prior_high([10.0, 11.0, 50.0], lookback=2), 11.0)


if __name__ == "__main__":
    unittest.main()
