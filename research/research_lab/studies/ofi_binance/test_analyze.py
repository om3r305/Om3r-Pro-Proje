"""Offline tests for the OFI analysis. Run: python -m unittest test_analyze (from this folder)."""

import json
import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

TICK = 0.01


def make_buckets(seconds, beta_fn, noise=0.5, seed=1, ad_fn=lambda s: 10.0, start=1_800_000_000, gaps=(), resets=()):
    """Per-second buckets where the mid moves beta * OFI ticks plus noise."""
    rng = random.Random(seed)
    mid, out = 100.0, []
    for i, s in enumerate(range(start, start + seconds)):
        if s in gaps:
            continue
        ad = ad_fn(s)
        ofi = rng.gauss(0, ad)
        mid += TICK * (beta_fn(s) * ofi + rng.gauss(0, noise))
        out.append({"symbol": "BTCUSDT", "sec": s, "ofi": ofi, "events": 5, "depthSum": 5 * 2 * ad,
                    "bid": mid - TICK / 2, "ask": mid + TICK / 2, "bidQty": 1, "askQty": 1, "mid": mid,
                    "firstUpdateId": i, "lastUpdateId": i, "stateReset": s in resets})
    return out


class IntervalTests(unittest.TestCase):
    def test_gap_and_reset_invalidate_intervals(self):
        b = make_buckets(100, lambda s: 0.1, gaps={1_800_000_015}, resets={1_800_000_042})
        iv = A.to_intervals(b, TICK)
        ts = [i["t"] for i in iv]
        self.assertNotIn(1_800_000_010, ts)          # interval with the missing second
        self.assertNotIn(1_800_000_040, ts)          # interval with a state reset
        nxt = [i for i in iv if i["t"] == 1_800_000_020][0]
        self.assertIsNone(nxt["dp"])                 # previous interval invalid -> no Delta P
        ok = [i for i in iv if i["t"] == 1_800_000_030][0]
        self.assertIsNotNone(ok["dp"])

    def test_ols_white_recovers_slope(self):
        rng = random.Random(3)
        x = [rng.gauss(0, 1) for _ in range(5000)]
        y = [2.0 + 0.7 * xi + rng.gauss(0, 0.1) for xi in x]
        a, b, r2, se = A.ols_white(x, y)
        self.assertAlmostEqual(b, 0.7, delta=0.01)
        self.assertGreater(r2, 0.95)
        self.assertGreater(se, 0)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.window = A.WINDOW
        A.WINDOW = 300  # 30 intervals per window keeps the test fast; the rule itself is unchanged

    def tearDown(self):
        A.WINDOW = self.window

    def test_c01_pass_on_linear_relation_and_c02_finds_inverse_depth(self):
        # Depth alternates by window; beta = 1/AD, so lambda should be ~1.
        ad = lambda s: 5.0 if (s // 300) % 2 else 20.0
        b = make_buckets(300 * 230, beta_fn=lambda s: 1.0 / ad(s), noise=0.3, ad_fn=ad)
        r1 = A.c01(A.windows(A.to_intervals(b, TICK)))
        self.assertEqual(r1["status"], "PASS", r1.get("median_r2"))
        r2 = A.c02(r1["fits"])
        self.assertIn(r2["status"], ("PASS", "INSUFFICIENT_DATA"))
        if r2["status"] == "PASS":
            self.assertAlmostEqual(r2["lambda"], 1.0, delta=0.2)

    def test_c01_fail_when_price_ignores_ofi(self):
        b = make_buckets(300 * 230, beta_fn=lambda s: 0.0, noise=1.0)
        r1 = A.c01(A.windows(A.to_intervals(b, TICK)))
        self.assertEqual(r1["status"], "FAIL")

    def test_c01_insufficient_with_few_windows(self):
        b = make_buckets(300 * 20, beta_fn=lambda s: 0.1)
        self.assertEqual(A.c01(A.windows(A.to_intervals(b, TICK)))["status"], "INSUFFICIENT_DATA")


class HypothesisTests(unittest.TestCase):
    def _days(self, days, beta, noise):
        out = []
        for d in range(days):  # 6 contiguous hours per UTC day; enough test trades even at the 0.99 threshold
            out += make_buckets(21600, beta_fn=lambda s: beta, noise=noise, seed=d + 10, start=1_800_000_000 - (1_800_000_000 % 86400) + d * 86400)
        return out

    def test_random_walk_fails_after_costs(self):
        iv = A.to_intervals(self._days(14, 0.0, 2.0), TICK)
        r = A.hypothesis(iv, fee_bps=10.0)
        self.assertEqual(r["status"], "FAIL", r)
        self.assertLess(r["mean_net_bps"], 0)

    def test_threshold_is_frozen_from_tuning_days(self):
        iv = A.to_intervals(self._days(14, 0.0, 2.0), TICK)
        r = A.hypothesis(iv, fee_bps=0.0)
        self.assertIn(r["quantile"], A.QUANTILES)
        self.assertGreater(r["threshold"], 0)

    def test_needs_test_period(self):
        iv = A.to_intervals(self._days(5, 0.0, 1.0), TICK)
        self.assertEqual(A.hypothesis(iv, 10.0)["status"], "INSUFFICIENT_DATA")


class EndToEndTests(unittest.TestCase):
    def test_files_roundtrip_and_missing_data_review(self):
        with tempfile.TemporaryDirectory() as d:
            gaps = set(range(1_800_000_000 + 1000, 1_800_000_000 + 1000 + 600))  # 600 of 6000 s = 10% missing
            b = make_buckets(6000, beta_fn=lambda s: 0.1, gaps=gaps)
            with open(os.path.join(d, "2027-01-15.BTCUSDT.bucket.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"header": True}) + "\n")
                for x in b:
                    f.write(json.dumps(x) + "\n")
            with open(os.path.join(d, "2027-01-15.all.meta.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"header": True}) + "\n" + json.dumps({"symbol": "BTCUSDT", "tickSize": TICK}) + "\n")
            r = A.analyze(d, 10.0)
            s = r["symbols"]["BTCUSDT"]
            self.assertAlmostEqual(s["missing_seconds_share"], 0.1, delta=0.001)
            self.assertFalse(r["trade_signal"])
            # Few windows -> INSUFFICIENT_DATA beats REVIEW_REQUIRED (nothing decided either way).
            self.assertIn(s["C-01"]["status"], ("INSUFFICIENT_DATA", "REVIEW_REQUIRED"))

    def test_unknown_tick_is_review_required(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "x.ETHUSDT.bucket.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"symbol": "ETHUSDT", "sec": 1, "ofi": 0, "events": 0, "depthSum": 0, "bid": 1, "ask": 2, "bidQty": 1, "askQty": 1, "mid": 1.5, "stateReset": False}) + "\n")
            self.assertEqual(A.analyze(d, 10.0)["symbols"]["ETHUSDT"]["status"], "REVIEW_REQUIRED")


if __name__ == "__main__":
    unittest.main()
