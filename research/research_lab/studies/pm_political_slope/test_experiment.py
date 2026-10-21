"""Offline tests (no network) for the political-slope experiment. Run: python -m unittest test_experiment"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify import classify_polymarket_domain  # noqa: E402
from run import ALLOWED_HOSTS, FetchFailed, Http, binary_resolution, fetch_failures_too_high, month_chunks, week_chunks, parse_ts, price_at_horizon, reference_time  # noqa: E402
from stats import cluster_bootstrap_slope, decide, fit_slope, logit  # noqa: E402


def synthetic(n, true_slope, seed=1, intercept=0.0):
    rng = random.Random(seed)
    ps, ys = [], []
    for _ in range(n):
        p = rng.uniform(0.03, 0.97)
        q = 1 / (1 + math.exp(-(intercept + true_slope * logit(p))))
        ps.append(p)
        ys.append(1 if rng.random() < q else 0)
    return ps, ys


class SlopeTests(unittest.TestCase):
    def test_recovers_known_slopes(self):
        for true in (0.6, 1.0, 1.5):
            ps, ys = synthetic(20000, true, seed=int(true * 10))
            b, a = fit_slope(ps, ys)
            self.assertAlmostEqual(b, true, delta=0.06, msg=f"true {true} got {b}")
            self.assertAlmostEqual(a, 0.0, delta=0.06)

    def test_degenerate_inputs(self):
        self.assertIsNone(fit_slope([0.5, 0.6], [1, 1]))
        self.assertIsNone(fit_slope([], []))
        self.assertIsNone(fit_slope([0.5], [1, 0]))

    def test_price_clip_matches_paper(self):
        self.assertAlmostEqual(logit(0.0), logit(0.01))
        self.assertAlmostEqual(logit(1.0), logit(0.99))


class BootstrapAndDecisionTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_order_independent(self):
        ps, ys = synthetic(600, 1.4, seed=7)
        rows = [(f"e{i // 3}", p, y) for i, (p, y) in enumerate(zip(ps, ys))]
        a = cluster_bootstrap_slope(rows, resamples=200)
        b = cluster_bootstrap_slope(list(reversed(rows)), resamples=200)
        self.assertEqual(a["lower"], b["lower"])
        self.assertEqual(a["upper"], b["upper"])
        self.assertLess(a["lower"], a["upper"])

    def test_clustering_widens_interval_for_duplicated_events(self):
        ps, ys = synthetic(300, 1.2, seed=3)
        independent = [(f"e{i}", p, y) for i, (p, y) in enumerate(zip(ps, ys))]
        # Same rows repeated 5x inside one event each: clustered CI must not shrink like n*5.
        clustered = [(f"e{i}", p, y) for i, (p, y) in enumerate(zip(ps, ys)) for _ in range(5)]
        ci_i = cluster_bootstrap_slope(independent, resamples=300)
        ci_c = cluster_bootstrap_slope(clustered, resamples=300)
        width_i = ci_i["upper"] - ci_i["lower"]
        width_c = ci_c["upper"] - ci_c["lower"]
        self.assertGreater(width_c, 0.7 * width_i)

    def test_preregistered_decision_rule(self):
        self.assertEqual(decide(99, 50, 1.2, 1.5)[0], "INSUFFICIENT_DATA")
        self.assertEqual(decide(500, 29, 1.2, 1.5)[0], "INSUFFICIENT_DATA")
        self.assertEqual(decide(500, 50, None, None)[0], "INSUFFICIENT_DATA")
        self.assertEqual(decide(500, 50, 1.05, 1.6)[0], "PASS")
        self.assertEqual(decide(500, 50, 0.7, 0.95)[0], "FAIL")
        self.assertEqual(decide(500, 50, 0.9, 1.3)[0], "UNVERIFIED")


class DataHandlingTests(unittest.TestCase):
    def test_no_lookahead_and_staleness(self):
        hist = [{"t": 1000, "p": 0.2}, {"t": 5000, "p": 0.4}, {"t": 9000, "p": 0.9}]
        self.assertEqual(price_at_horizon(hist, 8000), 0.4)          # never the later 0.9
        self.assertEqual(price_at_horizon(hist, 9000), 0.9)          # exactly at target is allowed
        self.assertIsNone(price_at_horizon(hist, 9000 + 6 * 3600 + 1))  # newest point older than 6h -> stale
        self.assertIsNone(price_at_horizon([{"t": 1, "p": 1.4}], 2))  # invalid price ignored

    def test_reference_time_is_the_earlier_of_end_and_close(self):
        m = {"endDate": "2026-03-08T06:00:00Z", "closedTime": "2026-03-10 03:28:53+00"}
        self.assertEqual(reference_time(m), parse_ts("2026-03-08T06:00:00Z"))
        self.assertEqual(reference_time({"closedTime": "2026-03-10 03:28:53+00"}), parse_ts("2026-03-10T03:28:53Z"))
        self.assertIsNone(reference_time({}))

    def test_only_clean_binary_resolutions(self):
        base = {"outcomes": '["Yes", "No"]', "umaResolutionStatus": "resolved"}
        self.assertEqual(binary_resolution({**base, "outcomePrices": '["1", "0"]'}), 1)
        self.assertEqual(binary_resolution({**base, "outcomePrices": '["0", "1"]'}), 0)
        self.assertIsNone(binary_resolution({**base, "outcomePrices": '["0.5", "0.5"]'}))
        self.assertIsNone(binary_resolution({**base, "umaResolutionStatus": "disputed", "outcomePrices": '["1", "0"]'}))
        self.assertIsNone(binary_resolution({"outcomes": '["Up", "Down"]', "umaResolutionStatus": "resolved", "outcomePrices": '["1", "0"]'}))

    def test_classification_first_match_like_paper(self):
        self.assertEqual(classify_polymarket_domain("Will PH win the most seats in the 2026 Colombian Senate election?"), "Politics")
        self.assertEqual(classify_polymarket_domain("Will the Lakers win the election of MVP?"), "Sports")  # Sports matched first
        self.assertEqual(classify_polymarket_domain("Will Bitcoin hit $150k?"), "Crypto")
        self.assertEqual(classify_polymarket_domain(""), "Other")

    def test_month_chunks_cover_window_without_gaps(self):
        ch = month_chunks("2026-01-01T00:00:00Z", "2026-10-01T00:00:00Z")
        self.assertEqual(len(ch), 9)
        self.assertEqual(ch[0], ("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"))
        self.assertEqual(ch[-1], ("2026-09-01T00:00:00Z", "2026-10-01T00:00:00Z"))
        for (a, b), (c, d) in zip(ch, ch[1:]):
            self.assertEqual(b, c)
        self.assertEqual(month_chunks("2026-11-15T00:00:00Z", "2027-01-10T00:00:00Z")[-1], ("2027-01-01T00:00:00Z", "2027-01-10T00:00:00Z"))

    def test_week_chunks_cover_window_without_gaps(self):
        ch = week_chunks("2026-01-01T00:00:00Z", "2026-10-01T00:00:00Z")
        self.assertEqual(ch[0], ("2026-01-01T00:00:00Z", "2026-01-08T00:00:00Z"))
        self.assertEqual(ch[-1][1], "2026-10-01T00:00:00Z")
        for (a, b), (c, d) in zip(ch, ch[1:]):
            self.assertEqual(b, c)
        self.assertEqual(len(ch), 39)  # 273 days -> 38 full weeks + 1 partial

    def test_request_budget_exhaustion_is_a_fetch_failure_not_a_crash(self):
        h = Http(max_requests=0)
        with self.assertRaises(FetchFailed):
            h.get("https://gamma-api.polymarket.com/markets/keyset?limit=1")

    def test_fetch_failure_rule(self):
        self.assertFalse(fetch_failures_too_high(0, 0))
        self.assertFalse(fetch_failures_too_high(2, 98))   # exactly 2% -> still acceptable
        self.assertTrue(fetch_failures_too_high(3, 97))    # 3% -> REVIEW_REQUIRED
        self.assertTrue(fetch_failures_too_high(1, 0))

    def test_http_host_allow_list(self):
        with self.assertRaises(ValueError):
            Http().get("https://example.com/x")
        self.assertEqual(ALLOWED_HOSTS, {"gamma-api.polymarket.com", "clob.polymarket.com"})


if __name__ == "__main__":
    unittest.main()
