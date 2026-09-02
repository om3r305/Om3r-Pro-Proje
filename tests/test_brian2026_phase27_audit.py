from dataclasses import replace
from datetime import datetime, timezone
import unittest

import numpy as np

from brian2026.market_structure import (
    StructureCandle,
    StructureConfig,
    compute_market_structure,
)
from brian2026.phase27_experiment import (
    HORIZON,
    _clean_label_horizon,
    _expanding_year_splits,
    _label_resolution_timestamps,
)
from brian2026.phase27_specialists import market_structure_specialist
from brian2026.portfolio import PortfolioBar, PortfolioConfig, simulate_portfolio
from brian2026.structure_exit import (
    StructureExitConfig,
    simulate_structure_aware_portfolio,
)


def _rows():
    base = 1_600_000_000.0
    return tuple(
        StructureCandle(base + i * 300, 100.0, 101.0, 99.0, 100.0, 100.0 + i)
        for i in range(4)
    )


def _features():
    return compute_market_structure(
        _rows(), StructureConfig(left_bars=1, right_bars=1, atr_period=2, volume_period=2)
    )


def _bars(high_second=101.0):
    rows = list(_rows())
    if high_second != 101.0:
        rows[1] = replace(rows[1], high=high_second)
    return tuple(PortfolioBar(x.close_timestamp, x.open, x.high, x.low, x.close) for x in rows)


class Phase27AuditInvariantTests(unittest.TestCase):
    def test_fixed_structure_mode_is_exact_portfolio_passthrough(self):
        bars = _bars()
        actions = ("BUY", "WAIT", "WAIT", "WAIT")
        features = _features()
        config = PortfolioConfig(stop_loss_pct=50, take_profit_pct=50, max_holding_bars=10)
        expected = simulate_portfolio(bars, actions, config)
        actual = simulate_structure_aware_portfolio(
            bars, actions, features, StructureExitConfig("fixed"), config
        )
        self.assertEqual(actual, expected)

    def test_structure_exit_uses_real_portfolio_state_and_reason(self):
        bars = _bars()
        actions = ("BUY", "WAIT", "WAIT", "WAIT")
        features = list(_features())
        features[1] = replace(features[1], bearish_choch=True, atr=1.0)
        features[2] = replace(features[2], bearish_choch=True, atr=1.0)
        config = PortfolioConfig(stop_loss_pct=50, take_profit_pct=50, max_holding_bars=10)
        result = simulate_structure_aware_portfolio(
            bars, actions, tuple(features), StructureExitConfig("hybrid"), config
        )
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].reason, "STRUCTURE_BEARISH_CHOCH")
        self.assertEqual(result.trades[0].exit_timestamp, bars[1].timestamp)

    def test_lifecycle_exit_has_priority_over_structure_exit(self):
        bars = _bars(high_second=103.0)
        actions = ("BUY", "WAIT", "WAIT", "WAIT")
        features = list(_features())
        features[1] = replace(features[1], bearish_choch=True, atr=1.0)
        config = PortfolioConfig(stop_loss_pct=50, take_profit_pct=1, max_holding_bars=10)
        result = simulate_structure_aware_portfolio(
            bars, actions, tuple(features), StructureExitConfig("hybrid"), config
        )
        self.assertEqual(result.trades[0].reason, "TARGET")

    def test_failed_breakdown_is_bullish_exit_evidence_for_short_only(self):
        bars = _bars()
        features = list(_features())
        features[1] = replace(
            features[1], failed_breakdown=True, failed_breakout=False, atr=1.0
        )
        config = PortfolioConfig(stop_loss_pct=50, take_profit_pct=50, max_holding_bars=10)

        long_result = simulate_structure_aware_portfolio(
            bars,
            ("BUY", "WAIT", "WAIT", "WAIT"),
            tuple(features),
            StructureExitConfig("hybrid", exit_on_choch=False, exit_on_bos=False,
                                exit_on_momentum_deterioration=False,
                                trail_confirmed_structure=False),
            config,
        )
        self.assertNotEqual(long_result.trades[0].reason, "STRUCTURE_FAILED_BREAKDOWN")

        short_result = simulate_structure_aware_portfolio(
            bars,
            ("SELL", "WAIT", "WAIT", "WAIT"),
            tuple(features),
            StructureExitConfig("hybrid", exit_on_choch=False, exit_on_bos=False,
                                exit_on_momentum_deterioration=False,
                                trail_confirmed_structure=False),
            config,
        )
        self.assertEqual(short_result.trades[0].reason, "STRUCTURE_FAILED_BREAKDOWN")

    def test_label_resolution_is_explicit_and_horizon_correct(self):
        t = np.arange(20, dtype=float) * 300.0
        resolution = _label_resolution_timestamps(t)
        self.assertEqual(resolution[0], t[HORIZON])
        self.assertTrue(np.isinf(resolution[-1]))

    def test_source_gap_clean_mask_rejects_labels_crossing_gap(self):
        clean = np.ones(20, dtype=bool)
        clean[8] = False
        usable = _clean_label_horizon(clean)
        self.assertFalse(usable[2])  # its 30-minute path reaches index 8
        self.assertFalse(usable[8])
        self.assertTrue(usable[9])

    def test_yearly_robustness_is_past_only(self):
        stamps = np.asarray([
            datetime(2020, 1, 2, tzinfo=timezone.utc).timestamp(),
            datetime(2020, 12, 1, tzinfo=timezone.utc).timestamp(),
            datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp(),
            datetime(2021, 12, 1, tzinfo=timezone.utc).timestamp(),
            datetime(2022, 1, 2, tzinfo=timezone.utc).timestamp(),
            datetime(2022, 12, 1, tzinfo=timezone.utc).timestamp(),
        ])
        resolution = stamps + 1800.0
        splits = list(_expanding_year_splits(stamps, resolution))
        self.assertTrue(splits)
        for _, train, test in splits:
            self.assertLess(float(stamps[train].max()), float(stamps[test].min()))

    def test_market_structure_specialist_uses_completed_htf_context(self):
        feature = replace(_features()[-1], state="RANGE", atr=1.0, bullish_sweep=True)
        base = market_structure_specialist(feature)
        aligned = market_structure_specialist(feature, structure_15m=1.0, structure_1h=1.0)
        self.assertEqual(base.action, "BUY")
        self.assertEqual(aligned.action, "BUY")
        self.assertGreater(aligned.confidence, base.confidence)


if __name__ == "__main__":
    unittest.main()
