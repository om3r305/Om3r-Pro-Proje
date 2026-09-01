from pathlib import Path
import tempfile
import unittest

try:
    from brian2026.engine import BrianEngine
    from brian2026.metrics import calculate
    from brian2026.promotion_gate import PromotionGate, PromotionPolicy
    from brian2026.replay_lab import Candle, ReplayConfig, simulate_long
    from brian2026.types import MarketSnapshot, TradeOutcome
except ModuleNotFoundError:
    from Proje1.brian2026.engine import BrianEngine
    from Proje1.brian2026.metrics import calculate
    from Proje1.brian2026.promotion_gate import PromotionGate, PromotionPolicy
    from Proje1.brian2026.replay_lab import Candle, ReplayConfig, simulate_long
    from Proje1.brian2026.types import MarketSnapshot, TradeOutcome


class BrianCoreTests(unittest.TestCase):
    def test_strong_trend_can_produce_directional_decision(self):
        with tempfile.TemporaryDirectory() as td:
            b = BrianEngine({
                "shadow_only": True,
                "meta": {"min_confidence": 0.55, "min_consensus": 0.50, "min_abs_score": 0.05, "disagreement_wait": 0.90},
                "risk": {"min_trade_confidence": 0.55}
            }, runtime_root=td)
            s = MarketSnapshot("X", 100.0, regime="TREND", features={
                "ema_fast": 102.0, "ema_slow": 99.0, "ema_slope_pct": 0.4,
                "rsi": 58.0, "return_5": 0.5, "book_imbalance": 0.6,
                "spread_bps": 1.0, "breakout_score": 0.8, "volume_z": 2.0,
                "acceleration": 0.2, "zscore": 0.1, "bb_position": 0.55, "atr_pct": 1.0
            })
            d = b.decide(s)
            self.assertIn(d.action, {"BUY", "WAIT"})
            self.assertTrue(d.shadow_only)

    def test_replay_is_deterministic_and_charges_costs(self):
        candles = [
            Candle(100, 100.2, 99.9, 100),
            Candle(100, 101.2, 99.95, 101),
            Candle(101, 101.5, 100.8, 101.2),
        ]
        cfg = ReplayConfig(tp_pct=0.8, sl_pct=0.5, fee_bps_each_side=10, slippage_bps_each_side=0)
        a = simulate_long(candles, 0, cfg)
        b = simulate_long(candles, 0, cfg)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a, 0.6, places=6)

    def test_promotion_requires_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            gate = PromotionGate(td, PromotionPolicy(min_trades=10, min_profit_factor=1.1))
            weak = calculate([1.0, -0.5])
            r = gate.review(weak)
            self.assertEqual(r["status"], "REJECTED")
            self.assertFalse(r["live_applied"])

    def test_learning_updates_memory(self):
        with tempfile.TemporaryDirectory() as td:
            b = BrianEngine({"meta": {"min_confidence": 0.5, "min_consensus": 0.5, "min_abs_score": 0.0, "disagreement_wait": 1.0},
                             "risk": {"min_trade_confidence": 0.5}}, runtime_root=td)
            s = MarketSnapshot("X", 100.0, regime="TREND", features={"ema_fast": 101, "ema_slow": 99, "return_5": 0.4})
            d = b.decide(s)
            b.learn(TradeOutcome(d.decision_id, "X", 1.0, 1.0, "TP", 60))
            rows = b.memory.tail(kind="decision_outcome")
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
