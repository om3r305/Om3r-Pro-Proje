from pathlib import Path
import tempfile
import unittest

try:
    from brian2026.bridge import LegacyShadowBridge
    from brian2026.engine import BrianEngine
    from brian2026.features import FeatureSnapshot, from_closed_candles
    from brian2026.metrics import calculate
    from brian2026.promotion_gate import PromotionGate, PromotionPolicy
    from brian2026.replay_lab import Candle, ReplayConfig, simulate_long
    from brian2026.safety import shadow_workflow_guard
    from brian2026.specialists import run_specialists
    from brian2026.types import MarketSnapshot, TradeOutcome
except ModuleNotFoundError:
    from Proje1.brian2026.bridge import LegacyShadowBridge
    from Proje1.brian2026.engine import BrianEngine
    from Proje1.brian2026.features import FeatureSnapshot, from_closed_candles
    from Proje1.brian2026.metrics import calculate
    from Proje1.brian2026.promotion_gate import PromotionGate, PromotionPolicy
    from Proje1.brian2026.replay_lab import Candle, ReplayConfig, simulate_long
    from Proje1.brian2026.safety import shadow_workflow_guard
    from Proje1.brian2026.specialists import run_specialists
    from Proje1.brian2026.types import MarketSnapshot, TradeOutcome


class BrianCoreTests(unittest.TestCase):
    def test_shadow_mode_cannot_be_disabled_by_config(self):
        with tempfile.TemporaryDirectory() as td:
            b = BrianEngine({"shadow_only": False}, runtime_root=td)
            d = b.decide(MarketSnapshot("X", 100.0))
            self.assertTrue(b.shadow_only)
            self.assertTrue(d.shadow_only)
            self.assertFalse(any(hasattr(b, name) for name in ("execute", "place_order", "open_trade", "close_trade")))

    def test_missing_features_are_explicit_and_specialists_abstain(self):
        snap = FeatureSnapshot("X", 123.0, 100.0, "UNKNOWN")
        market = snap.to_market()
        self.assertEqual(market.features, {})
        self.assertIn("ema_fast", market.context["unavailable_features"])
        votes = run_specialists(market)
        self.assertTrue(all(v.action == "WAIT" for v in votes))
        self.assertTrue(all(v.rationale.startswith("unavailable:") for v in votes))

    def test_closed_candle_snapshot_is_real_and_deterministic(self):
        rows = []
        for i in range(30):
            close = 100.0 + i * 0.2
            rows.append((1_000_000 + i * 60_000, close - 0.1, close + 0.2,
                         close - 0.2, close, 1_000.0 + i * 10.0))
        a = from_closed_candles(
            symbol="X", price=106.0, regime="TREND", candles=rows,
            order_book={"spread_bps": 2.0, "book_imbalance": 0.25, "wall_score": 0.5},
            legacy_predictor_confidence=0.77, legacy_signal_fired=True,
            legacy_slot="pred", timestamp=999.0,
        )
        b = from_closed_candles(
            symbol="X", price=106.0, regime="TREND", candles=rows,
            order_book={"spread_bps": 2.0, "book_imbalance": 0.25, "wall_score": 0.5},
            legacy_predictor_confidence=0.77, legacy_signal_fired=True,
            legacy_slot="pred", timestamp=999.0,
        )
        self.assertEqual(a, b)
        self.assertIsNotNone(a.ema_fast)
        self.assertIsNotNone(a.rsi)
        self.assertEqual(a.recent_high, max(row[2] for row in rows[-21:-1]))
        self.assertEqual(a.to_market().features["book_imbalance"], 0.25)

    def test_typed_bridge_remains_observational(self):
        with tempfile.TemporaryDirectory() as td:
            bridge = LegacyShadowBridge(BrianEngine({}, runtime_root=td))
            snapshot = FeatureSnapshot("X", 1.0, 100.0, "UNKNOWN", legacy_slot="pred")
            result = bridge.review_snapshot(snapshot)
            self.assertTrue(result["shadow_only"])
            self.assertNotIn("qty", result)

    def test_shadow_workflow_guard(self):
        self.assertTrue(shadow_workflow_guard({"brian2026": {"shadow_enabled": True}}))
        self.assertFalse(shadow_workflow_guard({"brian2026": {"shadow_enabled": False}}))

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
