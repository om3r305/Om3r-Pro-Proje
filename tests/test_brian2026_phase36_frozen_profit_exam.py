from __future__ import annotations

from pathlib import Path
import json

from brian2026.counterfactual_learner import CausalCounterfactualLearner, FEATURE_NAMES, _AssetState
from brian2026.phase36_frozen_profit_exam import (
    COST_STRESS_MULTIPLIERS,
    MIN_ACTIVE_STEPS,
    PHASE36_EXAM_MONTHS,
    PHASE36_SYMBOLS,
    _evaluate,
    candidate_gate,
)
from brian2026.world_model import HistoricalBar, MultiAssetHistory


def _trained_learner() -> CausalCounterfactualLearner:
    learner = CausalCounterfactualLearner()
    for index, symbol in enumerate(PHASE36_SYMBOLS):
        weights = [0.0] * len(FEATURE_NAMES)
        weights[0] = 0.001 + index * 0.0001
        learner._models[symbol] = _AssetState(
            weights=weights,
            weighted_samples=1000.0,
            updates=1000,
            error_ewma=0.0001,
        )
    learner.episodes_learned = 1000
    return learner


def _history(rows: int = 80) -> MultiAssetHistory:
    bases = {"BTCUSDT": 60000.0, "ETHUSDT": 3000.0, "SOLUSDT": 120.0, "BNBUSDT": 500.0, "XRPUSDT": 0.6}
    mapping = {}
    for symbol in PHASE36_SYMBOLS:
        price = bases[symbol]
        bars = []
        for i in range(rows):
            ts = 1710000000.0 + i * 300.0
            close = price * (1.001 if i % 4 < 3 else 0.999)
            bars.append(HistoricalBar(symbol, ts, price, max(price, close) * 1.0002, min(price, close) * 0.9998, close, 1000.0 + i))
            price = close
        mapping[symbol] = tuple(bars)
    return MultiAssetHistory.from_mapping("phase36-test-history", mapping)


def _policy_result(return_pct: float, active_steps: int = 50, pf: float | None = 1.2, dd: float = 2.0, ruined: bool = False):
    return {"metrics": {"return_pct": return_pct, "active_steps": active_steps, "net_step_profit_factor": pf, "max_drawdown_pct": dd, "ruined": ruined}}


def test_phase36_exam_months_and_cost_stresses_are_preregistered() -> None:
    assert PHASE36_EXAM_MONTHS == ((2024, 3), (2024, 4))
    assert COST_STRESS_MULTIPLIERS == (1.0, 1.5, 2.0)
    assert MIN_ACTIVE_STEPS == 20
    assert all(year < 2026 for year, _ in PHASE36_EXAM_MONTHS)


def test_candidate_gate_requires_both_months_and_cost_stress() -> None:
    monthly = {"2024-03": _policy_result(0.5), "2024-04": _policy_result(0.2)}
    native = _policy_result(0.3)
    stresses = {"1.0x": _policy_result(0.6), "1.5x": _policy_result(0.1), "2.0x": _policy_result(-0.1)}
    decision, checks = candidate_gate(monthly, native, stresses)
    assert decision == "DEVELOPMENT_CANDIDATE"
    assert all(checks.values())

    monthly["2024-04"] = _policy_result(-0.01)
    decision, checks = candidate_gate(monthly, native, stresses)
    assert decision == "INSUFFICIENT_EVIDENCE"
    assert checks["both_reserved_months_net_positive"] is False


def test_frozen_evaluation_does_not_mutate_checkpoint() -> None:
    learner = _trained_learner()
    checkpoint = learner.state_dict()
    before = json.dumps(checkpoint, sort_keys=True)
    result = _evaluate(_history(), checkpoint, mode="profit", gym_config=__import__("brian2026.phase35_training", fromlist=["PHASE35_GYM_CONFIG"]).PHASE35_GYM_CONFIG)
    after = json.dumps(checkpoint, sort_keys=True)
    assert before == after
    assert result["policy"]["shadow_only"] is True
    assert result["policy"]["live_execution"] is False
    assert result["metrics"]["steps"] > 0
