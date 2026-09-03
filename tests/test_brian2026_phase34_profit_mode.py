from __future__ import annotations

from brian2026.counterfactual_learner import (
    CausalCounterfactualLearner,
    FEATURE_NAMES,
    _AssetState,
)
from brian2026.curriculum_runner import PolicyObservation
from brian2026.market_gym import GymBar, GymFrame, MarketGymConfig
from brian2026.phase34_profit_exam import candidate_gate, evaluate_frozen_policy
from brian2026.profit_mode import FrozenNativePolicy, ProfitSeekingShadowPolicy
from brian2026.world_model import HistoricalBar, MultiAssetHistory


def _trained_learner(bias_prediction: float, *, error: float = 0.001) -> CausalCounterfactualLearner:
    learner = CausalCounterfactualLearner()
    learner._models["A"] = _AssetState(
        weights=[bias_prediction] + [0.0] * (len(FEATURE_NAMES) - 1),
        weighted_samples=100.0,
        updates=100,
        error_ewma=error,
    )
    return learner


def _observation() -> PolicyObservation:
    frames = (
        GymFrame(1_700_000_000.0, (GymBar("A", 1_700_000_000.0, 100.0, 101.0, 99.0, 100.0, 10.0, 1_700_000_000.0),)),
        GymFrame(1_700_000_300.0, (GymBar("A", 1_700_000_300.0, 100.0, 101.0, 99.0, 100.0, 10.0, 1_700_000_300.0),)),
    )
    return PolicyObservation(frames, 500.0, (), 1, 500.0)


def _history(move: float, length: int = 40) -> MultiAssetHistory:
    price = 100.0
    rows = []
    for index in range(length):
        timestamp = 1_700_000_000.0 + index * 300.0
        open_price = price
        close = open_price * (1.0 + move)
        rows.append(HistoricalBar(
            "A",
            timestamp,
            open_price,
            max(open_price, close) * 1.001,
            min(open_price, close) * 0.999,
            close,
            100.0,
        ))
        price = close
    return MultiAssetHistory.from_mapping("phase34-test-history", {"A": rows})


def test_profit_mode_rejects_edge_that_does_not_pay_conservative_round_trip_cost() -> None:
    learner = _trained_learner(0.0028)
    gym_config = MarketGymConfig(fee_bps=10.0, assumed_spread_bps=2.0, slippage_bps=1.0)
    native = FrozenNativePolicy(learner)
    profit = ProfitSeekingShadowPolicy(learner, gym_config)

    assert native.act(_observation()).weights == (("A", 0.25),)
    assert profit.act(_observation()).weights == ()


def test_profit_mode_takes_strong_positive_net_edge_without_mutating_learner() -> None:
    learner = _trained_learner(0.02)
    before = learner.training_state_id
    policy = ProfitSeekingShadowPolicy(learner, MarketGymConfig())
    allocation = policy.act(_observation())

    assert allocation.weights == (("A", 0.25),)
    assert learner.training_state_id == before
    assert policy.training_state_id != before


def test_profit_mode_is_hard_unlevered_and_respects_drawdown_budget() -> None:
    learner = _trained_learner(0.02)
    policy = ProfitSeekingShadowPolicy(learner, MarketGymConfig())
    observation = _observation()
    deep_drawdown = PolicyObservation(
        observation.visible_frames,
        equity=200.0,
        current_weights=(),
        step_index=observation.step_index,
        starting_equity=500.0,
    )
    allocation = policy.act(deep_drawdown)
    assert allocation.gross_exposure == 0.0


def test_frozen_profit_exam_can_gain_on_simple_trend_and_preserves_checkpoint() -> None:
    learner = _trained_learner(0.02)
    state = learner.training_state_id
    policy = ProfitSeekingShadowPolicy(
        learner,
        MarketGymConfig(fee_bps=0.0, assumed_spread_bps=0.0, slippage_bps=0.0),
    )
    result, metrics = evaluate_frozen_policy(
        _history(0.01),
        learner,
        policy,
        gym_config=MarketGymConfig(fee_bps=0.0, assumed_spread_bps=0.0, slippage_bps=0.0),
    )
    assert result.return_pct > 0
    assert metrics["positive_quarters"] >= 2
    assert learner.training_state_id == state


def test_candidate_gate_requires_robust_net_gain_not_just_activity() -> None:
    native = {
        "return_pct": 1.0,
    }
    robust_profit = {
        "return_pct": 1.5,
        "net_step_profit_factor": 1.2,
        "max_drawdown_pct": 4.0,
        "positive_quarters": 3,
        "ruined": False,
    }
    decision, checks = candidate_gate(native, robust_profit)
    assert decision == "DEVELOPMENT_CANDIDATE"
    assert all(checks.values())

    fragile_profit = dict(robust_profit)
    fragile_profit["positive_quarters"] = 1
    decision, checks = candidate_gate(native, fragile_profit)
    assert decision == "INSUFFICIENT_EVIDENCE"
    assert checks["at_least_two_positive_chronological_quarters"] is False


def test_candidate_gate_rejects_profit_mode_that_underperforms_frozen_native() -> None:
    native = {"return_pct": 2.0}
    profit = {
        "return_pct": 1.0,
        "net_step_profit_factor": 1.1,
        "max_drawdown_pct": 3.0,
        "positive_quarters": 3,
        "ruined": False,
    }
    decision, checks = candidate_gate(native, profit)
    assert decision == "INSUFFICIENT_EVIDENCE"
    assert checks["not_worse_than_frozen_native_return"] is False
