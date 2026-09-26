from __future__ import annotations

import pytest

from brian2026.phase53_turnover_rebalance import (
    TurnoverConfig,
    plan_turnover_constrained_rebalance,
)


def test_turnover_plan_reaches_target_when_within_budget() -> None:
    result = plan_turnover_constrained_rebalance(
        {"BTC": 0.20, "ETH": 0.10},
        {"BTC": 0.25, "ETH": 0.05},
        config=TurnoverConfig(max_l1_turnover=0.20),
    )
    assert result.desired_turnover == pytest.approx(0.10)
    assert result.planned_turnover == pytest.approx(0.10)
    assert result.planned_weights == pytest.approx({"BTC": 0.25, "ETH": 0.05})
    assert result.residual_l1_to_target == pytest.approx(0.0)


def test_discretionary_rotation_is_scaled_to_l1_turnover_budget() -> None:
    result = plan_turnover_constrained_rebalance(
        {"BTC": 0.20, "ETH": 0.20},
        {"BTC": 0.40, "ETH": 0.40},
        config=TurnoverConfig(max_l1_turnover=0.10),
    )
    assert result.mandatory_risk_turnover == pytest.approx(0.0)
    assert result.discretionary_turnover == pytest.approx(0.40)
    assert result.discretionary_scale == pytest.approx(0.25)
    assert result.planned_turnover == pytest.approx(0.10)
    assert result.planned_weights == pytest.approx({"BTC": 0.25, "ETH": 0.25})


def test_hard_risk_reduction_is_not_blocked_by_turnover_limit() -> None:
    result = plan_turnover_constrained_rebalance(
        {"BTC": 0.60, "ETH": 0.30},
        {"BTC": 0.10, "ETH": 0.05},
        config=TurnoverConfig(max_l1_turnover=0.20, risk_reduction_bypass=True),
    )
    assert result.mandatory_risk_turnover == pytest.approx(0.75)
    assert result.planned_weights == pytest.approx({"BTC": 0.10, "ETH": 0.05})
    assert result.planned_turnover == pytest.approx(0.75)
    assert result.turnover_limit_exceeded_only_for_risk_reduction is True
    assert result.risk_reductions_not_blocked is True


def test_opposite_side_flip_closes_first_then_uses_remaining_budget_to_reopen() -> None:
    result = plan_turnover_constrained_rebalance(
        {"BTC": 0.30},
        {"BTC": -0.30},
        config=TurnoverConfig(max_l1_turnover=0.45, risk_reduction_bypass=True),
    )
    # Mandatory close to flat consumes 0.30 turnover. Only 0.15 remains
    # for the requested 0.30 short opening.
    assert result.mandatory_risk_turnover == pytest.approx(0.30)
    assert result.discretionary_turnover == pytest.approx(0.30)
    assert result.discretionary_scale == pytest.approx(0.50)
    assert result.planned_weights["BTC"] == pytest.approx(-0.15)
    assert result.planned_turnover == pytest.approx(0.45)


def test_when_risk_bypass_disabled_all_changes_share_same_turnover_scale() -> None:
    result = plan_turnover_constrained_rebalance(
        {"BTC": 0.60, "ETH": 0.10},
        {"BTC": 0.20, "ETH": 0.30},
        config=TurnoverConfig(max_l1_turnover=0.30, risk_reduction_bypass=False),
    )
    assert result.desired_turnover == pytest.approx(0.60)
    assert result.planned_turnover == pytest.approx(0.30)
    assert result.discretionary_scale == pytest.approx(0.50)
    assert result.planned_weights["BTC"] == pytest.approx(0.40)
    assert result.planned_weights["ETH"] == pytest.approx(0.20)
    assert result.turnover_limit_exceeded_only_for_risk_reduction is False


def test_planner_never_overshoots_target_path() -> None:
    current = {"BTC": 0.10, "ETH": -0.20, "SOL": 0.0}
    target = {"BTC": 0.50, "ETH": -0.05, "SOL": 0.20}
    result = plan_turnover_constrained_rebalance(
        current,
        target,
        config=TurnoverConfig(max_l1_turnover=0.25),
    )
    for leg in result.legs:
        low = min(leg.current_weight, leg.target_weight) - 1e-12
        high = max(leg.current_weight, leg.target_weight) + 1e-12
        assert low <= leg.planned_weight <= high
    assert result.shadow_only is True
    assert result.live_execution is False


def test_nonfinite_weights_fail_closed() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        plan_turnover_constrained_rebalance(
            {"BTC": float("nan")},
            {"BTC": 0.2},
        )
