from __future__ import annotations

import pytest

from brian2026.phase53_turnover_rebalance import TurnoverConfig, plan_turnover_constrained_rebalance
from brian2026.phase55_rebalance_execution_intents import (
    FlatPositionReceipt,
    compile_rebalance_execution_plan,
    release_reversal_open,
)


TS = 1_760_000_000.0


def _compile(current, target, *, max_turnover=2.0):
    turnover = plan_turnover_constrained_rebalance(
        current,
        target,
        config=TurnoverConfig(max_l1_turnover=max_turnover, risk_reduction_bypass=True),
    )
    assets = set(current) | set(target)
    return compile_rebalance_execution_plan(
        turnover,
        expected_edge_bps_by_asset={asset: 30.0 for asset in assets},
        confidence_by_asset={asset: 0.8 for asset in assets},
        evidence_ids_by_asset={asset: (f"ev-{asset}",) for asset in assets},
        created_at=TS,
        max_slippage_bps=8.0,
        ttl_seconds=60,
    )


def test_open_uses_delta_weight_not_final_target_weight() -> None:
    plan = _compile({}, {"BTC": 0.25})
    instruction = plan.instructions[0]
    assert instruction.kind == "OPEN"
    assert instruction.trade_intent is not None
    assert instruction.trade_intent.target_weight == pytest.approx(0.25)

    # Existing 0.20 -> final 0.30 must order only +0.10, not +0.30.
    plan = _compile({"BTC": 0.20}, {"BTC": 0.30})
    instruction = plan.instructions[0]
    assert instruction.kind == "INCREASE"
    assert instruction.trade_intent.target_weight == pytest.approx(0.10)
    assert instruction.planned_weight == pytest.approx(0.30)


def test_same_side_reduction_is_reduce_only_and_does_not_require_alpha_to_lower_risk() -> None:
    turnover = plan_turnover_constrained_rebalance(
        {"BTC": 0.40},
        {"BTC": 0.15},
        config=TurnoverConfig(max_l1_turnover=1.0),
    )
    plan = compile_rebalance_execution_plan(
        turnover,
        expected_edge_bps_by_asset={},
        confidence_by_asset={},
        evidence_ids_by_asset={},
        created_at=TS,
        max_slippage_bps=8.0,
        ttl_seconds=60,
    )
    instruction = plan.instructions[0]
    assert instruction.kind == "REDUCE"
    reduction = instruction.reduction_intent
    assert reduction is not None
    assert reduction.reduce_only is True
    assert reduction.current_direction == 1
    assert reduction.order_direction == -1
    assert reduction.reduce_weight == pytest.approx(0.25)
    assert reduction.resulting_weight == pytest.approx(0.15)


def test_full_close_is_reduce_only_and_cannot_exceed_current_exposure() -> None:
    plan = _compile({"ETH": -0.30}, {"ETH": 0.0})
    instruction = plan.instructions[0]
    assert instruction.kind == "CLOSE"
    reduction = instruction.reduction_intent
    assert reduction is not None
    assert reduction.current_direction == -1
    assert reduction.order_direction == 1
    assert reduction.reduce_weight == pytest.approx(0.30)
    assert reduction.resulting_weight == pytest.approx(0.0)


def test_reversal_is_split_into_close_and_pending_open() -> None:
    plan = _compile({"BTC": 0.30}, {"BTC": -0.20})
    instruction = plan.instructions[0]
    assert instruction.kind == "REVERSAL_CLOSE"
    assert instruction.trade_intent is None
    assert instruction.reduction_intent is not None
    assert instruction.reduction_intent.resulting_weight == pytest.approx(0.0)
    assert instruction.reduction_intent.reduce_weight == pytest.approx(0.30)
    assert instruction.pending_reversal is not None
    assert instruction.pending_reversal.target_weight == pytest.approx(-0.20)
    assert instruction.pending_reversal.automatic_release is False


def test_reversal_open_requires_authoritative_reconciled_flat_receipt() -> None:
    plan = _compile({"BTC": 0.30}, {"BTC": -0.20})
    pending = plan.instructions[0].pending_reversal
    assert pending is not None

    with pytest.raises(ValueError, match="authoritative"):
        release_reversal_open(
            pending,
            FlatPositionReceipt("BTC", TS + 10, 0.0, True, False, "local-only"),
        )

    with pytest.raises(ValueError, match="completed reconciliation"):
        release_reversal_open(
            pending,
            FlatPositionReceipt("BTC", TS + 10, 0.0, False, True, "venue"),
        )

    with pytest.raises(ValueError, match="not flat"):
        release_reversal_open(
            pending,
            FlatPositionReceipt("BTC", TS + 10, 0.01, True, True, "venue"),
        )

    released = release_reversal_open(
        pending,
        FlatPositionReceipt("BTC", TS + 10, 0.0, True, True, "venue-reconciled"),
    )
    assert released.direction == -1
    assert released.target_weight == pytest.approx(-0.20)
    assert released.created_at == pytest.approx(TS + 10)
    assert released.live_execution is False


def test_expired_pending_reversal_does_not_open_new_risk() -> None:
    plan = _compile({"BTC": 0.30}, {"BTC": -0.20})
    pending = plan.instructions[0].pending_reversal
    assert pending is not None
    with pytest.raises(ValueError, match="expired"):
        release_reversal_open(
            pending,
            FlatPositionReceipt("BTC", TS + 61, 0.0, True, True, "venue"),
        )


def test_new_risk_leg_requires_grounded_evidence_and_edge_confidence() -> None:
    turnover = plan_turnover_constrained_rebalance(
        {},
        {"SOL": 0.20},
        config=TurnoverConfig(max_l1_turnover=1.0),
    )
    with pytest.raises(ValueError, match="missing edge/confidence"):
        compile_rebalance_execution_plan(
            turnover,
            expected_edge_bps_by_asset={},
            confidence_by_asset={},
            evidence_ids_by_asset={"SOL": ("ev",)},
            created_at=TS,
            max_slippage_bps=8.0,
            ttl_seconds=60,
        )

    with pytest.raises(ValueError, match="requires evidence ids"):
        compile_rebalance_execution_plan(
            turnover,
            expected_edge_bps_by_asset={"SOL": 20.0},
            confidence_by_asset={"SOL": 0.7},
            evidence_ids_by_asset={},
            created_at=TS,
            max_slippage_bps=8.0,
            ttl_seconds=60,
        )


def test_zero_delta_assets_are_skipped_and_plan_id_is_deterministic() -> None:
    first = _compile({"BTC": 0.20}, {"BTC": 0.20})
    second = _compile({"BTC": 0.20}, {"BTC": 0.20})
    assert first.instructions == ()
    assert first.skipped_assets == ("BTC",)
    assert first.plan_id == second.plan_id
    assert first.shadow_only is True
    assert first.live_execution is False
    assert first.automatic_release is False
