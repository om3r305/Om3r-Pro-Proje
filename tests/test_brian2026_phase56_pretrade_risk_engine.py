from __future__ import annotations

import pytest

from brian2026.phase45_execution_contract import TradeIntent
from brian2026.phase55_rebalance_execution_intents import RiskReductionIntent
from brian2026.phase56_pretrade_risk_engine import (
    InstrumentRiskLimits,
    PreTradeAccountState,
    PreTradeRiskEngine,
    PreTradeRiskPolicy,
    review_new_risk_intent,
    review_reduce_only_intent,
)


def _trade(weight: float = 0.10) -> TradeIntent:
    return TradeIntent(
        intent_id="new-risk",
        asset_id="BTCUSDT",
        direction=1 if weight > 0 else -1,
        target_weight=weight,
        expected_edge_bps=30.0,
        confidence=0.8,
        max_slippage_bps=8.0,
        created_at=1_760_000_000.0,
        ttl_seconds=60,
        evidence_ids=("ev",),
    )


def _reduction(
    *,
    current: float = 0.30,
    resulting: float = 0.10,
) -> RiskReductionIntent:
    direction = 1 if current > 0 else -1
    reduce_weight = abs(current - resulting)
    return RiskReductionIntent(
        intent_id="reduce",
        asset_id="BTCUSDT",
        current_direction=direction,
        order_direction=-direction,
        reduce_weight=reduce_weight,
        current_weight=current,
        resulting_weight=resulting,
        reason="risk reduction",
        created_at=1_760_000_000.0,
        ttl_seconds=60,
    )


def test_active_allows_new_risk_only_within_cash_and_notional_limits() -> None:
    receipt = review_new_risk_intent(
        _trade(0.10),
        trading_state="ACTIVE",
        account=PreTradeAccountState(
            equity_usd=1000.0,
            available_cash_usd=500.0,
            open_position_weight=0.0,
        ),
        limits=InstrumentRiskLimits(
            min_notional=10.0,
            max_notional=500.0,
            max_notional_per_order=200.0,
        ),
    )
    assert receipt.allowed is True
    assert receipt.requested_notional_usd == pytest.approx(100.0)
    assert receipt.projected_position_weight == pytest.approx(0.10)
    assert receipt.live_execution is False


def test_reducing_and_halted_states_block_new_risk() -> None:
    account = PreTradeAccountState(1000.0, 1000.0, 0.20)
    limits = InstrumentRiskLimits(max_notional_per_order=500.0)

    reducing = review_new_risk_intent(
        _trade(0.10),
        trading_state="REDUCING",
        account=account,
        limits=limits,
    )
    assert reducing.allowed is False
    assert "trading_state_active" in reducing.reasons

    halted = review_new_risk_intent(
        _trade(0.10),
        trading_state="HALTED",
        account=account,
        limits=limits,
    )
    assert halted.allowed is False
    assert "trading_state_active" in halted.reasons


def test_active_and_reducing_allow_valid_reduce_only_but_halted_denies() -> None:
    intent = _reduction(current=0.30, resulting=0.10)
    account = PreTradeAccountState(1000.0, 0.0, 0.30)
    limits = InstrumentRiskLimits(min_notional=1.0, max_notional_per_order=500.0)

    active = review_reduce_only_intent(
        intent,
        trading_state="ACTIVE",
        account=account,
        limits=limits,
    )
    assert active.allowed is True
    assert active.reduce_only is True
    assert active.projected_position_weight == pytest.approx(0.10)

    reducing = review_reduce_only_intent(
        intent,
        trading_state="REDUCING",
        account=account,
        limits=limits,
    )
    assert reducing.allowed is True

    halted = review_reduce_only_intent(
        intent,
        trading_state="HALTED",
        account=account,
        limits=limits,
    )
    assert halted.allowed is False
    assert "trading_state_allows_reduction" in halted.reasons


def test_reduce_only_must_match_identified_open_position_and_opposite_side() -> None:
    intent = _reduction(current=0.30, resulting=0.10)

    flat = review_reduce_only_intent(
        intent,
        trading_state="REDUCING",
        account=PreTradeAccountState(1000.0, 0.0, 0.0),
        limits=InstrumentRiskLimits(),
    )
    assert flat.allowed is False
    assert "identified_open_position" in flat.reasons

    wrong_side_account = review_reduce_only_intent(
        intent,
        trading_state="REDUCING",
        account=PreTradeAccountState(1000.0, 0.0, -0.30),
        limits=InstrumentRiskLimits(),
    )
    assert wrong_side_account.allowed is False
    assert "current_direction_matches_intent" in wrong_side_account.reasons


def test_reduce_only_cannot_exceed_or_flip_current_position() -> None:
    # Constructor itself rejects an over-reduction, so emulate a mismatch where
    # the real account exposure is smaller than the intent's identified state.
    intent = _reduction(current=0.30, resulting=0.10)
    receipt = review_reduce_only_intent(
        intent,
        trading_state="REDUCING",
        account=PreTradeAccountState(1000.0, 0.0, 0.15),
        limits=InstrumentRiskLimits(),
    )
    assert receipt.allowed is False
    assert "reduction_not_larger_than_position" in receipt.reasons
    assert receipt.projected_position_weight is None


def test_cash_and_max_notional_are_independent_hard_denials() -> None:
    intent = _trade(0.40)
    receipt = review_new_risk_intent(
        intent,
        trading_state="ACTIVE",
        account=PreTradeAccountState(
            equity_usd=1000.0,
            available_cash_usd=300.0,
            open_position_weight=0.0,
        ),
        limits=InstrumentRiskLimits(
            min_notional=10.0,
            max_notional=600.0,
            max_notional_per_order=350.0,
        ),
    )
    assert receipt.allowed is False
    assert "configured_max_notional_per_order" in receipt.reasons
    assert "cash_available" in receipt.reasons


def test_engine_has_no_bypass_and_dispatches_by_intent_type() -> None:
    engine = PreTradeRiskEngine(
        PreTradeRiskPolicy(
            trading_state="REDUCING",
            limits=InstrumentRiskLimits(max_notional_per_order=500.0),
        )
    )
    account = PreTradeAccountState(1000.0, 500.0, 0.30)

    new_risk = engine.review(_trade(0.10), account)
    reduction = engine.review(_reduction(current=0.30, resulting=0.10), account)

    assert new_risk.allowed is False
    assert reduction.allowed is True
    assert not hasattr(engine.policy, "bypass")


def test_min_notional_is_enforced() -> None:
    receipt = review_new_risk_intent(
        _trade(0.001),
        trading_state="ACTIVE",
        account=PreTradeAccountState(1000.0, 1000.0, 0.0),
        limits=InstrumentRiskLimits(min_notional=5.0),
    )
    assert receipt.allowed is False
    assert "instrument_min_notional" in receipt.reasons
