from __future__ import annotations

import pytest

from brian2026.phase45_execution_contract import TradeIntent
from brian2026.phase55_rebalance_execution_intents import RiskReductionIntent
from brian2026.phase56_pretrade_risk_engine import (
    InstrumentRiskLimits,
    PreTradeAccountState,
    PreTradeRiskEngine,
)
from brian2026.phase68_operational_risk_governor import (
    ClosedTrade,
    EquityPoint,
    OperationalRiskGovernor,
    OperationalRiskPolicy,
    RuntimeHealthEvent,
)


NOW = 1_760_000_000.0


def _flat_equity() -> tuple[EquityPoint, ...]:
    return (
        EquityPoint(NOW - 600, 1000.0),
        EquityPoint(NOW, 1000.0),
    )


def _healthy_policy(**overrides) -> OperationalRiskPolicy:
    values = dict(
        drawdown_lookback_seconds=86_400,
        max_drawdown_fraction=0.20,
        daily_loss_lookback_seconds=86_400,
        max_daily_loss_fraction=0.20,
        stoploss_lookback_seconds=3_600,
        stoploss_limit=4,
        stoploss_required_profit=0.0,
        stoploss_lock_seconds=600,
        asset_cooldown_seconds=300,
        execution_failure_lookback_seconds=900,
        max_consecutive_execution_failures=3,
        reconciliation_failure_lookback_seconds=900,
        max_reconciliation_failures=2,
        unknown_outcome_lookback_seconds=3_600,
        max_unknown_order_outcomes=1,
        max_market_data_age_seconds=30.0,
    )
    values.update(overrides)
    return OperationalRiskPolicy(**values)


def _evaluate(
    governor: OperationalRiskGovernor,
    *,
    now: float = NOW,
    equity_points=(),
    trades=(),
    health=(),
    market_data_timestamp: float | None = None,
    manual_halt: bool = False,
    manual_release: bool = False,
):
    return governor.evaluate(
        now=now,
        equity_points=equity_points or _flat_equity(),
        closed_trades=trades,
        health_events=health,
        market_data_timestamp=(
            now if market_data_timestamp is None else market_data_timestamp
        ),
        manual_halt=manual_halt,
        manual_release=manual_release,
    )


def test_healthy_runtime_stays_active() -> None:
    governor = OperationalRiskGovernor(_healthy_policy())
    receipt = _evaluate(governor)
    assert receipt.trading_state == "ACTIVE"
    assert receipt.recommended_state == "ACTIVE"
    assert receipt.reasons == ()
    assert receipt.asset_open_allowed("BTCUSDT") is True
    assert receipt.live_execution is False


def test_max_drawdown_uses_strict_breach_and_halts_only_above_threshold() -> None:
    policy = _healthy_policy(
        max_drawdown_fraction=0.10,
        max_daily_loss_fraction=0.90,
    )

    exactly = OperationalRiskGovernor(policy)
    exact_receipt = _evaluate(
        exactly,
        equity_points=(
            EquityPoint(NOW - 300, 1000.0),
            EquityPoint(NOW, 900.0),
        ),
    )
    assert exact_receipt.max_drawdown_fraction == pytest.approx(0.10)
    assert exact_receipt.trading_state == "ACTIVE"

    breached = OperationalRiskGovernor(policy)
    breach_receipt = _evaluate(
        breached,
        equity_points=(
            EquityPoint(NOW - 300, 1000.0),
            EquityPoint(NOW - 100, 890.0),
            EquityPoint(NOW, 920.0),
        ),
    )
    assert breach_receipt.max_drawdown_fraction == pytest.approx(0.11)
    assert breach_receipt.trading_state == "HALTED"
    assert any(reason.startswith("max_drawdown:") for reason in breach_receipt.reasons)


def test_window_loss_can_halt_even_without_larger_intrawindow_drawdown_policy() -> None:
    governor = OperationalRiskGovernor(
        _healthy_policy(
            max_drawdown_fraction=0.50,
            max_daily_loss_fraction=0.05,
        )
    )
    receipt = _evaluate(
        governor,
        equity_points=(
            EquityPoint(NOW - 300, 1000.0),
            EquityPoint(NOW, 940.0),
        ),
    )
    assert receipt.window_loss_fraction == pytest.approx(0.06)
    assert receipt.trading_state == "HALTED"
    assert any(reason.startswith("window_loss:") for reason in receipt.reasons)


def test_stoploss_guard_enters_reducing_until_lock_expires() -> None:
    policy = _healthy_policy(
        stoploss_limit=3,
        stoploss_lock_seconds=600,
        max_drawdown_fraction=0.90,
        max_daily_loss_fraction=0.90,
    )
    trades = tuple(
        ClosedTrade(
            asset_id=f"ASSET{index}",
            closed_at=NOW - 100 - index * 10,
            pnl_quote=-5.0,
            return_fraction=-0.01,
            exit_reason="STOP_LOSS",
        )
        for index in range(3)
    )
    governor = OperationalRiskGovernor(policy)

    locked = _evaluate(governor, trades=trades)
    assert locked.trading_state == "REDUCING"
    assert locked.qualifying_stoplosses == 3
    assert locked.stoploss_lock_until is not None
    assert locked.stoploss_lock_until > NOW

    released = _evaluate(
        governor,
        now=NOW + 601,
        trades=trades,
        market_data_timestamp=NOW + 601,
        equity_points=(
            EquityPoint(NOW, 1000.0),
            EquityPoint(NOW + 601, 1000.0),
        ),
    )
    assert released.trading_state == "ACTIVE"


def test_stoploss_guard_ignores_profitable_or_non_stop_exits() -> None:
    governor = OperationalRiskGovernor(_healthy_policy(stoploss_limit=2))
    trades = (
        ClosedTrade("BTC", NOW - 10, 10.0, 0.02, "STOP_LOSS"),
        ClosedTrade("ETH", NOW - 20, -10.0, -0.02, "TARGET"),
    )
    receipt = _evaluate(governor, trades=trades)
    assert receipt.qualifying_stoplosses == 0
    assert receipt.trading_state == "ACTIVE"


def test_asset_cooldown_blocks_reentry_without_global_halt() -> None:
    governor = OperationalRiskGovernor(
        _healthy_policy(asset_cooldown_seconds=300)
    )
    trade = ClosedTrade(
        "BTCUSDT",
        NOW - 100,
        20.0,
        0.02,
        "TARGET",
    )
    receipt = _evaluate(governor, trades=(trade,))
    assert receipt.trading_state == "ACTIVE"
    assert receipt.blocked_assets == ("BTCUSDT",)
    assert receipt.asset_open_allowed("BTCUSDT") is False
    assert receipt.asset_open_allowed("ETHUSDT") is True

    later = _evaluate(
        governor,
        now=NOW + 250,
        trades=(trade,),
        market_data_timestamp=NOW + 250,
        equity_points=(
            EquityPoint(NOW, 1000.0),
            EquityPoint(NOW + 250, 1000.0),
        ),
    )
    assert later.blocked_assets == ()
    assert later.asset_open_allowed("BTCUSDT") is True


def test_consecutive_execution_failures_enter_reducing_and_success_resets_streak() -> None:
    policy = _healthy_policy(max_consecutive_execution_failures=3)
    events = (
        RuntimeHealthEvent(NOW - 30, "EXECUTION_FAILURE"),
        RuntimeHealthEvent(NOW - 20, "EXECUTION_FAILURE"),
        RuntimeHealthEvent(NOW - 10, "EXECUTION_FAILURE"),
    )
    governor = OperationalRiskGovernor(policy)
    bad = _evaluate(governor, health=events)
    assert bad.consecutive_execution_failures == 3
    assert bad.trading_state == "REDUCING"

    recovered = _evaluate(
        governor,
        now=NOW + 1,
        health=events + (RuntimeHealthEvent(NOW + 1, "EXECUTION_SUCCESS"),),
        market_data_timestamp=NOW + 1,
        equity_points=(
            EquityPoint(NOW, 1000.0),
            EquityPoint(NOW + 1, 1000.0),
        ),
    )
    assert recovered.consecutive_execution_failures == 0
    assert recovered.trading_state == "ACTIVE"


def test_reconciliation_failures_enter_reducing() -> None:
    governor = OperationalRiskGovernor(
        _healthy_policy(max_reconciliation_failures=2)
    )
    events = (
        RuntimeHealthEvent(NOW - 20, "RECONCILIATION_FAILURE"),
        RuntimeHealthEvent(NOW - 10, "RECONCILIATION_FAILURE"),
    )
    receipt = _evaluate(governor, health=events)
    assert receipt.reconciliation_failures == 2
    assert receipt.trading_state == "REDUCING"


def test_unknown_order_outcome_and_stale_market_data_halt() -> None:
    unknown = OperationalRiskGovernor(_healthy_policy())
    unknown_receipt = _evaluate(
        unknown,
        health=(RuntimeHealthEvent(NOW - 1, "UNKNOWN_ORDER_OUTCOME"),),
    )
    assert unknown_receipt.unknown_order_outcomes == 1
    assert unknown_receipt.trading_state == "HALTED"

    stale = OperationalRiskGovernor(_healthy_policy(max_market_data_age_seconds=10.0))
    stale_receipt = _evaluate(
        stale,
        market_data_timestamp=NOW - 10.1,
    )
    assert stale_receipt.market_data_age_seconds == pytest.approx(10.1)
    assert stale_receipt.trading_state == "HALTED"


def test_halted_state_is_latched_until_trigger_clears_and_manual_release_is_requested() -> None:
    governor = OperationalRiskGovernor(
        _healthy_policy(max_market_data_age_seconds=10.0)
    )
    first = _evaluate(governor, market_data_timestamp=NOW - 20)
    assert first.trading_state == "HALTED"
    assert first.halt_latched is True

    still_latched = _evaluate(
        governor,
        now=NOW + 1,
        market_data_timestamp=NOW + 1,
        equity_points=(
            EquityPoint(NOW, 1000.0),
            EquityPoint(NOW + 1, 1000.0),
        ),
    )
    assert still_latched.recommended_state == "ACTIVE"
    assert still_latched.trading_state == "HALTED"
    assert "halt_latched_manual_release_required" in still_latched.reasons

    released = _evaluate(
        governor,
        now=NOW + 2,
        market_data_timestamp=NOW + 2,
        equity_points=(
            EquityPoint(NOW, 1000.0),
            EquityPoint(NOW + 2, 1000.0),
        ),
        manual_release=True,
    )
    assert released.trading_state == "ACTIVE"
    assert released.halt_latched is False
    assert "manual_halt_release" in released.reasons


def test_manual_release_cannot_override_an_active_severe_trigger() -> None:
    governor = OperationalRiskGovernor(
        _healthy_policy(max_market_data_age_seconds=10.0)
    )
    receipt = _evaluate(
        governor,
        market_data_timestamp=NOW - 20,
        manual_release=True,
    )
    assert receipt.recommended_state == "HALTED"
    assert receipt.trading_state == "HALTED"
    assert receipt.halt_latched is True


def test_future_events_are_ignored_and_future_market_timestamp_is_rejected() -> None:
    governor = OperationalRiskGovernor(_healthy_policy())
    receipt = _evaluate(
        governor,
        equity_points=(
            EquityPoint(NOW - 10, 1000.0),
            EquityPoint(NOW, 1000.0),
            EquityPoint(NOW + 10, 1.0),
        ),
        trades=(
            ClosedTrade("BTC", NOW + 10, -999.0, -0.99, "LIQUIDATION"),
        ),
        health=(
            RuntimeHealthEvent(NOW + 10, "UNKNOWN_ORDER_OUTCOME"),
        ),
    )
    assert receipt.trading_state == "ACTIVE"
    assert receipt.qualifying_stoplosses == 0
    assert receipt.unknown_order_outcomes == 0

    with pytest.raises(ValueError, match="cannot be in the future"):
        _evaluate(
            governor,
            market_data_timestamp=NOW + 1,
        )


def test_reducing_receipt_wires_directly_into_phase56_new_risk_vs_reduce_only_gate() -> None:
    governor = OperationalRiskGovernor(
        _healthy_policy(max_consecutive_execution_failures=2)
    )
    receipt = _evaluate(
        governor,
        health=(
            RuntimeHealthEvent(NOW - 2, "EXECUTION_FAILURE"),
            RuntimeHealthEvent(NOW - 1, "EXECUTION_FAILURE"),
        ),
    )
    assert receipt.trading_state == "REDUCING"

    engine = PreTradeRiskEngine(
        receipt.pretrade_policy(
            InstrumentRiskLimits(max_notional_per_order=500.0)
        )
    )
    account = PreTradeAccountState(
        equity_usd=1000.0,
        available_cash_usd=500.0,
        open_position_weight=0.30,
    )
    new_risk = TradeIntent(
        intent_id="new-risk",
        asset_id="BTCUSDT",
        direction=1,
        target_weight=0.10,
        expected_edge_bps=20.0,
        confidence=0.8,
        max_slippage_bps=8.0,
        created_at=NOW,
        ttl_seconds=60,
        evidence_ids=("ev",),
    )
    reduction = RiskReductionIntent(
        intent_id="reduce",
        asset_id="BTCUSDT",
        current_direction=1,
        order_direction=-1,
        reduce_weight=0.20,
        current_weight=0.30,
        resulting_weight=0.10,
        reason="governor risk reduction",
        created_at=NOW,
        ttl_seconds=60,
    )

    assert engine.review(new_risk, account).allowed is False
    assert engine.review(reduction, account).allowed is True


def test_halted_receipt_blocks_both_new_risk_and_reduce_submissions_in_phase56() -> None:
    governor = OperationalRiskGovernor(
        _healthy_policy(max_market_data_age_seconds=10.0)
    )
    receipt = _evaluate(governor, market_data_timestamp=NOW - 20)
    assert receipt.trading_state == "HALTED"

    engine = PreTradeRiskEngine(receipt.pretrade_policy())
    account = PreTradeAccountState(1000.0, 500.0, 0.30)
    reduction = RiskReductionIntent(
        intent_id="reduce",
        asset_id="BTCUSDT",
        current_direction=1,
        order_direction=-1,
        reduce_weight=0.20,
        current_weight=0.30,
        resulting_weight=0.10,
        reason="halted reduction probe",
        created_at=NOW,
        ttl_seconds=60,
    )
    assert engine.review(reduction, account).allowed is False


def test_governor_receipt_is_deterministic_for_same_inputs_and_initial_state() -> None:
    kwargs = dict(
        now=NOW,
        equity_points=_flat_equity(),
        closed_trades=(),
        health_events=(),
        market_data_timestamp=NOW,
    )
    first = OperationalRiskGovernor(_healthy_policy()).evaluate(**kwargs)
    second = OperationalRiskGovernor(_healthy_policy()).evaluate(**kwargs)
    assert first.receipt_id == second.receipt_id
    assert first.to_dict() == second.to_dict()
