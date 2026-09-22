from __future__ import annotations

import pytest

from brian2026.phase44_portfolio_brain import (
    PortfolioRiskLimits,
    PortfolioSignal,
    construct_portfolio_book,
)
from brian2026.phase45_execution_contract import (
    CreateExecutorAction,
    ExecutionFill,
    PositionExecutorConfig,
    ShadowExecutorOrchestrator,
    ShadowPositionHold,
    StoreExecutorAction,
    StopExecutorAction,
    TradeIntent,
    TripleBarrierPolicy,
    create_position_executor_action,
    intents_from_portfolio_book,
)


def _intent() -> TradeIntent:
    return TradeIntent(
        intent_id="btc-long-1",
        asset_id="BTCUSDT",
        direction=1,
        target_weight=0.20,
        expected_edge_bps=35.0,
        confidence=0.78,
        max_slippage_bps=8.0,
        created_at=1_760_000_000.0,
        ttl_seconds=45,
        evidence_ids=("ev-a", "ev-b"),
    )


def test_trade_intent_is_time_bounded_and_shadow_only() -> None:
    intent = _intent()
    assert intent.expired(intent.created_at + 45) is False
    assert intent.expired(intent.created_at + 46) is True
    assert intent.shadow_only is True
    assert intent.live_execution is False

    with pytest.raises(ValueError, match="sign must match"):
        TradeIntent(
            intent_id="bad",
            asset_id="BTCUSDT",
            direction=1,
            target_weight=-0.2,
            expected_edge_bps=10,
            confidence=0.5,
            max_slippage_bps=5,
            created_at=1.0,
            ttl_seconds=10,
            evidence_ids=("ev",),
        )


def test_position_executor_config_is_forced_to_shadow_adapter() -> None:
    barrier = TripleBarrierPolicy(
        stop_loss_fraction=0.01,
        take_profit_fraction=0.02,
        time_limit_seconds=3600,
    )
    action = create_position_executor_action(
        _intent(),
        equity_usd=500.0,
        reference_price=60_000.0,
        barrier=barrier,
    )
    assert isinstance(action, CreateExecutorAction)
    assert action.executor_config.connector_name == "shadow"
    assert action.executor_config.amount_quote == pytest.approx(100.0)
    assert action.executor_config.side == "BUY"
    assert action.executor_config.live_execution is False

    with pytest.raises(ValueError, match="shadow adapter"):
        PositionExecutorConfig(
            executor_id="x",
            intent_id="y",
            connector_name="binance",
            trading_pair="BTCUSDT",
            side="BUY",
            amount_quote=100.0,
            entry_price=60_000.0,
            barrier=barrier,
        )


def test_triple_barrier_requires_market_stop_and_time_limit() -> None:
    with pytest.raises(ValueError, match="stop-loss must close at market"):
        TripleBarrierPolicy(stop_loss_fraction=0.01, stop_loss_order_type="LIMIT")
    with pytest.raises(ValueError, match="time-limit must close at market"):
        TripleBarrierPolicy(time_limit_seconds=60, time_limit_order_type="LIMIT")


def test_position_hold_incremental_accounting_matches_reduce_and_flip_semantics() -> None:
    hold = ShadowPositionHold("BTCUSDT")

    assert hold.apply_fill(ExecutionFill("f1", "o1", True, 1.0, 100.0, 0.1, 1.0))
    assert hold.net_amount_base == pytest.approx(1.0)
    assert hold.avg_entry_price == pytest.approx(100.0)

    assert hold.apply_fill(ExecutionFill("f2", "o2", True, 1.0, 110.0, 0.1, 2.0))
    assert hold.net_amount_base == pytest.approx(2.0)
    assert hold.avg_entry_price == pytest.approx(105.0)

    assert hold.apply_fill(ExecutionFill("f3", "o3", False, 0.5, 60.0, 0.1, 3.0))
    assert hold.net_amount_base == pytest.approx(1.5)
    assert hold.avg_entry_price == pytest.approx(105.0)
    assert hold.realized_pnl_quote == pytest.approx(7.5)

    # Sell two base at 90: close remaining 1.5 long, then flip 0.5 short at 90.
    assert hold.apply_fill(ExecutionFill("f4", "o4", False, 2.0, 180.0, 0.2, 4.0))
    assert hold.net_amount_base == pytest.approx(-0.5)
    assert hold.avg_entry_price == pytest.approx(90.0)
    assert hold.realized_pnl_quote == pytest.approx(-15.0)
    assert hold.unrealized_pnl(80.0) == pytest.approx(5.0)

    before = (
        hold.net_amount_base,
        hold.avg_entry_price,
        hold.realized_pnl_quote,
        hold.volume_traded_quote,
        hold.cumulative_fees_quote,
    )
    assert hold.apply_fill(ExecutionFill("f4", "o4", False, 2.0, 180.0, 0.2, 4.0)) is False
    after = (
        hold.net_amount_base,
        hold.avg_entry_price,
        hold.realized_pnl_quote,
        hold.volume_traded_quote,
        hold.cumulative_fees_quote,
    )
    assert after == before


def test_orchestrator_uses_create_stop_store_lifecycle_without_order_transport() -> None:
    action = create_position_executor_action(
        _intent(),
        equity_usd=500.0,
        reference_price=60_000.0,
        barrier=TripleBarrierPolicy(stop_loss_fraction=0.01, take_profit_fraction=0.02),
    )
    orchestrator = ShadowExecutorOrchestrator()
    state = orchestrator.apply(action)
    assert state.status == "RUNNING"
    assert state.hold is not None

    state = orchestrator.apply(StopExecutorAction("brian", state.config.executor_id, keep_position=True))
    assert state.status == "SHUTTING_DOWN"
    assert state.close_type == "POSITION_HOLD"

    state.terminate()
    assert state.status == "TERMINATED"
    stored = orchestrator.apply(StoreExecutorAction("brian", state.config.executor_id))
    assert stored.status == "TERMINATED"
    assert state.config.executor_id not in orchestrator.active
    assert state.config.executor_id in orchestrator.stored


def test_phase44_book_translates_to_evidence_lineaged_trade_intents() -> None:
    signals = (
        PortfolioSignal("news", "BTCUSDT", 0.8, evidence_ids=("ev-news-btc",)),
        PortfolioSignal("market", "BTCUSDT", 0.6, evidence_ids=("ev-market-btc",)),
        PortfolioSignal("news", "ETHUSDT", -0.7, evidence_ids=("ev-news-eth",)),
        PortfolioSignal("market", "ETHUSDT", -0.5, evidence_ids=("ev-market-eth",)),
    )
    plan = construct_portfolio_book(
        signals,
        {"news": 1.0, "market": 1.0},
        gross_target=0.6,
        limits=PortfolioRiskLimits(max_position_pct=0.35, max_gross_exposure=0.6),
    )
    intents = intents_from_portfolio_book(
        plan,
        created_at=1_760_000_000.0,
        expected_edge_bps_by_asset={"BTCUSDT": 40.0, "ETHUSDT": 25.0},
        confidence_by_asset={"BTCUSDT": 0.8, "ETHUSDT": 0.7},
        max_slippage_bps=8.0,
        ttl_seconds=45,
    )
    assert {intent.asset_id for intent in intents} == {"BTCUSDT", "ETHUSDT"}
    btc = next(intent for intent in intents if intent.asset_id == "BTCUSDT")
    eth = next(intent for intent in intents if intent.asset_id == "ETHUSDT")
    assert btc.direction == 1
    assert eth.direction == -1
    assert set(btc.evidence_ids) == set(plan.source_evidence_ids)
    assert all(intent.shadow_only and not intent.live_execution for intent in intents)
