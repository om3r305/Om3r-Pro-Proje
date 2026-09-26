from __future__ import annotations

import pytest

from brian2026.phase46_execution_simulator import LiquidityLevel, OrderBookSnapshot
from brian2026.phase53_turnover_rebalance import TurnoverConfig, plan_turnover_constrained_rebalance
from brian2026.phase55_rebalance_execution_intents import compile_rebalance_execution_plan
from brian2026.phase56_pretrade_risk_engine import InstrumentRiskLimits
from brian2026.phase57_shadow_execution_cycle import ExecutionMarketInput
from brian2026.phase68_operational_risk_governor import (
    ClosedTrade,
    EquityPoint,
    OperationalRiskGovernor,
    OperationalRiskPolicy,
    RuntimeHealthEvent,
)
from brian2026.phase69_governed_shadow_execution import (
    run_governed_shadow_execution_cycle,
)


TS = 1_760_000_000.0


def _policy(**overrides) -> OperationalRiskPolicy:
    values = dict(
        max_drawdown_fraction=0.50,
        max_daily_loss_fraction=0.50,
        stoploss_limit=10,
        asset_cooldown_seconds=300,
        max_consecutive_execution_failures=3,
        max_reconciliation_failures=3,
        max_unknown_order_outcomes=2,
        max_market_data_age_seconds=30.0,
    )
    values.update(overrides)
    return OperationalRiskPolicy(**values)


def _risk_receipt(
    *,
    trades=(),
    health=(),
    market_data_timestamp: float = TS,
    policy: OperationalRiskPolicy | None = None,
):
    governor = OperationalRiskGovernor(policy or _policy())
    return governor.evaluate(
        now=TS,
        equity_points=(
            EquityPoint(TS - 600, 1000.0),
            EquityPoint(TS, 1000.0),
        ),
        closed_trades=trades,
        health_events=health,
        market_data_timestamp=market_data_timestamp,
    )


def _plan(current, target):
    turnover = plan_turnover_constrained_rebalance(
        current,
        target,
        config=TurnoverConfig(max_l1_turnover=2.0, risk_reduction_bypass=True),
    )
    assets = set(current) | set(target)
    return compile_rebalance_execution_plan(
        turnover,
        expected_edge_bps_by_asset={asset: 30.0 for asset in assets},
        confidence_by_asset={asset: 0.8 for asset in assets},
        evidence_ids_by_asset={asset: (f"ev-{asset}",) for asset in assets},
        created_at=TS,
        max_slippage_bps=20.0,
        ttl_seconds=60,
    )


def _market(price: float = 100.0) -> ExecutionMarketInput:
    return ExecutionMarketInput(
        reference_price=price,
        tick_size=0.01,
        snapshots=(
            OrderBookSnapshot(
                timestamp=TS + 0.01,
                bids=(
                    LiquidityLevel(price - 0.10, 20.0),
                    LiquidityLevel(price - 0.20, 20.0),
                ),
                asks=(
                    LiquidityLevel(price, 20.0),
                    LiquidityLevel(price + 0.10, 20.0),
                ),
            ),
        ),
    )


def _run(plan, risk, current, *, cash=1000.0):
    assets = {instruction.asset_id for instruction in plan.instructions}
    return run_governed_shadow_execution_cycle(
        plan,
        operational_risk=risk,
        equity_usd=1000.0,
        available_cash_usd=cash,
        current_weights=current,
        markets={asset: _market() for asset in assets},
        risk_limits_by_asset={
            asset: InstrumentRiskLimits(max_notional_per_order=500.0)
            for asset in assets
        },
    )


def test_asset_cooldown_blocks_only_that_assets_new_risk() -> None:
    plan = _plan({}, {"BTCUSDT": 0.20, "ETHUSDT": 0.20})
    risk = _risk_receipt(
        trades=(
            ClosedTrade(
                "BTCUSDT",
                TS - 60,
                10.0,
                0.01,
                "TARGET",
            ),
        )
    )
    result = _run(plan, risk, {})

    assert result.trading_state == "ACTIVE"
    assert result.blocked_new_risk_assets == ("BTCUSDT",)
    by_asset = {item.asset_id: item for item in result.cycle.items}

    assert by_asset["BTCUSDT"].risk_receipt.allowed is False
    assert "asset_new_risk_allowed" in by_asset["BTCUSDT"].risk_receipt.reasons
    assert by_asset["BTCUSDT"].execution_receipt is None

    assert by_asset["ETHUSDT"].risk_receipt.allowed is True
    assert by_asset["ETHUSDT"].execution_receipt is not None
    assert by_asset["ETHUSDT"].execution_receipt.status == "FILLED"
    assert result.cycle.reserved_new_risk_cash_usd == pytest.approx(200.0)


def test_asset_cooldown_does_not_block_reduce_only_for_same_asset() -> None:
    plan = _plan({"BTCUSDT": 0.30}, {"BTCUSDT": 0.10})
    risk = _risk_receipt(
        trades=(
            ClosedTrade(
                "BTCUSDT",
                TS - 60,
                5.0,
                0.01,
                "TARGET",
            ),
        )
    )
    result = _run(plan, risk, {"BTCUSDT": 0.30})
    item = result.cycle.items[0]

    assert result.blocked_new_risk_assets == ("BTCUSDT",)
    assert item.instruction_kind == "REDUCE"
    assert item.risk_receipt.allowed is True
    assert item.risk_receipt.reduce_only is True
    assert item.execution_receipt is not None
    assert item.execution_receipt.status == "FILLED"


def test_global_reducing_allows_reduction_but_denies_unrelated_new_risk() -> None:
    plan = _plan(
        {"BTCUSDT": 0.30},
        {"BTCUSDT": 0.10, "ETHUSDT": 0.20},
    )
    risk = _risk_receipt(
        health=(
            RuntimeHealthEvent(TS - 3, "EXECUTION_FAILURE"),
            RuntimeHealthEvent(TS - 2, "EXECUTION_FAILURE"),
            RuntimeHealthEvent(TS - 1, "EXECUTION_FAILURE"),
        ),
        policy=_policy(max_consecutive_execution_failures=3),
    )
    result = _run(plan, risk, {"BTCUSDT": 0.30})
    by_asset = {item.asset_id: item for item in result.cycle.items}

    assert result.trading_state == "REDUCING"
    assert by_asset["BTCUSDT"].risk_receipt.allowed is True
    assert by_asset["BTCUSDT"].risk_receipt.reduce_only is True
    assert by_asset["BTCUSDT"].execution_receipt is not None

    assert by_asset["ETHUSDT"].risk_receipt.allowed is False
    assert "trading_state_active" in by_asset["ETHUSDT"].risk_receipt.reasons
    assert by_asset["ETHUSDT"].execution_receipt is None
    assert result.cycle.reserved_new_risk_cash_usd == pytest.approx(0.0)


def test_global_halted_blocks_reduction_and_new_risk() -> None:
    plan = _plan(
        {"BTCUSDT": 0.30},
        {"BTCUSDT": 0.10, "ETHUSDT": 0.20},
    )
    risk = _risk_receipt(
        market_data_timestamp=TS - 31,
        policy=_policy(max_market_data_age_seconds=30.0),
    )
    result = _run(plan, risk, {"BTCUSDT": 0.30})

    assert result.trading_state == "HALTED"
    assert set(result.cycle.denied_assets) == {"BTCUSDT", "ETHUSDT"}
    assert all(item.risk_receipt.allowed is False for item in result.cycle.items)
    assert all(item.execution_receipt is None for item in result.cycle.items)


def test_cooldown_denial_does_not_reserve_cash_needed_by_allowed_asset() -> None:
    plan = _plan({}, {"BTCUSDT": 0.40, "ETHUSDT": 0.40})
    risk = _risk_receipt(
        trades=(
            ClosedTrade("BTCUSDT", TS - 30, 1.0, 0.001, "TARGET"),
        )
    )
    result = _run(plan, risk, {}, cash=500.0)
    by_asset = {item.asset_id: item for item in result.cycle.items}

    assert by_asset["BTCUSDT"].risk_receipt.allowed is False
    assert by_asset["BTCUSDT"].new_risk_cash_reserved_usd == pytest.approx(0.0)
    assert by_asset["ETHUSDT"].risk_receipt.allowed is True
    assert by_asset["ETHUSDT"].new_risk_cash_reserved_usd == pytest.approx(400.0)
    assert result.cycle.remaining_unreserved_cash_usd == pytest.approx(100.0)


def test_expired_asset_cooldown_allows_new_risk_again() -> None:
    plan = _plan({}, {"BTCUSDT": 0.20})
    governor = OperationalRiskGovernor(_policy(asset_cooldown_seconds=300))
    risk = governor.evaluate(
        now=TS,
        equity_points=(
            EquityPoint(TS - 600, 1000.0),
            EquityPoint(TS, 1000.0),
        ),
        closed_trades=(
            ClosedTrade("BTCUSDT", TS - 301, 1.0, 0.001, "TARGET"),
        ),
        health_events=(),
        market_data_timestamp=TS,
    )
    result = _run(plan, risk, {})

    assert result.blocked_new_risk_assets == ()
    assert result.cycle.items[0].risk_receipt.allowed is True
    assert result.cycle.items[0].execution_receipt is not None


def test_governed_execution_records_governor_identity_and_is_deterministic() -> None:
    plan = _plan({}, {"BTCUSDT": 0.20})
    risk = _risk_receipt()
    kwargs = dict(
        plan=plan,
        operational_risk=risk,
        equity_usd=1000.0,
        available_cash_usd=500.0,
        current_weights={},
        markets={"BTCUSDT": _market()},
        risk_limits_by_asset={
            "BTCUSDT": InstrumentRiskLimits(max_notional_per_order=500.0)
        },
    )
    first = run_governed_shadow_execution_cycle(**kwargs)
    second = run_governed_shadow_execution_cycle(**kwargs)

    assert first.operational_risk_receipt_id == risk.receipt_id
    assert first.policy_fingerprint == second.policy_fingerprint
    assert first.result_id == second.result_id
    assert first.cycle.to_dict() == second.cycle.to_dict()
    assert first.shadow_only is True
    assert first.live_execution is False
