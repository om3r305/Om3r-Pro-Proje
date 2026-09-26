from __future__ import annotations

import pytest

from brian2026.phase46_execution_simulator import LiquidityLevel, OrderBookSnapshot
from brian2026.phase53_turnover_rebalance import TurnoverConfig, plan_turnover_constrained_rebalance
from brian2026.phase55_rebalance_execution_intents import compile_rebalance_execution_plan
from brian2026.phase56_pretrade_risk_engine import InstrumentRiskLimits
from brian2026.phase57_shadow_execution_cycle import (
    ExecutionMarketInput,
    run_shadow_execution_cycle,
)


TS = 1_760_000_000.0


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


def test_same_cycle_new_risk_cannot_spend_same_cash_twice() -> None:
    plan = _plan({}, {"BTC": 0.40, "ETH": 0.40})
    cycle = run_shadow_execution_cycle(
        plan,
        equity_usd=1000.0,
        available_cash_usd=500.0,
        current_weights={},
        markets={"BTC": _market(), "ETH": _market()},
        risk_limits_by_asset={
            "BTC": InstrumentRiskLimits(max_notional_per_order=500.0),
            "ETH": InstrumentRiskLimits(max_notional_per_order=500.0),
        },
    )
    by_asset = {item.asset_id: item for item in cycle.items}
    assert by_asset["BTC"].risk_receipt.allowed is True
    assert by_asset["BTC"].new_risk_cash_reserved_usd == pytest.approx(400.0)
    assert by_asset["BTC"].execution_receipt is not None
    assert by_asset["BTC"].execution_receipt.status == "FILLED"

    assert by_asset["ETH"].risk_receipt.allowed is False
    assert "cash_available" in by_asset["ETH"].risk_receipt.reasons
    assert by_asset["ETH"].execution_receipt is None
    assert cycle.reserved_new_risk_cash_usd == pytest.approx(400.0)
    assert cycle.remaining_unreserved_cash_usd == pytest.approx(100.0)
    assert cycle.denied_assets == ("ETH",)


def test_reducing_state_allows_reduction_but_blocks_new_risk() -> None:
    plan = _plan({"BTC": 0.30}, {"BTC": 0.10, "ETH": 0.20})
    cycle = run_shadow_execution_cycle(
        plan,
        equity_usd=1000.0,
        available_cash_usd=500.0,
        current_weights={"BTC": 0.30},
        markets={"BTC": _market(), "ETH": _market()},
        risk_limits_by_asset={},
        trading_state="REDUCING",
    )
    by_asset = {item.asset_id: item for item in cycle.items}
    assert by_asset["BTC"].instruction_kind == "REDUCE"
    assert by_asset["BTC"].risk_receipt.allowed is True
    assert by_asset["BTC"].execution_receipt is not None
    assert by_asset["BTC"].execution_receipt.status == "FILLED"

    assert by_asset["ETH"].instruction_kind == "OPEN"
    assert by_asset["ETH"].risk_receipt.allowed is False
    assert by_asset["ETH"].execution_receipt is None


def test_reversal_cycle_executes_only_close_leg_and_keeps_open_pending() -> None:
    plan = _plan({"BTC": 0.30}, {"BTC": -0.20})
    cycle = run_shadow_execution_cycle(
        plan,
        equity_usd=1000.0,
        available_cash_usd=500.0,
        current_weights={"BTC": 0.30},
        markets={"BTC": _market()},
        risk_limits_by_asset={},
    )
    assert len(cycle.items) == 1
    item = cycle.items[0]
    assert item.instruction_kind == "REVERSAL_CLOSE"
    assert item.risk_receipt.allowed is True
    assert item.execution_receipt is not None
    assert item.execution_receipt.side == "SELL"
    assert item.pending_reversal is not None
    assert item.pending_reversal.automatic_release is False
    assert cycle.pending_reversal_assets == ("BTC",)
    assert cycle.reserved_new_risk_cash_usd == pytest.approx(0.0)


def test_simulated_reduction_does_not_credit_cash_for_later_new_risk() -> None:
    plan = _plan({"BTC": 0.30}, {"BTC": 0.10, "ETH": 0.10})
    cycle = run_shadow_execution_cycle(
        plan,
        equity_usd=1000.0,
        available_cash_usd=50.0,
        current_weights={"BTC": 0.30},
        markets={"BTC": _market(), "ETH": _market()},
        risk_limits_by_asset={},
    )
    by_asset = {item.asset_id: item for item in cycle.items}
    assert by_asset["BTC"].risk_receipt.allowed is True
    assert by_asset["BTC"].execution_receipt is not None

    # The simulated BTC sale is not authoritative cash reconciliation, so ETH
    # still sees only the original $50 and cannot spend imaginary proceeds.
    assert by_asset["ETH"].risk_receipt.allowed is False
    assert "cash_available" in by_asset["ETH"].risk_receipt.reasons
    assert cycle.initial_available_cash_usd == pytest.approx(50.0)
    assert cycle.account_state_mutated is False


def test_halted_state_denies_all_cycle_submissions() -> None:
    plan = _plan({"BTC": 0.30}, {"BTC": 0.10, "ETH": 0.10})
    cycle = run_shadow_execution_cycle(
        plan,
        equity_usd=1000.0,
        available_cash_usd=500.0,
        current_weights={"BTC": 0.30},
        markets={"BTC": _market(), "ETH": _market()},
        risk_limits_by_asset={},
        trading_state="HALTED",
    )
    assert cycle.items
    assert all(item.risk_receipt.allowed is False for item in cycle.items)
    assert all(item.execution_receipt is None for item in cycle.items)
    assert set(cycle.denied_assets) == {"BTC", "ETH"}


def test_missing_market_input_fails_before_execution_simulation() -> None:
    plan = _plan({}, {"BTC": 0.10})
    with pytest.raises(KeyError, match="missing execution market input"):
        run_shadow_execution_cycle(
            plan,
            equity_usd=1000.0,
            available_cash_usd=500.0,
            current_weights={},
            markets={},
            risk_limits_by_asset={},
        )


def test_cycle_is_deterministic_with_default_fill_model() -> None:
    plan = _plan({}, {"BTC": 0.10})
    kwargs = dict(
        plan=plan,
        equity_usd=1000.0,
        available_cash_usd=500.0,
        current_weights={},
        markets={"BTC": _market()},
        risk_limits_by_asset={},
    )
    first = run_shadow_execution_cycle(**kwargs)
    second = run_shadow_execution_cycle(**kwargs)
    assert first.cycle_id == second.cycle_id
    assert first.to_dict() == second.to_dict()
    assert first.shadow_only is True
    assert first.live_execution is False
