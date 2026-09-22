from __future__ import annotations

import pytest

from brian2026.phase45_execution_contract import (
    TradeIntent,
    TripleBarrierPolicy,
    create_position_executor_action,
)
from brian2026.phase46_execution_simulator import (
    InflightOrderRegistry,
    LiquidityLevel,
    OrderBookSnapshot,
    ProbabilisticFillModel,
    StaticLatencyModel,
    simulate_create_action,
    simulate_order,
)


def _book(timestamp: float = 10.0) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=timestamp,
        bids=(
            LiquidityLevel(99.0, 1.0),
            LiquidityLevel(98.0, 2.0),
            LiquidityLevel(97.0, 10.0),
        ),
        asks=(
            LiquidityLevel(100.0, 1.0),
            LiquidityLevel(101.0, 1.0),
            LiquidityLevel(102.0, 10.0),
        ),
    )


def test_static_latency_adds_base_to_each_operation() -> None:
    latency = StaticLatencyModel(
        base_latency_ms=100.0,
        insert_latency_ms=200.0,
        update_latency_ms=300.0,
        delete_latency_ms=400.0,
    )
    assert latency.effective_insert_ms == pytest.approx(300.0)
    assert latency.effective_update_ms == pytest.approx(400.0)
    assert latency.effective_delete_ms == pytest.approx(500.0)


def test_inflight_registry_first_receipt_releases_duplicate_submit() -> None:
    registry = InflightOrderRegistry()
    registry.submit(("order-1",))
    registry.submit(("order-1",))
    assert registry.contains("order-1")
    registry.receipt(("order-1",))
    assert not registry.contains("order-1")
    registry.receipt(("order-1",))
    assert not registry.contains("order-1")


def test_tiered_market_fill_walks_visible_liquidity_and_computes_vwap() -> None:
    receipt = simulate_order(
        side="BUY",
        order_type="MARKET",
        requested_base=3.0,
        submit_timestamp=9.0,
        snapshots=(_book(),),
        tick_size=1.0,
    )
    assert receipt.status == "FILLED"
    assert receipt.filled_base == pytest.approx(3.0)
    assert receipt.levels_consumed == 3
    assert receipt.average_fill_price == pytest.approx((100.0 + 101.0 + 102.0) / 3.0)
    assert receipt.best_reference_price == pytest.approx(100.0)
    assert receipt.adverse_slippage_bps > 0.0


def test_insufficient_visible_depth_produces_partial_fill() -> None:
    shallow = OrderBookSnapshot(
        timestamp=10.0,
        bids=(LiquidityLevel(99.0, 1.0),),
        asks=(LiquidityLevel(100.0, 1.0), LiquidityLevel(101.0, 1.0)),
    )
    receipt = simulate_order(
        side="BUY",
        order_type="MARKET",
        requested_base=3.0,
        submit_timestamp=9.0,
        snapshots=(shallow,),
        tick_size=1.0,
    )
    assert receipt.status == "PARTIAL_FILL"
    assert receipt.filled_base == pytest.approx(2.0)
    assert receipt.fill_fraction == pytest.approx(2 / 3)


def test_probabilistic_limit_fill_and_one_tick_slippage_are_seeded_contracts() -> None:
    no_fill = simulate_order(
        side="BUY",
        order_type="LIMIT",
        requested_base=1.0,
        submit_timestamp=9.0,
        snapshots=(_book(),),
        limit_price=101.0,
        tick_size=1.0,
        fill_model=ProbabilisticFillModel(prob_fill_on_limit=0.0, prob_slippage=0.0, seed=1),
    )
    assert no_fill.status == "NO_FILL"

    slipped = simulate_order(
        side="BUY",
        order_type="MARKET",
        requested_base=1.0,
        submit_timestamp=9.0,
        snapshots=(_book(),),
        tick_size=1.0,
        fill_model=ProbabilisticFillModel(prob_fill_on_limit=1.0, prob_slippage=1.0, seed=1),
    )
    assert slipped.status == "FILLED"
    assert slipped.slipped_one_tick is True
    assert slipped.average_fill_price == pytest.approx(101.0)


def test_latency_selects_first_snapshot_after_venue_arrival() -> None:
    early = OrderBookSnapshot(
        timestamp=10.10,
        bids=(LiquidityLevel(99.0, 10.0),),
        asks=(LiquidityLevel(100.0, 10.0),),
    )
    late = OrderBookSnapshot(
        timestamp=10.31,
        bids=(LiquidityLevel(109.0, 10.0),),
        asks=(LiquidityLevel(110.0, 10.0),),
    )
    receipt = simulate_order(
        side="BUY",
        order_type="MARKET",
        requested_base=1.0,
        submit_timestamp=10.0,
        snapshots=(early, late),
        latency=StaticLatencyModel(base_latency_ms=100.0, insert_latency_ms=200.0),
        tick_size=1.0,
    )
    assert receipt.venue_timestamp == pytest.approx(10.30)
    assert receipt.snapshot_timestamp == pytest.approx(10.31)
    assert receipt.average_fill_price == pytest.approx(110.0)


def test_intent_slippage_budget_vetoes_projected_bad_fill_before_applying_it() -> None:
    receipt = simulate_order(
        side="BUY",
        order_type="MARKET",
        requested_base=3.0,
        submit_timestamp=9.0,
        snapshots=(_book(),),
        tick_size=1.0,
        max_slippage_bps=10.0,
    )
    assert receipt.status == "VETO_SLIPPAGE"
    assert receipt.filled_base == 0.0
    assert receipt.average_fill_price is not None
    assert receipt.adverse_slippage_bps is not None
    assert receipt.adverse_slippage_bps > 10.0


def test_phase45_create_action_can_be_replayed_without_exchange_transport() -> None:
    intent = TradeIntent(
        intent_id="btc-market",
        asset_id="BTCUSDT",
        direction=1,
        target_weight=0.2,
        expected_edge_bps=30.0,
        confidence=0.8,
        max_slippage_bps=250.0,
        created_at=9.0,
        ttl_seconds=10,
        evidence_ids=("ev-1",),
    )
    action = create_position_executor_action(
        intent,
        equity_usd=500.0,
        reference_price=100.0,
        barrier=TripleBarrierPolicy(open_order_type="MARKET"),
    )
    receipt = simulate_create_action(
        action,
        intent,
        snapshots=(_book(),),
        tick_size=1.0,
    )
    assert receipt.status == "FILLED"
    assert receipt.requested_base == pytest.approx(1.0)
    assert receipt.shadow_only is True
    assert receipt.live_execution is False
