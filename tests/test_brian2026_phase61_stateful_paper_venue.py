from __future__ import annotations

import pytest

from brian2026.evidence_ledger import content_hash
from brian2026.phase50_execution_reconciliation import LocalPositionState
from brian2026.phase56_pretrade_risk_engine import PreTradeRiskReceipt
from brian2026.phase46_execution_simulator import SimulatedExecutionReceipt
from brian2026.phase57_shadow_execution_cycle import (
    ShadowExecutionCycle,
    ShadowExecutionCycleItem,
)
from brian2026.phase60_shadow_state_ledger import (
    ShadowAccountState,
    ShadowStateLedger,
)
from brian2026.phase61_stateful_paper_venue import (
    PaperVenue,
    PaperVenueConfig,
    PaperVenueConflictError,
)


TS = 1_760_000_000.0


def _risk(
    asset: str,
    notional: float,
    *,
    allowed: bool = True,
    reduce_only: bool = False,
) -> PreTradeRiskReceipt:
    checks = (("fixture_check", allowed),)
    return PreTradeRiskReceipt(
        action="ALLOW" if allowed else "DENY",
        trading_state="REDUCING" if reduce_only else "ACTIVE",
        asset_id=asset,
        requested_notional_usd=notional,
        reduce_only=reduce_only,
        reasons=() if allowed else ("fixture_check",),
        projected_position_weight=None,
        checks=checks,
    )


def _execution(
    *,
    side: str,
    status: str = "FILLED",
    requested_base: float = 1.0,
    filled_base: float = 1.0,
    price: float = 100.0,
    venue_timestamp: float = TS + 0.1,
) -> SimulatedExecutionReceipt:
    fill_fraction = 0.0 if filled_base <= 0 else filled_base / requested_base
    return SimulatedExecutionReceipt(
        status=status,
        side=side,
        order_type="MARKET",
        submit_timestamp=TS,
        venue_timestamp=venue_timestamp,
        snapshot_timestamp=venue_timestamp,
        requested_base=requested_base,
        filled_base=filled_base,
        fill_fraction=fill_fraction,
        average_fill_price=price if filled_base > 0 or status == "VETO_SLIPPAGE" else None,
        best_reference_price=price,
        adverse_slippage_bps=0.0,
        levels_consumed=1 if filled_base > 0 else 0,
        slipped_one_tick=False,
        reason="phase61 fixture",
    )


def _item(
    asset: str,
    execution: SimulatedExecutionReceipt | None,
    *,
    reduce_only: bool = False,
    allowed: bool = True,
    instruction_kind: str = "OPEN",
    notional: float | None = None,
) -> ShadowExecutionCycleItem:
    requested_notional = (
        float(notional)
        if notional is not None
        else (
            0.0
            if execution is None
            else execution.requested_base * (execution.average_fill_price or execution.best_reference_price or 0.0)
        )
    )
    risk = _risk(
        asset,
        requested_notional,
        allowed=allowed,
        reduce_only=reduce_only,
    )
    return ShadowExecutionCycleItem(
        instruction_kind=instruction_kind,
        asset_id=asset,
        risk_receipt=risk,
        execution_receipt=execution,
        pending_reversal=None,
        new_risk_cash_reserved_usd=0.0 if reduce_only else requested_notional,
        status="fixture",
    )


def _cycle(cycle_id: str, *items: ShadowExecutionCycleItem) -> ShadowExecutionCycle:
    return ShadowExecutionCycle(
        source_plan_id=f"plan-{cycle_id}",
        items=tuple(items),
        initial_available_cash_usd=1000.0,
        reserved_new_risk_cash_usd=sum(
            item.new_risk_cash_reserved_usd for item in items
        ),
        remaining_unreserved_cash_usd=max(
            0.0,
            1000.0 - sum(item.new_risk_cash_reserved_usd for item in items),
        ),
        denied_assets=tuple(
            sorted(item.asset_id for item in items if not item.risk_receipt.allowed)
        ),
        pending_reversal_assets=(),
        cycle_id=cycle_id,
    )


def test_long_fill_updates_cash_position_average_and_fee() -> None:
    venue = PaperVenue(PaperVenueConfig(account_id="paper-acct", starting_cash_usd=1000.0, fee_bps=10.0))
    cycle = _cycle(
        "c1",
        _item("BTCUSDT", _execution(side="BUY", requested_base=2.0, filled_base=2.0, price=100.0), notional=200.0),
    )
    receipt = venue.apply_cycle(cycle)
    assert receipt.outcomes[0].status == "FILLED"
    assert receipt.outcomes[0].acknowledged is True
    assert venue.cash_usd == pytest.approx(799.8)
    position = venue.position("BTCUSDT")
    assert position.quantity == pytest.approx(2.0)
    assert position.avg_entry_price == pytest.approx(100.0)
    assert position.realized_pnl_quote == pytest.approx(0.0)
    assert len(position.source_fill_ids) == 1


def test_same_side_increase_uses_weighted_average_and_partial_fill_quantity() -> None:
    venue = PaperVenue(PaperVenueConfig(account_id="paper-acct", starting_cash_usd=1000.0, fee_bps=0.0))
    venue.apply_cycle(
        _cycle(
            "c1",
            _item("BTCUSDT", _execution(side="BUY", requested_base=2.0, filled_base=2.0, price=100.0), notional=200.0),
        )
    )
    second = venue.apply_cycle(
        _cycle(
            "c2",
            _item(
                "BTCUSDT",
                _execution(
                    side="BUY",
                    status="PARTIAL_FILL",
                    requested_base=2.0,
                    filled_base=1.0,
                    price=110.0,
                ),
                notional=220.0,
            ),
        )
    )
    assert second.outcomes[0].status == "PARTIAL_FILL"
    assert second.outcomes[0].fill_fraction == pytest.approx(0.5)
    position = venue.position("BTCUSDT")
    assert position.quantity == pytest.approx(3.0)
    assert position.avg_entry_price == pytest.approx((2 * 100 + 1 * 110) / 3)


def test_reduction_realizes_pnl_and_preserves_remaining_entry_average() -> None:
    venue = PaperVenue(PaperVenueConfig(account_id="paper-acct", starting_cash_usd=1000.0, fee_bps=0.0))
    venue.apply_cycle(
        _cycle(
            "open",
            _item("BTCUSDT", _execution(side="BUY", requested_base=2.0, filled_base=2.0, price=100.0), notional=200.0),
        )
    )
    venue.apply_cycle(
        _cycle(
            "reduce",
            _item(
                "BTCUSDT",
                _execution(side="SELL", requested_base=1.0, filled_base=1.0, price=120.0),
                reduce_only=True,
                instruction_kind="REDUCE",
                notional=120.0,
            ),
        )
    )
    position = venue.position("BTCUSDT")
    assert position.quantity == pytest.approx(1.0)
    assert position.avg_entry_price == pytest.approx(100.0)
    assert position.realized_pnl_quote == pytest.approx(20.0)
    assert venue.cash_usd == pytest.approx(920.0)


def test_reversal_closes_old_side_then_opens_remainder_at_new_fill_price() -> None:
    venue = PaperVenue(PaperVenueConfig(account_id="paper-acct", starting_cash_usd=1000.0, fee_bps=0.0))
    venue.apply_cycle(
        _cycle(
            "open",
            _item("BTCUSDT", _execution(side="BUY", requested_base=1.0, filled_base=1.0, price=100.0), notional=100.0),
        )
    )
    venue.apply_cycle(
        _cycle(
            "flip",
            _item("BTCUSDT", _execution(side="SELL", requested_base=2.0, filled_base=2.0, price=90.0), notional=180.0),
        )
    )
    position = venue.position("BTCUSDT")
    assert position.quantity == pytest.approx(-1.0)
    assert position.avg_entry_price == pytest.approx(90.0)
    assert position.realized_pnl_quote == pytest.approx(-10.0)
    assert venue.cash_usd == pytest.approx(1080.0)


def test_new_buy_rejects_if_cash_cannot_fund_fill_plus_fee() -> None:
    venue = PaperVenue(PaperVenueConfig(account_id="paper-acct", starting_cash_usd=50.0, fee_bps=10.0))
    receipt = venue.apply_cycle(
        _cycle(
            "too-expensive",
            _item("BTCUSDT", _execution(side="BUY", requested_base=1.0, filled_base=1.0, price=100.0), notional=100.0),
        )
    )
    outcome = receipt.outcomes[0]
    assert outcome.status == "VENUE_REJECTED_BALANCE"
    assert outcome.acknowledged is False
    assert outcome.filled_base == pytest.approx(0.0)
    assert venue.cash_usd == pytest.approx(50.0)
    assert venue.position("BTCUSDT").quantity == pytest.approx(0.0)


def test_reduce_only_buy_can_close_short_even_when_cash_is_insufficient() -> None:
    venue = PaperVenue(PaperVenueConfig(account_id="paper-acct", starting_cash_usd=50.0, fee_bps=0.0))
    venue.apply_cycle(
        _cycle(
            "short-open",
            _item("BTCUSDT", _execution(side="SELL", requested_base=2.0, filled_base=2.0, price=100.0), notional=200.0),
        )
    )
    assert venue.cash_usd == pytest.approx(250.0)
    receipt = venue.apply_cycle(
        _cycle(
            "short-close",
            _item(
                "BTCUSDT",
                _execution(side="BUY", requested_base=2.0, filled_base=2.0, price=200.0),
                reduce_only=True,
                instruction_kind="CLOSE",
                notional=400.0,
            ),
        )
    )
    assert receipt.outcomes[0].status == "FILLED"
    assert venue.position("BTCUSDT").quantity == pytest.approx(0.0)
    assert venue.position("BTCUSDT").realized_pnl_quote == pytest.approx(-200.0)
    assert venue.cash_usd == pytest.approx(-150.0)


def test_risk_denied_and_local_veto_never_reach_paper_fill_state() -> None:
    venue = PaperVenue(PaperVenueConfig(account_id="paper-acct", starting_cash_usd=1000.0))
    denied = _item(
        "BTCUSDT",
        None,
        allowed=False,
        notional=100.0,
    )
    veto = _item(
        "ETHUSDT",
        _execution(
            side="BUY",
            status="VETO_SLIPPAGE",
            requested_base=1.0,
            filled_base=0.0,
            price=100.0,
        ),
        notional=100.0,
    )
    receipt = venue.apply_cycle(_cycle("gates", denied, veto))
    assert [row.status for row in receipt.outcomes] == ["RISK_DENIED", "LOCAL_VETO"]
    assert receipt.fill_ids == ()
    assert venue.cash_usd == pytest.approx(1000.0)


def test_cycle_application_is_idempotent_and_conflicting_reuse_fails() -> None:
    venue = PaperVenue(PaperVenueConfig(account_id="paper-acct", starting_cash_usd=1000.0, fee_bps=0.0))
    cycle = _cycle(
        "idem",
        _item("BTCUSDT", _execution(side="BUY", price=100.0), notional=100.0),
    )
    first = venue.apply_cycle(cycle)
    cash_after = venue.cash_usd
    duplicate = venue.apply_cycle(cycle)
    assert duplicate == first
    assert venue.cash_usd == pytest.approx(cash_after)
    assert venue.state_version == 1

    conflicting = _cycle(
        "idem",
        _item("BTCUSDT", _execution(side="BUY", price=101.0), notional=101.0),
    )
    with pytest.raises(PaperVenueConflictError, match="different shadow execution evidence"):
        venue.apply_cycle(conflicting)


def test_explicit_flat_position_report_is_generated_for_tracked_asset() -> None:
    venue = PaperVenue(PaperVenueConfig(account_id="paper-acct"))
    reports = venue.generate_position_reports(("BTCUSDT", "ETHUSDT"))
    assert set(reports) == {"BTCUSDT", "ETHUSDT"}
    assert reports["BTCUSDT"].explicit is True
    assert reports["BTCUSDT"].quantity == pytest.approx(0.0)
    assert reports["BTCUSDT"].avg_entry_price is None


def test_phase61_to_phase50_to_phase60_closes_authoritative_paper_loop() -> None:
    venue = PaperVenue(PaperVenueConfig(account_id="paper-acct", starting_cash_usd=1000.0, fee_bps=0.0))
    cycle = _cycle(
        "paper-loop",
        _item("BTCUSDT", _execution(side="BUY", requested_base=1.0, filled_base=1.0, price=100.0), notional=100.0),
    )
    venue.apply_cycle(cycle)
    position = venue.position("BTCUSDT")
    local = {
        "BTCUSDT": LocalPositionState(
            account_id="paper-acct",
            asset_id="BTCUSDT",
            quantity=position.quantity,
            avg_entry_price=position.avg_entry_price,
            source_fill_ids=position.source_fill_ids,
        )
    }
    reconciliation = venue.reconcile_against_local(
        local,
        tracked_assets=("BTCUSDT", "ETHUSDT"),
    )
    assert reconciliation.ready is True
    assert all(dict(reconciliation.checks).values())

    state = venue.build_reconciled_state(
        reconciliation,
        marks={"BTCUSDT": 110.0},
        observed_at=TS + 10,
        source_ref="paper-loop-account-report",
    )
    assert state.source_kind == "RECONCILED_PAPER"
    assert state.reconciliation_hash == content_hash(reconciliation.to_dict())
    assert state.equity_usd == pytest.approx(1010.0)
    assert state.position_weights[0][0] == "BTCUSDT"
    assert state.position_weights[0][1] == pytest.approx(110.0 / 1010.0)

    genesis = ShadowAccountState(
        account_id="paper-acct",
        observed_at=TS,
        equity_usd=1000.0,
        available_cash_usd=1000.0,
        position_weights=(),
        covered_assets=("BTCUSDT", "ETHUSDT"),
        source_kind="GENESIS",
        source_ref="paper-loop-genesis",
    )
    ledger = ShadowStateLedger(genesis)
    ledger.append_cycle(cycle)
    ledger.commit_reconciled_state(cycle.cycle_id, reconciliation, state)
    assert ledger.head_state.state_id == state.state_id
    assert ledger.verify_integrity() is True


def test_mismatched_local_mirror_blocks_reconciled_state_creation() -> None:
    venue = PaperVenue(PaperVenueConfig(account_id="paper-acct", starting_cash_usd=1000.0, fee_bps=0.0))
    venue.apply_cycle(
        _cycle(
            "mismatch",
            _item("BTCUSDT", _execution(side="BUY", price=100.0), notional=100.0),
        )
    )
    local = {
        "BTCUSDT": LocalPositionState(
            account_id="paper-acct",
            asset_id="BTCUSDT",
            quantity=0.5,
            avg_entry_price=100.0,
        )
    }
    reconciliation = venue.reconcile_against_local(
        local,
        tracked_assets=("BTCUSDT",),
    )
    assert reconciliation.ready is False
    with pytest.raises(PaperVenueConflictError, match="must pass"):
        venue.build_reconciled_state(
            reconciliation,
            marks={"BTCUSDT": 100.0},
            observed_at=TS + 10,
            source_ref="blocked",
        )


def test_open_position_requires_mark_price_for_authoritative_snapshot() -> None:
    venue = PaperVenue(PaperVenueConfig(account_id="paper-acct", starting_cash_usd=1000.0, fee_bps=0.0))
    venue.apply_cycle(
        _cycle(
            "mark",
            _item("BTCUSDT", _execution(side="BUY", price=100.0), notional=100.0),
        )
    )
    position = venue.position("BTCUSDT")
    reconciliation = venue.reconcile_against_local(
        {
            "BTCUSDT": LocalPositionState(
                account_id="paper-acct",
                asset_id="BTCUSDT",
                quantity=position.quantity,
                avg_entry_price=position.avg_entry_price,
                source_fill_ids=position.source_fill_ids,
            )
        },
        tracked_assets=("BTCUSDT",),
    )
    with pytest.raises(KeyError, match="missing mark price"):
        venue.build_reconciled_state(
            reconciliation,
            marks={},
            observed_at=TS + 10,
            source_ref="missing-mark",
        )
