from __future__ import annotations

from dataclasses import replace

import pytest

from brian2026.evidence_ledger import content_hash
from brian2026.phase46_execution_simulator import SimulatedExecutionReceipt
from brian2026.phase56_pretrade_risk_engine import PreTradeRiskReceipt
from brian2026.phase57_shadow_execution_cycle import (
    ShadowExecutionCycle,
    ShadowExecutionCycleItem,
)
from brian2026.phase61_stateful_paper_venue import (
    PaperCycleReceipt,
    PaperOrderOutcome,
    PaperVenue,
    PaperVenueConfig,
)
from brian2026.phase64_local_execution_projector import (
    LocalExecutionProjectionError,
    LocalExecutionProjector,
)


TS = 1_760_000_000.0


def _risk(
    asset: str,
    *,
    allowed: bool = True,
    reduce_only: bool = False,
    notional: float = 100.0,
) -> PreTradeRiskReceipt:
    return PreTradeRiskReceipt(
        action="ALLOW" if allowed else "DENY",
        trading_state="REDUCING" if reduce_only else "ACTIVE",
        asset_id=asset,
        requested_notional_usd=notional,
        reduce_only=reduce_only,
        reasons=() if allowed else ("fixture-denied",),
        projected_position_weight=None,
        checks=(("fixture", allowed),),
    )


def _execution(
    *,
    side: str,
    status: str = "FILLED",
    requested_base: float = 1.0,
    filled_base: float = 1.0,
    price: float = 100.0,
    ts_offset: float = 0.1,
) -> SimulatedExecutionReceipt:
    return SimulatedExecutionReceipt(
        status=status,
        side=side,
        order_type="MARKET",
        submit_timestamp=TS,
        venue_timestamp=TS + ts_offset,
        snapshot_timestamp=TS + ts_offset,
        requested_base=requested_base,
        filled_base=filled_base,
        fill_fraction=(
            0.0 if requested_base <= 0 else filled_base / requested_base
        ),
        average_fill_price=(
            None if status == "NO_FILL" else price
        ),
        best_reference_price=price,
        adverse_slippage_bps=0.0,
        levels_consumed=1 if filled_base > 0 else 0,
        slipped_one_tick=False,
        reason="phase64 fixture",
    )


def _item(
    asset: str,
    *,
    execution: SimulatedExecutionReceipt | None,
    allowed: bool = True,
    reduce_only: bool = False,
    kind: str = "OPEN",
    notional: float = 100.0,
) -> ShadowExecutionCycleItem:
    return ShadowExecutionCycleItem(
        instruction_kind=kind,
        asset_id=asset,
        risk_receipt=_risk(
            asset,
            allowed=allowed,
            reduce_only=reduce_only,
            notional=notional,
        ),
        execution_receipt=execution,
        pending_reversal=None,
        new_risk_cash_reserved_usd=0.0 if reduce_only or not allowed else notional,
        status="fixture",
    )


def _cycle(
    cycle_id: str,
    *items: ShadowExecutionCycleItem,
) -> ShadowExecutionCycle:
    reserved = sum(item.new_risk_cash_reserved_usd for item in items)
    return ShadowExecutionCycle(
        source_plan_id=f"plan-{cycle_id}",
        items=tuple(items),
        initial_available_cash_usd=2_000.0,
        reserved_new_risk_cash_usd=reserved,
        remaining_unreserved_cash_usd=max(0.0, 2_000.0 - reserved),
        denied_assets=tuple(
            sorted(item.asset_id for item in items if not item.risk_receipt.allowed)
        ),
        pending_reversal_assets=(),
        cycle_id=cycle_id,
    )


def _fills_by_id(venue: PaperVenue):
    return {fill.fill_id: fill for fill in venue.fills}


def _receipt_identity(receipt: PaperCycleReceipt) -> str:
    return content_hash({
        "schema_version": receipt.schema_version,
        "cycle_id": receipt.cycle_id,
        "cycle_hash": receipt.cycle_hash,
        "outcomes": [row.to_dict() for row in receipt.outcomes],
        "fill_ids": list(receipt.fill_ids),
        "cash_before_usd": receipt.cash_before_usd,
        "cash_after_usd": receipt.cash_after_usd,
        "state_version_before": receipt.state_version_before,
        "state_version_after": receipt.state_version_after,
    })


def test_projector_and_paper_venue_reconcile_when_same_fill_events_arrive() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=2_000.0,
            fee_bps=0.0,
        )
    )
    projector = LocalExecutionProjector("paper-acct")
    cycle = _cycle(
        "independent",
        _item(
            "BTCUSDT",
            execution=_execution(
                side="BUY",
                requested_base=1.0,
                filled_base=1.0,
                price=100.0,
            ),
        ),
    )
    paper = venue.apply_cycle(cycle)
    projection = projector.process_cycle(paper, _fills_by_id(venue))

    assert projection.fills_applied == 1
    assert projection.duplicate is False
    assert projector.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert projector.position("BTCUSDT").avg_entry_price == pytest.approx(100.0)

    reconciliation = venue.reconcile_against_local(
        projector.local_positions(tracked_assets=("BTCUSDT", "ETHUSDT")),
        tracked_assets=("BTCUSDT", "ETHUSDT"),
    )
    assert reconciliation.ready is True
    assert all(dict(reconciliation.checks).values())


def test_missing_fill_event_is_atomic_and_phase50_detects_divergence() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=2_000.0,
            fee_bps=0.0,
        )
    )
    projector = LocalExecutionProjector("paper-acct")
    cycle = _cycle(
        "missing-fill",
        _item(
            "BTCUSDT",
            execution=_execution(side="BUY", price=100.0),
        ),
    )
    paper = venue.apply_cycle(cycle)

    with pytest.raises(LocalExecutionProjectionError, match="missing local execution fill"):
        projector.process_cycle(paper, {})

    assert projector.projection_version == 0
    assert projector.orders == {}
    assert projector.positions == {}

    reconciliation = venue.reconcile_against_local(
        projector.local_positions(tracked_assets=("BTCUSDT",)),
        tracked_assets=("BTCUSDT",),
    )
    assert reconciliation.ready is False
    assert dict(reconciliation.checks)["positions_reconciled"] is False


def test_two_fill_cycle_validates_every_fill_before_any_local_mutation() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=2_000.0,
            fee_bps=0.0,
        )
    )
    projector = LocalExecutionProjector("paper-acct")
    cycle = _cycle(
        "atomic-two",
        _item("BTCUSDT", execution=_execution(side="BUY", price=100.0)),
        _item("ETHUSDT", execution=_execution(side="BUY", price=200.0)),
    )
    paper = venue.apply_cycle(cycle)
    fills = _fills_by_id(venue)
    one_fill_id = paper.fill_ids[0]
    partial_mapping = {one_fill_id: fills[one_fill_id]}

    with pytest.raises(LocalExecutionProjectionError, match="missing local execution fill"):
        projector.process_cycle(paper, partial_mapping)

    assert projector.projection_version == 0
    assert projector.orders == {}
    assert projector.positions == {}


def test_no_fill_and_risk_denied_orders_are_projected_without_position_changes() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=2_000.0,
            fee_bps=0.0,
        )
    )
    projector = LocalExecutionProjector("paper-acct")
    cycle = _cycle(
        "nonfills",
        _item(
            "BTCUSDT",
            execution=_execution(
                side="BUY",
                status="NO_FILL",
                requested_base=1.0,
                filled_base=0.0,
                price=100.0,
            ),
        ),
        _item(
            "ETHUSDT",
            execution=None,
            allowed=False,
        ),
    )
    paper = venue.apply_cycle(cycle)
    result = projector.process_cycle(paper, _fills_by_id(venue))

    assert result.orders_projected == 2
    assert result.fills_applied == 0
    assert projector.orders[paper.outcomes[0].paper_order_id].status == "ACKNOWLEDGED_NO_FILL"
    assert projector.orders[paper.outcomes[1].paper_order_id].status == "RISK_DENIED"
    assert projector.local_positions() == {}


def test_local_projector_independently_handles_reduce_and_direction_reversal() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=2_000.0,
            fee_bps=0.0,
        )
    )
    projector = LocalExecutionProjector("paper-acct")

    open_cycle = _cycle(
        "open",
        _item(
            "BTCUSDT",
            execution=_execution(side="BUY", requested_base=1.0, filled_base=1.0, price=100.0),
        ),
    )
    open_receipt = venue.apply_cycle(open_cycle)
    projector.process_cycle(open_receipt, _fills_by_id(venue))

    flip_cycle = _cycle(
        "flip",
        _item(
            "BTCUSDT",
            execution=_execution(side="SELL", requested_base=2.0, filled_base=2.0, price=90.0),
            notional=180.0,
        ),
    )
    flip_receipt = venue.apply_cycle(flip_cycle)
    projector.process_cycle(flip_receipt, _fills_by_id(venue))

    local = projector.position("BTCUSDT")
    paper = venue.position("BTCUSDT")
    assert local.quantity == pytest.approx(-1.0)
    assert local.avg_entry_price == pytest.approx(90.0)
    assert local.realized_pnl_quote == pytest.approx(-10.0)
    assert local.quantity == pytest.approx(paper.quantity)
    assert local.avg_entry_price == pytest.approx(paper.avg_entry_price)

    reconciliation = venue.reconcile_against_local(
        projector.local_positions(tracked_assets=("BTCUSDT",)),
        tracked_assets=("BTCUSDT",),
    )
    assert reconciliation.ready is True


def test_duplicate_cycle_is_idempotent_and_does_not_apply_fill_twice() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=2_000.0,
            fee_bps=0.0,
        )
    )
    projector = LocalExecutionProjector("paper-acct")
    cycle = _cycle(
        "idem",
        _item("BTCUSDT", execution=_execution(side="BUY", price=100.0)),
    )
    paper = venue.apply_cycle(cycle)
    fills = _fills_by_id(venue)

    first = projector.process_cycle(paper, fills)
    duplicate = projector.process_cycle(paper, fills)

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert first.projection_hash == duplicate.projection_hash
    assert projector.projection_version == 1
    assert projector.position("BTCUSDT").quantity == pytest.approx(1.0)


def test_tampered_receipt_is_rejected_before_projection() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=2_000.0,
            fee_bps=0.0,
        )
    )
    projector = LocalExecutionProjector("paper-acct")
    cycle = _cycle(
        "tamper-receipt",
        _item("BTCUSDT", execution=_execution(side="BUY", price=100.0)),
    )
    paper = venue.apply_cycle(cycle)
    tampered = replace(paper, cash_after_usd=paper.cash_after_usd + 1.0)

    with pytest.raises(LocalExecutionProjectionError, match="receipt content hash mismatch"):
        projector.process_cycle(tampered, _fills_by_id(venue))
    assert projector.projection_version == 0


def test_outcome_average_price_must_reconcile_fill_events() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=2_000.0,
            fee_bps=0.0,
        )
    )
    projector = LocalExecutionProjector("paper-acct")
    cycle = _cycle(
        "bad-average",
        _item("BTCUSDT", execution=_execution(side="BUY", price=100.0)),
    )
    paper = venue.apply_cycle(cycle)
    outcome = paper.outcomes[0]
    bad_outcome = replace(outcome, average_fill_price=101.0)
    forged = PaperCycleReceipt(
        cycle_id=paper.cycle_id,
        cycle_hash=paper.cycle_hash,
        outcomes=(bad_outcome,),
        fill_ids=paper.fill_ids,
        cash_before_usd=paper.cash_before_usd,
        cash_after_usd=paper.cash_after_usd,
        state_version_before=paper.state_version_before,
        state_version_after=paper.state_version_after,
        receipt_id="placeholder",
    )
    forged = replace(forged, receipt_id=_receipt_identity(forged))

    with pytest.raises(LocalExecutionProjectionError, match="average_fill_price"):
        projector.process_cycle(forged, _fills_by_id(venue))
    assert projector.positions == {}


def test_fill_content_hash_is_verified_independently_from_receipt() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=2_000.0,
            fee_bps=0.0,
        )
    )
    projector = LocalExecutionProjector("paper-acct")
    cycle = _cycle(
        "bad-fill",
        _item("BTCUSDT", execution=_execution(side="BUY", price=100.0)),
    )
    paper = venue.apply_cycle(cycle)
    original = venue.fills[0]
    forged = replace(original, price=101.0)

    with pytest.raises(LocalExecutionProjectionError, match="fill content hash mismatch"):
        projector.process_cycle(
            paper,
            {original.fill_id: forged},
        )
    assert projector.projection_version == 0


def test_projector_manifest_is_content_addressed_shadow_state() -> None:
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=2_000.0,
            fee_bps=0.0,
        )
    )
    projector = LocalExecutionProjector("paper-acct")
    cycle = _cycle(
        "manifest",
        _item("BTCUSDT", execution=_execution(side="BUY", price=100.0)),
    )
    paper = venue.apply_cycle(cycle)
    projector.process_cycle(paper, _fills_by_id(venue))

    manifest = projector.manifest()
    assert manifest["account_id"] == "paper-acct"
    assert manifest["projection_version"] == 1
    assert manifest["shadow_only"] is True
    assert manifest["live_execution"] is False
    assert isinstance(manifest["projection_hash"], str)
    assert len(manifest["projection_hash"]) == 64
