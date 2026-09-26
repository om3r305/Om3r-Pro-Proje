from __future__ import annotations

import pytest

from brian2026.phase46_execution_simulator import SimulatedExecutionReceipt
from brian2026.phase56_pretrade_risk_engine import PreTradeRiskReceipt
from brian2026.phase57_shadow_execution_cycle import (
    ShadowExecutionCycle,
    ShadowExecutionCycleItem,
)
from brian2026.phase60_shadow_state_ledger import ShadowAccountState, ShadowStateLedger
from brian2026.phase61_stateful_paper_venue import PaperVenue, PaperVenueConfig
from brian2026.phase63_crash_recovery import create_runtime_checkpoint
from brian2026.phase64_local_execution_projector import LocalExecutionProjector
from brian2026.phase65_event_sourced_local_recovery import (
    audit_partial_local_replay,
    replay_local_execution_history,
    restore_runtime_with_local_projection,
)


TS = 1_760_000_000.0


def _cycle(
    cycle_id: str,
    *,
    price: float,
    side: str = "BUY",
) -> ShadowExecutionCycle:
    notional = price
    risk = PreTradeRiskReceipt(
        action="ALLOW",
        trading_state="ACTIVE",
        asset_id="BTCUSDT",
        requested_notional_usd=notional,
        reduce_only=False,
        reasons=(),
        projected_position_weight=0.1,
        checks=(("fixture", True),),
    )
    execution = SimulatedExecutionReceipt(
        status="FILLED",
        side=side,
        order_type="MARKET",
        submit_timestamp=TS,
        venue_timestamp=TS + 0.1,
        snapshot_timestamp=TS + 0.1,
        requested_base=1.0,
        filled_base=1.0,
        fill_fraction=1.0,
        average_fill_price=price,
        best_reference_price=price,
        adverse_slippage_bps=0.0,
        levels_consumed=1,
        slipped_one_tick=False,
        reason="phase65 fixture",
    )
    item = ShadowExecutionCycleItem(
        instruction_kind="OPEN",
        asset_id="BTCUSDT",
        risk_receipt=risk,
        execution_receipt=execution,
        pending_reversal=None,
        new_risk_cash_reserved_usd=notional,
        status="fixture",
    )
    return ShadowExecutionCycle(
        source_plan_id=f"plan-{cycle_id}",
        items=(item,),
        initial_available_cash_usd=2_000.0,
        reserved_new_risk_cash_usd=notional,
        remaining_unreserved_cash_usd=2_000.0 - notional,
        denied_assets=(),
        pending_reversal_assets=(),
        cycle_id=cycle_id,
    )


def _genesis() -> ShadowAccountState:
    return ShadowAccountState(
        account_id="paper-acct",
        observed_at=TS,
        equity_usd=2_000.0,
        available_cash_usd=2_000.0,
        position_weights=(),
        covered_assets=("BTCUSDT", "ETHUSDT"),
        source_kind="GENESIS",
        source_ref="phase65-genesis",
    )


def _venue() -> PaperVenue:
    return PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=2_000.0,
            fee_bps=0.0,
        )
    )


def _commit_cycle(
    ledger: ShadowStateLedger,
    venue: PaperVenue,
    projector: LocalExecutionProjector,
    cycle: ShadowExecutionCycle,
    *,
    observed_at: float,
    mark: float,
) -> None:
    ledger.append_cycle(cycle)
    paper = venue.apply_cycle(cycle)
    projector.process_cycle(
        paper,
        {fill.fill_id: fill for fill in venue.fills},
    )
    tracked = ("BTCUSDT", "ETHUSDT")
    reconciliation = venue.reconcile_against_local(
        projector.local_positions(tracked_assets=tracked),
        tracked_assets=tracked,
    )
    assert reconciliation.ready is True
    state = venue.build_reconciled_state(
        reconciliation,
        marks={"BTCUSDT": mark},
        observed_at=observed_at,
        source_ref=f"paper-report-{cycle.cycle_id}",
    )
    ledger.commit_reconciled_state(
        cycle.cycle_id,
        reconciliation,
        state,
    )


def test_pending_fill_crash_rebuilds_local_cache_from_paper_events() -> None:
    ledger = ShadowStateLedger(_genesis())
    venue = _venue()
    cycle = _cycle("pending", price=100.0)

    ledger.append_cycle(cycle)
    venue.apply_cycle(cycle)
    checkpoint = create_runtime_checkpoint(ledger, venue)

    restored = restore_runtime_with_local_projection(checkpoint)

    assert restored.pending_cycle_id == "pending"
    assert restored.ledger.pending_cycle_id == "pending"
    assert restored.venue.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert restored.projector.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert restored.reconciliation.ready is True
    assert restored.replay.complete is True
    assert restored.replay.cycles_replayed == 1
    assert restored.replay.fills_projected == 1
    assert len(restored.runtime_hash) == 64


def test_committed_runtime_restores_independent_local_projection_and_reconciles() -> None:
    ledger = ShadowStateLedger(_genesis())
    venue = _venue()
    projector = LocalExecutionProjector("paper-acct")
    _commit_cycle(
        ledger,
        venue,
        projector,
        _cycle("committed", price=100.0),
        observed_at=TS + 10,
        mark=110.0,
    )
    checkpoint = create_runtime_checkpoint(ledger, venue)

    restored = restore_runtime_with_local_projection(checkpoint)

    assert restored.pending_cycle_id is None
    assert restored.projector.manifest()["projection_hash"] == projector.manifest()["projection_hash"]
    assert restored.projector.position("BTCUSDT").quantity == pytest.approx(
        venue.position("BTCUSDT").quantity
    )
    assert restored.projector.position("BTCUSDT").avg_entry_price == pytest.approx(
        venue.position("BTCUSDT").avg_entry_price
    )
    assert restored.reconciliation.ready is True
    assert all(dict(restored.reconciliation.checks).values())


def test_multiple_cycles_replay_to_same_local_position_without_snapshot_copy() -> None:
    ledger = ShadowStateLedger(_genesis())
    venue = _venue()
    projector = LocalExecutionProjector("paper-acct")
    _commit_cycle(
        ledger,
        venue,
        projector,
        _cycle("c1", price=100.0),
        observed_at=TS + 10,
        mark=100.0,
    )
    _commit_cycle(
        ledger,
        venue,
        projector,
        _cycle("c2", price=120.0),
        observed_at=TS + 20,
        mark=120.0,
    )
    checkpoint = create_runtime_checkpoint(ledger, venue)

    replayed, receipt = replay_local_execution_history(checkpoint.paper)

    assert receipt.cycles_available == 2
    assert receipt.cycles_replayed == 2
    assert receipt.fills_available == 2
    assert receipt.fills_projected == 2
    assert receipt.complete is True
    assert replayed.position("BTCUSDT").quantity == pytest.approx(2.0)
    assert replayed.position("BTCUSDT").avg_entry_price == pytest.approx(110.0)
    assert replayed.manifest()["projection_hash"] == projector.manifest()["projection_hash"]


def test_truncated_local_event_replay_is_detected_by_phase50() -> None:
    ledger = ShadowStateLedger(_genesis())
    venue = _venue()
    projector = LocalExecutionProjector("paper-acct")
    _commit_cycle(
        ledger,
        venue,
        projector,
        _cycle("c1", price=100.0),
        observed_at=TS + 10,
        mark=100.0,
    )
    _commit_cycle(
        ledger,
        venue,
        projector,
        _cycle("c2", price=120.0),
        observed_at=TS + 20,
        mark=120.0,
    )
    checkpoint = create_runtime_checkpoint(ledger, venue)

    reconciliation = audit_partial_local_replay(
        checkpoint,
        through_state_version=1,
    )

    assert reconciliation.ready is False
    assert dict(reconciliation.checks)["positions_reconciled"] is False
    btc = next(row for row in reconciliation.results if row.asset_id == "BTCUSDT")
    assert btc.local_quantity == pytest.approx(1.0)
    assert btc.venue_quantity == pytest.approx(2.0)


def test_zero_cycle_runtime_recovers_flat_local_cache_and_explicit_flat_reports() -> None:
    ledger = ShadowStateLedger(_genesis())
    venue = _venue()
    checkpoint = create_runtime_checkpoint(ledger, venue)

    restored = restore_runtime_with_local_projection(checkpoint)

    assert restored.projector.projection_version == 0
    assert restored.projector.local_positions() == {}
    assert restored.replay.fills_projected == 0
    assert restored.reconciliation.ready is True
    assert set(restored.reconciliation.tracked_assets) == {"BTCUSDT", "ETHUSDT"}


@pytest.mark.parametrize("bad_version", [-1, 2])
def test_partial_replay_range_is_fail_closed(bad_version: int) -> None:
    ledger = ShadowStateLedger(_genesis())
    venue = _venue()
    cycle = _cycle("one-cycle", price=100.0)
    ledger.append_cycle(cycle)
    venue.apply_cycle(cycle)
    checkpoint = create_runtime_checkpoint(ledger, venue)

    with pytest.raises(ValueError, match="through_state_version"):
        replay_local_execution_history(
            checkpoint.paper,
            through_state_version=bad_version,
        )


def test_full_restore_is_deterministic_for_same_checkpoint() -> None:
    ledger = ShadowStateLedger(_genesis())
    venue = _venue()
    cycle = _cycle("deterministic", price=100.0)
    ledger.append_cycle(cycle)
    venue.apply_cycle(cycle)
    checkpoint = create_runtime_checkpoint(ledger, venue)

    first = restore_runtime_with_local_projection(checkpoint)
    second = restore_runtime_with_local_projection(checkpoint)

    assert first.runtime_hash == second.runtime_hash
    assert first.replay == second.replay
    assert first.projector.manifest() == second.projector.manifest()
    assert first.reconciliation.to_dict() == second.reconciliation.to_dict()
