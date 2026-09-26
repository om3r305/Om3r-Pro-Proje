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
from brian2026.phase64_local_execution_projector import (
    LocalExecutionProjector,
    ProjectedPosition,
)
from brian2026.phase66_runtime_coordinator import (
    RuntimeCoordinatorError,
    ShadowPaperRuntimeCoordinator,
)


TS = 1_760_000_000.0


def _genesis(account_id: str = "paper-acct") -> ShadowAccountState:
    return ShadowAccountState(
        account_id=account_id,
        observed_at=TS,
        equity_usd=1000.0,
        available_cash_usd=1000.0,
        position_weights=(),
        covered_assets=("BTCUSDT", "ETHUSDT"),
        source_kind="GENESIS",
        source_ref="phase66-genesis",
    )


def _coordinator() -> ShadowPaperRuntimeCoordinator:
    ledger = ShadowStateLedger(_genesis())
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=1000.0,
            fee_bps=0.0,
        )
    )
    projector = LocalExecutionProjector("paper-acct")
    return ShadowPaperRuntimeCoordinator(ledger, venue, projector)


def _cycle(
    cycle_id: str,
    *,
    asset: str = "BTCUSDT",
    side: str = "BUY",
    price: float = 100.0,
    status: str = "FILLED",
    filled_base: float = 1.0,
    allowed: bool = True,
) -> ShadowExecutionCycle:
    risk = PreTradeRiskReceipt(
        action="ALLOW" if allowed else "DENY",
        trading_state="ACTIVE",
        asset_id=asset,
        requested_notional_usd=price,
        reduce_only=False,
        reasons=() if allowed else ("fixture-denied",),
        projected_position_weight=0.1 if allowed else None,
        checks=(("fixture", allowed),),
    )
    execution = None
    if allowed:
        execution = SimulatedExecutionReceipt(
            status=status,
            side=side,
            order_type="MARKET",
            submit_timestamp=TS,
            venue_timestamp=TS + 0.1,
            snapshot_timestamp=TS + 0.1,
            requested_base=1.0,
            filled_base=filled_base,
            fill_fraction=filled_base,
            average_fill_price=(
                None if status == "NO_FILL" else price
            ),
            best_reference_price=price,
            adverse_slippage_bps=0.0,
            levels_consumed=1 if filled_base > 0 else 0,
            slipped_one_tick=False,
            reason="phase66 fixture",
        )
    item = ShadowExecutionCycleItem(
        instruction_kind="OPEN",
        asset_id=asset,
        risk_receipt=risk,
        execution_receipt=execution,
        pending_reversal=None,
        new_risk_cash_reserved_usd=price if allowed else 0.0,
        status="fixture",
    )
    return ShadowExecutionCycle(
        source_plan_id=f"plan-{cycle_id}",
        items=(item,),
        initial_available_cash_usd=1000.0,
        reserved_new_risk_cash_usd=price if allowed else 0.0,
        remaining_unreserved_cash_usd=1000.0 - (price if allowed else 0.0),
        denied_assets=() if allowed else (asset,),
        pending_reversal_assets=(),
        cycle_id=cycle_id,
    )


def test_complete_cycle_commits_only_after_independent_reconciliation() -> None:
    runtime = _coordinator()
    cycle = _cycle("complete")

    receipt = runtime.process_cycle(
        cycle,
        marks={"BTCUSDT": 110.0},
        observed_at=TS + 10,
        source_ref="phase66-complete",
    )

    assert receipt.status == "COMMITTED"
    assert receipt.pending_cycle_id is None
    assert receipt.before_state_id != receipt.after_state_id
    assert receipt.commit_transition_id is not None
    assert runtime.ledger.pending_cycle_id is None
    assert runtime.venue.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert runtime.projector.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert runtime.ledger.head_state.equity_usd == pytest.approx(1010.0)
    assert runtime.ledger.verify_integrity() is True


def test_missing_mark_leaves_cycle_pending_without_mutating_authoritative_head() -> None:
    runtime = _coordinator()
    before = runtime.ledger.head_state
    cycle = _cycle("needs-mark")

    receipt = runtime.process_cycle(
        cycle,
        marks={},
        observed_at=TS + 10,
        source_ref="phase66-needs-mark",
    )

    assert receipt.status == "MARKS_REQUIRED"
    assert receipt.missing_mark_assets == ("BTCUSDT",)
    assert receipt.pending_cycle_id == cycle.cycle_id
    assert receipt.after_state_id == before.state_id
    assert runtime.ledger.head_state.state_id == before.state_id
    assert runtime.venue.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert runtime.projector.position("BTCUSDT").quantity == pytest.approx(1.0)


def test_pending_cycle_blocks_different_new_cycle() -> None:
    runtime = _coordinator()
    runtime.process_cycle(
        _cycle("pending"),
        marks={},
        observed_at=TS + 10,
        source_ref="phase66-pending",
    )

    with pytest.raises(RuntimeCoordinatorError, match="must resolve"):
        runtime.process_cycle(
            _cycle("too-early", asset="ETHUSDT", price=200.0),
            marks={"BTCUSDT": 100.0, "ETHUSDT": 200.0},
            observed_at=TS + 20,
            source_ref="phase66-too-early",
        )

    assert runtime.ledger.pending_cycle_id == "pending"
    assert runtime.venue.position("ETHUSDT").quantity == pytest.approx(0.0)


def test_same_pending_cycle_can_be_replayed_idempotently_with_missing_input_fixed() -> None:
    runtime = _coordinator()
    cycle = _cycle("same-cycle")
    first = runtime.process_cycle(
        cycle,
        marks={},
        observed_at=TS + 10,
        source_ref="phase66-first",
    )
    assert first.status == "MARKS_REQUIRED"
    paper_version = runtime.venue.state_version
    projection_version = runtime.projector.projection_version

    second = runtime.process_cycle(
        cycle,
        marks={"BTCUSDT": 105.0},
        observed_at=TS + 11,
        source_ref="phase66-second",
    )

    assert second.status == "COMMITTED"
    assert second.pending_cycle_id is None
    assert runtime.venue.state_version == paper_version
    assert runtime.projector.projection_version == projection_version
    assert runtime.venue.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert runtime.projector.position("BTCUSDT").quantity == pytest.approx(1.0)


def test_checkpoint_restart_can_resume_pending_without_original_cycle_payload() -> None:
    runtime = _coordinator()
    cycle = _cycle("crash-resume")
    pending = runtime.process_cycle(
        cycle,
        marks={},
        observed_at=TS + 10,
        source_ref="phase66-before-crash",
    )
    assert pending.status == "MARKS_REQUIRED"

    checkpoint = runtime.checkpoint()
    restored = ShadowPaperRuntimeCoordinator.restore(checkpoint)

    assert restored.ledger.pending_cycle_id == cycle.cycle_id
    assert restored.venue.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert restored.projector.position("BTCUSDT").quantity == pytest.approx(1.0)

    resumed = restored.resume_pending(
        marks={"BTCUSDT": 110.0},
        observed_at=TS + 20,
        source_ref="phase66-after-restart",
    )
    assert resumed.status == "COMMITTED"
    assert resumed.cycle_id == cycle.cycle_id
    assert resumed.pending_cycle_id is None
    assert restored.ledger.head_state.equity_usd == pytest.approx(1010.0)
    assert restored.ledger.verify_integrity() is True


def test_resume_pending_fails_closed_when_local_projection_diverges() -> None:
    runtime = _coordinator()
    cycle = _cycle("divergence")
    runtime.process_cycle(
        cycle,
        marks={},
        observed_at=TS + 10,
        source_ref="phase66-divergence-start",
    )

    runtime.projector._positions["BTCUSDT"] = ProjectedPosition(
        asset_id="BTCUSDT",
        quantity=0.5,
        avg_entry_price=100.0,
        realized_pnl_quote=0.0,
        source_fill_ids=runtime.projector.position("BTCUSDT").source_fill_ids,
    )

    resumed = runtime.resume_pending(
        marks={"BTCUSDT": 100.0},
        observed_at=TS + 20,
        source_ref="phase66-divergence-resume",
    )
    assert resumed.status == "RECONCILIATION_BLOCKED"
    assert resumed.pending_cycle_id == cycle.cycle_id
    assert resumed.after_state_id == resumed.before_state_id
    assert runtime.ledger.pending_cycle_id == cycle.cycle_id


def test_no_pending_resume_is_explicit_noop() -> None:
    runtime = _coordinator()
    receipt = runtime.resume_pending(
        marks={},
        observed_at=TS + 1,
        source_ref="phase66-no-pending",
    )
    assert receipt.status == "NO_PENDING_CYCLE"
    assert receipt.cycle_id is None
    assert receipt.pending_cycle_id is None
    assert receipt.before_state_id == receipt.after_state_id


def test_risk_denied_cycle_can_commit_flat_reconciled_state_without_marks() -> None:
    runtime = _coordinator()
    receipt = runtime.process_cycle(
        _cycle("risk-denied", allowed=False),
        marks={},
        observed_at=TS + 10,
        source_ref="phase66-risk-denied",
    )
    assert receipt.status == "COMMITTED"
    assert receipt.pending_cycle_id is None
    assert runtime.venue.position("BTCUSDT").quantity == pytest.approx(0.0)
    assert runtime.projector.position("BTCUSDT").quantity == pytest.approx(0.0)
    assert runtime.ledger.head_state.position_weights == ()


def test_account_identity_split_brain_is_rejected_at_construction() -> None:
    ledger = ShadowStateLedger(_genesis(account_id="ledger-acct"))
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="venue-acct",
            starting_cash_usd=1000.0,
        )
    )
    projector = LocalExecutionProjector("venue-acct")
    with pytest.raises(RuntimeCoordinatorError, match="account ids must match"):
        ShadowPaperRuntimeCoordinator(ledger, venue, projector)


def test_cycle_receipt_is_deterministic_for_same_completed_inputs_on_fresh_runtimes() -> None:
    cycle = _cycle("deterministic")
    first = _coordinator().process_cycle(
        cycle,
        marks={"BTCUSDT": 105.0},
        observed_at=TS + 10,
        source_ref="phase66-deterministic",
    )
    second = _coordinator().process_cycle(
        cycle,
        marks={"BTCUSDT": 105.0},
        observed_at=TS + 10,
        source_ref="phase66-deterministic",
    )
    assert first.receipt_id == second.receipt_id
    assert first.to_dict() == second.to_dict()
