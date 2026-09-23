from __future__ import annotations

from dataclasses import replace

import pytest

from brian2026.phase46_execution_simulator import SimulatedExecutionReceipt
from brian2026.phase56_pretrade_risk_engine import PreTradeRiskReceipt
from brian2026.phase57_shadow_execution_cycle import (
    ShadowExecutionCycle,
    ShadowExecutionCycleItem,
)
from brian2026.phase60_shadow_state_ledger import ShadowAccountState, ShadowStateLedger
from brian2026.phase61_stateful_paper_venue import PaperVenue, PaperVenueConfig
from brian2026.phase64_local_execution_projector import LocalExecutionProjector
from brian2026.phase66_durable_cycle_journal import DurableCycleJournal
from brian2026.phase66_runtime_coordinator import ShadowPaperRuntimeCoordinator
from brian2026.phase67_durable_runtime_orchestrator import (
    DurableRuntimeError,
    DurableShadowPaperRuntime,
)


TS = 1_760_000_000.0


def _genesis() -> ShadowAccountState:
    return ShadowAccountState(
        account_id="paper-acct",
        observed_at=TS,
        equity_usd=1000.0,
        available_cash_usd=1000.0,
        position_weights=(),
        covered_assets=("BTCUSDT", "ETHUSDT"),
        source_kind="GENESIS",
        source_ref="phase67-genesis",
    )


def _runtime() -> DurableShadowPaperRuntime:
    coordinator = ShadowPaperRuntimeCoordinator(
        ShadowStateLedger(_genesis()),
        PaperVenue(
            PaperVenueConfig(
                account_id="paper-acct",
                starting_cash_usd=1000.0,
                fee_bps=0.0,
            )
        ),
        LocalExecutionProjector("paper-acct"),
    )
    return DurableShadowPaperRuntime(coordinator, DurableCycleJournal())


def _cycle(
    cycle_id: str,
    *,
    asset: str = "BTCUSDT",
    price: float = 100.0,
) -> ShadowExecutionCycle:
    risk = PreTradeRiskReceipt(
        action="ALLOW",
        trading_state="ACTIVE",
        asset_id=asset,
        requested_notional_usd=price,
        reduce_only=False,
        reasons=(),
        projected_position_weight=0.1,
        checks=(("fixture", True),),
    )
    execution = SimulatedExecutionReceipt(
        status="FILLED",
        side="BUY",
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
        reason="phase67 fixture",
    )
    item = ShadowExecutionCycleItem(
        instruction_kind="OPEN",
        asset_id=asset,
        risk_receipt=risk,
        execution_receipt=execution,
        pending_reversal=None,
        new_risk_cash_reserved_usd=price,
        status="fixture",
    )
    return ShadowExecutionCycle(
        source_plan_id=f"plan-{cycle_id}",
        items=(item,),
        initial_available_cash_usd=1000.0,
        reserved_new_risk_cash_usd=price,
        remaining_unreserved_cash_usd=1000.0 - price,
        denied_assets=(),
        pending_reversal_assets=(),
        cycle_id=cycle_id,
    )


def test_normal_cycle_follows_full_write_ahead_stage_chain() -> None:
    runtime = _runtime()
    receipt = runtime.process_cycle(
        _cycle("normal"),
        marks={"BTCUSDT": 110.0},
        observed_at=TS + 10,
        source_ref="phase67-normal",
    )

    assert receipt.status == "COMMITTED"
    assert receipt.journal_stage == "COMMITTED"
    assert receipt.pending_cycle_id is None
    assert [entry.stage for entry in runtime.journal.entries] == [
        "CYCLE_CREATED",
        "PAPER_APPLIED",
        "LOCAL_PROJECTED",
        "RECONCILED",
        "COMMITTED",
    ]
    assert runtime.ledger.head_state.equity_usd == pytest.approx(1010.0)
    assert runtime.ledger.verify_integrity() is True
    assert runtime.journal.verify_integrity() is True


def test_crash_after_write_ahead_before_ledger_append_recovers_full_cycle_body() -> None:
    runtime = _runtime()
    cycle = _cycle("crash-write-ahead")
    runtime.journal_cycle(cycle)
    assert runtime.ledger.pending_cycle_id is None
    assert runtime.venue.state_version == 0

    checkpoint = runtime.checkpoint()
    restored = DurableShadowPaperRuntime.restore(checkpoint)

    assert restored.journal.latest_stage(cycle.cycle_id) == "CYCLE_CREATED"
    assert restored.ledger.pending_cycle_id is None
    assert restored.journal.cycle(cycle.cycle_id).to_dict() == cycle.to_dict()

    result = restored.advance_pending(
        marks={"BTCUSDT": 105.0},
        observed_at=TS + 10,
        source_ref="phase67-recover-write-ahead",
    )
    assert result.status == "COMMITTED"
    assert restored.venue.state_version == 1
    assert restored.venue.position("BTCUSDT").quantity == pytest.approx(1.0)


def test_crash_after_ledger_pending_before_paper_apply_resumes_without_duplicate_cycle() -> None:
    runtime = _runtime()
    cycle = _cycle("crash-ledger")
    runtime.journal_cycle(cycle)
    runtime.ledger.append_cycle(
        cycle,
        expected_state_id=runtime.ledger.head_state.state_id,
    )
    assert runtime.ledger.pending_cycle_id == cycle.cycle_id
    assert runtime.venue.state_version == 0

    restored = DurableShadowPaperRuntime.restore(runtime.checkpoint())
    result = restored.advance_pending(
        marks={"BTCUSDT": 105.0},
        observed_at=TS + 10,
        source_ref="phase67-recover-ledger",
    )

    assert result.status == "COMMITTED"
    assert restored.venue.state_version == 1
    assert restored.venue.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert restored.projector.position("BTCUSDT").quantity == pytest.approx(1.0)


def test_crash_after_paper_side_effect_before_journal_stage_is_forward_recovered() -> None:
    runtime = _runtime()
    cycle = _cycle("crash-paper")
    runtime.journal_cycle(cycle)
    runtime.ledger.append_cycle(
        cycle,
        expected_state_id=runtime.ledger.head_state.state_id,
    )
    runtime.venue.apply_cycle(cycle)
    assert runtime.journal.latest_stage(cycle.cycle_id) == "CYCLE_CREATED"
    assert runtime.venue.state_version == 1

    restored = DurableShadowPaperRuntime.restore(runtime.checkpoint())

    # Phase65 rebuilds local state from the durable paper events; Phase67 then
    # advances only journal metadata supported by those restored artifacts.
    assert restored.venue.state_version == 1
    assert restored.projector.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert restored.journal.latest_stage(cycle.cycle_id) == "RECONCILED"

    result = restored.advance_pending(
        marks={"BTCUSDT": 110.0},
        observed_at=TS + 10,
        source_ref="phase67-recover-paper",
    )
    assert result.status == "COMMITTED"
    assert restored.venue.state_version == 1
    assert restored.projector.projection_version == 1


def test_missing_marks_stops_at_reconciled_and_restart_can_finish_commit() -> None:
    runtime = _runtime()
    cycle = _cycle("marks-later")
    first = runtime.process_cycle(
        cycle,
        marks={},
        observed_at=TS + 10,
        source_ref="phase67-marks-missing",
    )
    assert first.status == "MARKS_REQUIRED"
    assert first.journal_stage == "RECONCILED"
    assert first.pending_cycle_id == cycle.cycle_id

    restored = DurableShadowPaperRuntime.restore(runtime.checkpoint())
    assert restored.journal.latest_stage(cycle.cycle_id) == "RECONCILED"

    second = restored.advance_pending(
        marks={"BTCUSDT": 120.0},
        observed_at=TS + 20,
        source_ref="phase67-marks-restored",
    )
    assert second.status == "COMMITTED"
    assert second.journal_stage == "COMMITTED"
    assert restored.ledger.pending_cycle_id is None
    assert restored.ledger.head_state.equity_usd == pytest.approx(1020.0)


def test_crash_after_ledger_commit_before_journal_commit_is_repaired_from_ledger_evidence() -> None:
    runtime = _runtime()
    cycle = _cycle("crash-after-ledger-commit")
    pending = runtime.process_cycle(
        cycle,
        marks={},
        observed_at=TS + 10,
        source_ref="phase67-before-ledger-commit",
    )
    assert pending.status == "MARKS_REQUIRED"
    assert runtime.journal.latest_stage(cycle.cycle_id) == "RECONCILED"

    tracked, reconciliation = runtime.coordinator.reconcile_current(
        extra_assets=("BTCUSDT",)
    )
    del tracked
    state = runtime.venue.build_reconciled_state(
        reconciliation,
        marks={"BTCUSDT": 110.0},
        observed_at=TS + 20,
        source_ref="phase67-manual-ledger-commit",
    )
    runtime.ledger.commit_reconciled_state(
        cycle.cycle_id,
        reconciliation,
        state,
        expected_state_id=runtime.ledger.head_state.state_id,
    )
    assert runtime.ledger.pending_cycle_id is None
    assert runtime.journal.latest_stage(cycle.cycle_id) == "RECONCILED"

    restored = DurableShadowPaperRuntime.restore(runtime.checkpoint())

    assert restored.ledger.pending_cycle_id is None
    assert restored.journal.latest_stage(cycle.cycle_id) == "COMMITTED"
    assert restored.journal.verify_integrity() is True


def test_different_cycle_cannot_enter_journal_while_one_is_active() -> None:
    runtime = _runtime()
    runtime.journal_cycle(_cycle("first"))

    with pytest.raises(DurableRuntimeError, match="must resolve"):
        runtime.journal_cycle(_cycle("second", asset="ETHUSDT", price=200.0))

    assert runtime.journal.cycle_ids == ("first",)


def test_replaying_same_write_ahead_cycle_is_idempotent() -> None:
    runtime = _runtime()
    cycle = _cycle("idem")
    first = runtime.journal_cycle(cycle)
    duplicate = runtime.journal_cycle(cycle)

    assert first.entry_id == duplicate.entry_id
    assert duplicate.duplicate is True
    assert len(runtime.journal.entries) == 1


def test_checkpoint_tamper_is_detected_before_recovery() -> None:
    runtime = _runtime()
    runtime.journal_cycle(_cycle("tamper"))
    checkpoint = runtime.checkpoint()

    # Mutate the nested manifest after checkpoint_id was computed.
    assert isinstance(checkpoint.journal_manifest, dict)
    checkpoint.journal_manifest["journal_hash"] = "0" * 64

    with pytest.raises(DurableRuntimeError, match="checkpoint content hash mismatch"):
        DurableShadowPaperRuntime.restore(checkpoint)


def test_no_pending_cycle_returns_explicit_noop() -> None:
    runtime = _runtime()
    receipt = runtime.advance_pending(
        marks={},
        observed_at=TS + 1,
        source_ref="phase67-noop",
    )
    assert receipt.status == "NO_PENDING_CYCLE"
    assert receipt.cycle_id is None
    assert receipt.journal_stage is None
    assert receipt.pending_cycle_id is None


def test_recovered_runtime_does_not_reapply_paper_or_projection_side_effects() -> None:
    runtime = _runtime()
    cycle = _cycle("no-duplicate-side-effects")
    first = runtime.process_cycle(
        cycle,
        marks={},
        observed_at=TS + 10,
        source_ref="phase67-first",
    )
    assert first.status == "MARKS_REQUIRED"
    assert runtime.venue.state_version == 1
    assert runtime.projector.projection_version == 1

    restored = DurableShadowPaperRuntime.restore(runtime.checkpoint())
    assert restored.venue.state_version == 1
    assert restored.projector.projection_version == 1

    completed = restored.advance_pending(
        marks={"BTCUSDT": 100.0},
        observed_at=TS + 20,
        source_ref="phase67-finish",
    )
    assert completed.status == "COMMITTED"
    assert restored.venue.state_version == 1
    assert restored.projector.projection_version == 1
