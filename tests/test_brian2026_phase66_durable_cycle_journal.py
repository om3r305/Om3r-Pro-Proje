from __future__ import annotations

from dataclasses import replace

import pytest

from brian2026.evidence_ledger import content_hash
from brian2026.phase46_execution_simulator import SimulatedExecutionReceipt
from brian2026.phase50_execution_reconciliation import ReconciliationBatchReport
from brian2026.phase56_pretrade_risk_engine import PreTradeRiskReceipt
from brian2026.phase57_shadow_execution_cycle import (
    ShadowExecutionCycle,
    ShadowExecutionCycleItem,
)
from brian2026.phase61_stateful_paper_venue import PaperVenue, PaperVenueConfig
from brian2026.phase64_local_execution_projector import LocalExecutionProjector
from brian2026.phase66_durable_cycle_journal import (
    CycleJournalError,
    DurableCycleJournal,
    restore_cycle_journal,
)


TS = 1_760_000_000.0


def _cycle(cycle_id: str = "cycle-66", price: float = 100.0) -> ShadowExecutionCycle:
    risk = PreTradeRiskReceipt(
        action="ALLOW",
        trading_state="ACTIVE",
        asset_id="BTCUSDT",
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
        reason="phase66 fixture",
    )
    item = ShadowExecutionCycleItem(
        instruction_kind="OPEN",
        asset_id="BTCUSDT",
        risk_receipt=risk,
        execution_receipt=execution,
        pending_reversal=None,
        new_risk_cash_reserved_usd=price,
        status="fixture",
    )
    return ShadowExecutionCycle(
        source_plan_id=f"plan-{cycle_id}",
        items=(item,),
        initial_available_cash_usd=1_000.0,
        reserved_new_risk_cash_usd=price,
        remaining_unreserved_cash_usd=1_000.0 - price,
        denied_assets=(),
        pending_reversal_assets=(),
        cycle_id=cycle_id,
    )


def _artifacts(cycle: ShadowExecutionCycle):
    venue = PaperVenue(
        PaperVenueConfig(
            account_id="paper-acct",
            starting_cash_usd=1_000.0,
            fee_bps=0.0,
        )
    )
    paper = venue.apply_cycle(cycle)
    projector = LocalExecutionProjector("paper-acct")
    projection = projector.process_cycle(
        paper,
        {fill.fill_id: fill for fill in venue.fills},
    )
    reconciliation = venue.reconcile_against_local(
        projector.local_positions(tracked_assets=("BTCUSDT",)),
        tracked_assets=("BTCUSDT",),
    )
    assert reconciliation.ready is True
    return venue, paper, projector, projection, reconciliation


def test_full_cycle_stage_chain_is_append_only_and_integrity_verified() -> None:
    cycle = _cycle()
    _, paper, _, projection, reconciliation = _artifacts(cycle)
    journal = DurableCycleJournal()

    created = journal.record_cycle(cycle)
    applied = journal.mark_paper_applied(cycle.cycle_id, paper)
    projected = journal.mark_local_projected(cycle.cycle_id, projection)
    reconciled = journal.mark_reconciliation(cycle.cycle_id, reconciliation)
    state_id = "a" * 64
    committed = journal.mark_committed(cycle.cycle_id, state_id=state_id)

    assert [row.stage for row in journal.entries] == [
        "CYCLE_CREATED",
        "PAPER_APPLIED",
        "LOCAL_PROJECTED",
        "RECONCILED",
        "COMMITTED",
    ]
    assert created.sequence == 0
    assert committed.sequence == 4
    assert journal.latest_stage(cycle.cycle_id) == "COMMITTED"
    assert journal.verify_integrity() is True
    assert all(
        journal.entries[index].previous_entry_id == journal.entries[index - 1].entry_id
        for index in range(1, len(journal.entries))
    )


def test_crash_after_cycle_created_restores_full_retryable_cycle_body() -> None:
    cycle = _cycle("crash-before-paper")
    journal = DurableCycleJournal()
    journal.record_cycle(cycle)

    restored = restore_cycle_journal(journal.manifest())

    assert restored.latest_stage(cycle.cycle_id) == "CYCLE_CREATED"
    assert restored.cycle(cycle.cycle_id).to_dict() == cycle.to_dict()
    assert restored.verify_integrity() is True


def test_same_cycle_body_is_idempotent_but_conflicting_body_is_rejected() -> None:
    cycle = _cycle("idem")
    journal = DurableCycleJournal()
    first = journal.record_cycle(cycle)
    duplicate = journal.record_cycle(cycle)

    assert first.entry_id == duplicate.entry_id
    assert duplicate.duplicate is True
    assert len(journal.entries) == 1

    conflicting = replace(
        cycle,
        remaining_unreserved_cash_usd=cycle.remaining_unreserved_cash_usd - 1.0,
    )
    with pytest.raises(CycleJournalError, match="different full cycle body"):
        journal.record_cycle(conflicting)


def test_illegal_stage_skips_are_blocked() -> None:
    cycle = _cycle("illegal")
    _, paper, _, projection, reconciliation = _artifacts(cycle)
    journal = DurableCycleJournal()
    journal.record_cycle(cycle)

    with pytest.raises(CycleJournalError, match="illegal journal stage transition"):
        journal.mark_local_projected(cycle.cycle_id, projection)

    journal.mark_paper_applied(cycle.cycle_id, paper)
    with pytest.raises(CycleJournalError, match="illegal journal stage transition"):
        journal.mark_reconciliation(cycle.cycle_id, reconciliation)

    with pytest.raises(CycleJournalError, match="illegal journal stage transition"):
        journal.mark_committed(cycle.cycle_id, state_id="b" * 64)


def test_reconciliation_required_can_later_advance_to_reconciled_then_commit() -> None:
    cycle = _cycle("reconcile-later")
    _, paper, _, projection, good = _artifacts(cycle)
    bad = ReconciliationBatchReport(
        results=good.results,
        tracked_assets=good.tracked_assets,
        reports_complete=good.reports_complete,
        unresolved_command_ids=("missing-event",),
        duplicate_fill_ids=(),
        checks=tuple(
            (name, False if name == "no_unknown_command_outcomes" else value)
            for name, value in good.checks
        ),
        ready=False,
    )

    journal = DurableCycleJournal()
    journal.record_cycle(cycle)
    journal.mark_paper_applied(cycle.cycle_id, paper)
    journal.mark_local_projected(cycle.cycle_id, projection)
    pending = journal.mark_reconciliation(cycle.cycle_id, bad)

    assert pending.stage == "RECONCILIATION_REQUIRED"
    assert journal.latest_stage(cycle.cycle_id) == "RECONCILIATION_REQUIRED"

    reconciled = journal.mark_reconciliation(cycle.cycle_id, good)
    assert reconciled.stage == "RECONCILED"
    journal.mark_committed(cycle.cycle_id, state_id="c" * 64)
    assert journal.latest_stage(cycle.cycle_id) == "COMMITTED"
    assert journal.verify_integrity() is True


def test_paper_receipt_content_id_is_reverified() -> None:
    cycle = _cycle("paper-hash")
    _, paper, _, _, _ = _artifacts(cycle)
    journal = DurableCycleJournal()
    journal.record_cycle(cycle)

    tampered = replace(
        paper,
        cash_after_usd=paper.cash_after_usd + 1.0,
    )
    with pytest.raises(CycleJournalError, match="paper receipt content hash mismatch"):
        journal.mark_paper_applied(cycle.cycle_id, tampered)


def test_duplicate_stage_artifact_is_idempotent_after_later_stages() -> None:
    cycle = _cycle("stage-idem")
    _, paper, _, projection, reconciliation = _artifacts(cycle)
    journal = DurableCycleJournal()
    journal.record_cycle(cycle)
    first_paper = journal.mark_paper_applied(cycle.cycle_id, paper)
    journal.mark_local_projected(cycle.cycle_id, projection)
    journal.mark_reconciliation(cycle.cycle_id, reconciliation)

    repeated = journal.mark_paper_applied(cycle.cycle_id, paper)

    assert repeated.duplicate is True
    assert repeated.entry_id == first_paper.entry_id
    assert journal.latest_stage(cycle.cycle_id) == "RECONCILED"


def test_abort_is_terminal() -> None:
    cycle = _cycle("abort")
    _, paper, _, _, _ = _artifacts(cycle)
    journal = DurableCycleJournal()
    journal.record_cycle(cycle)
    journal.mark_aborted(cycle.cycle_id, reason="operator rejected paper cycle")

    assert journal.latest_stage(cycle.cycle_id) == "ABORTED"
    with pytest.raises(CycleJournalError, match="illegal journal stage transition"):
        journal.mark_paper_applied(cycle.cycle_id, paper)


def test_manifest_roundtrip_preserves_full_cycle_and_stage_chain() -> None:
    cycle = _cycle("roundtrip")
    _, paper, _, projection, reconciliation = _artifacts(cycle)
    journal = DurableCycleJournal()
    journal.record_cycle(cycle)
    journal.mark_paper_applied(cycle.cycle_id, paper)
    journal.mark_local_projected(cycle.cycle_id, projection)
    journal.mark_reconciliation(cycle.cycle_id, reconciliation)

    manifest = journal.manifest()
    restored = restore_cycle_journal(manifest)

    assert restored.manifest() == manifest
    assert restored.cycle(cycle.cycle_id).to_dict() == cycle.to_dict()
    assert restored.latest_stage(cycle.cycle_id) == "RECONCILED"
    assert restored.verify_integrity() is True


def test_tampered_cycle_body_is_detected_even_if_outer_manifest_hash_is_recomputed() -> None:
    cycle = _cycle("tamper-body")
    journal = DurableCycleJournal()
    journal.record_cycle(cycle)
    manifest = journal.manifest()

    cycles = {
        key: dict(value)
        for key, value in manifest["cycles"].items()
    }
    cycles[cycle.cycle_id]["remaining_unreserved_cash_usd"] -= 1.0
    entries = list(manifest["entries"])
    forged = dict(manifest)
    forged["cycles"] = cycles
    forged["journal_hash"] = content_hash({
        "cycles": cycles,
        "entries": entries,
    })

    with pytest.raises(CycleJournalError, match="cycle hash"):
        restore_cycle_journal(forged)


def test_manifest_hash_tamper_is_rejected() -> None:
    cycle = _cycle("tamper-manifest")
    journal = DurableCycleJournal()
    journal.record_cycle(cycle)
    manifest = dict(journal.manifest())
    manifest["journal_hash"] = "0" * 64

    with pytest.raises(CycleJournalError, match="manifest hash mismatch"):
        restore_cycle_journal(manifest)
