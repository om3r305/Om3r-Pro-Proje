from __future__ import annotations

import pytest

from brian2026.evidence_ledger import content_hash
from brian2026.phase50_execution_reconciliation import ReconciliationBatchReport
from brian2026.phase57_shadow_execution_cycle import ShadowExecutionCycle
from brian2026.phase60_shadow_state_ledger import (
    ShadowAccountState,
    ShadowStateConflictError,
    ShadowStateLedger,
)


def _genesis() -> ShadowAccountState:
    return ShadowAccountState(
        account_id="paper-acct",
        observed_at=100.0,
        equity_usd=1000.0,
        available_cash_usd=700.0,
        position_weights=(("BTCUSDT", 0.30),),
        covered_assets=("BTCUSDT", "ETHUSDT"),
        source_kind="GENESIS",
        source_ref="paper-bootstrap-v1",
    )


def _cycle(cycle_id: str, *, remaining_cash: float = 700.0) -> ShadowExecutionCycle:
    return ShadowExecutionCycle(
        source_plan_id=f"plan-{cycle_id}",
        items=(),
        initial_available_cash_usd=700.0,
        reserved_new_risk_cash_usd=max(0.0, 700.0 - remaining_cash),
        remaining_unreserved_cash_usd=remaining_cash,
        denied_assets=(),
        pending_reversal_assets=(),
        cycle_id=cycle_id,
    )


def _reconciliation(*, ready: bool = True) -> ReconciliationBatchReport:
    checks = (
        ("authoritative_reports_present", ready),
        ("positions_reconciled", ready),
        ("history_contract_acceptable", ready),
        ("no_unknown_command_outcomes", ready),
        ("no_duplicate_fill_ids", ready),
    )
    return ReconciliationBatchReport(
        results=(),
        tracked_assets=("BTCUSDT", "ETHUSDT"),
        reports_complete=True,
        unresolved_command_ids=() if ready else ("unknown-cmd",),
        duplicate_fill_ids=(),
        checks=checks,
        ready=ready,
    )


def _reconciled_state(
    report: ReconciliationBatchReport,
    *,
    observed_at: float = 200.0,
    equity: float = 1010.0,
    cash: float = 610.0,
    covered_assets: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
) -> ShadowAccountState:
    return ShadowAccountState(
        account_id="paper-acct",
        observed_at=observed_at,
        equity_usd=equity,
        available_cash_usd=cash,
        position_weights=(("BTCUSDT", 0.20), ("ETHUSDT", 0.15)),
        covered_assets=covered_assets,
        source_kind="RECONCILED_PAPER",
        source_ref="paper-account-report-200",
        reconciliation_hash=content_hash(report.to_dict()),
    )


def test_genesis_is_content_addressed_and_integrity_verified() -> None:
    genesis = _genesis()
    ledger = ShadowStateLedger(genesis)
    assert ledger.head_state.state_id == genesis.state_id
    assert ledger.pending_cycle_id is None
    assert ledger.verify_integrity() is True
    manifest = ledger.manifest()
    assert manifest["append_only"] is True
    assert manifest["head_state_id"] == genesis.state_id
    assert manifest["live_execution"] is False


def test_simulation_cycle_is_recorded_without_mutating_account_state() -> None:
    ledger = ShadowStateLedger(_genesis())
    before = ledger.head_state
    cycle = _cycle("cycle-1", remaining_cash=300.0)

    proposal = ledger.append_cycle(cycle, expected_state_id=before.state_id)
    assert proposal.duplicate is False
    assert ledger.pending_cycle_id == "cycle-1"
    assert ledger.head_state.state_id == before.state_id
    assert ledger.head_state.available_cash_usd == pytest.approx(700.0)

    closed = ledger.close_simulation_cycle(cycle, expected_state_id=before.state_id)
    assert closed.duplicate is False
    assert ledger.pending_cycle_id is None
    assert ledger.head_state.state_id == before.state_id
    assert ledger.head_state.available_cash_usd == pytest.approx(700.0)
    assert [row.kind for row in ledger.transitions] == [
        "GENESIS",
        "CYCLE_PROPOSED",
        "SIMULATION_CLOSED",
    ]
    assert ledger.verify_integrity() is True


def test_second_cycle_is_blocked_until_prior_cycle_is_closed() -> None:
    ledger = ShadowStateLedger(_genesis())
    ledger.append_cycle(_cycle("cycle-1"))
    with pytest.raises(ShadowStateConflictError, match="must be closed"):
        ledger.append_cycle(_cycle("cycle-2"))

    ledger.close_simulation_cycle(_cycle("cycle-1"))
    receipt = ledger.append_cycle(_cycle("cycle-2"))
    assert receipt.cycle_id == "cycle-2"
    assert ledger.pending_cycle_id == "cycle-2"


def test_ready_reconciliation_commits_new_authoritative_head() -> None:
    ledger = ShadowStateLedger(_genesis())
    cycle = _cycle("cycle-commit")
    ledger.append_cycle(cycle)
    report = _reconciliation()
    state = _reconciled_state(report)

    receipt = ledger.commit_reconciled_state(
        cycle.cycle_id,
        report,
        state,
        expected_state_id=ledger.head_state.state_id,
    )
    assert receipt.duplicate is False
    assert ledger.pending_cycle_id is None
    assert ledger.head_state.state_id == state.state_id
    assert ledger.head_state.available_cash_usd == pytest.approx(610.0)
    assert ledger.transitions[-1].kind == "RECONCILED_COMMIT"
    assert ledger.transitions[-1].reconciliation_hash == content_hash(report.to_dict())
    assert ledger.verify_integrity() is True


def test_not_ready_reconciliation_cannot_mutate_state() -> None:
    ledger = ShadowStateLedger(_genesis())
    cycle = _cycle("cycle-bad")
    ledger.append_cycle(cycle)
    bad = _reconciliation(ready=False)
    state = ShadowAccountState(
        account_id="paper-acct",
        observed_at=200.0,
        equity_usd=1000.0,
        available_cash_usd=700.0,
        position_weights=(("BTCUSDT", 0.30),),
        covered_assets=("BTCUSDT", "ETHUSDT"),
        source_kind="RECONCILED_PAPER",
        source_ref="bad-report",
        reconciliation_hash=content_hash(bad.to_dict()),
    )
    before_id = ledger.head_state.state_id
    with pytest.raises(ShadowStateConflictError, match="not ready"):
        ledger.commit_reconciled_state(cycle.cycle_id, bad, state)
    assert ledger.head_state.state_id == before_id
    assert ledger.pending_cycle_id == cycle.cycle_id


def test_reconciliation_hash_must_match_exact_state_snapshot() -> None:
    ledger = ShadowStateLedger(_genesis())
    cycle = _cycle("cycle-hash")
    ledger.append_cycle(cycle)
    report = _reconciliation()
    state = ShadowAccountState(
        account_id="paper-acct",
        observed_at=200.0,
        equity_usd=1000.0,
        available_cash_usd=650.0,
        position_weights=(("BTCUSDT", 0.25),),
        covered_assets=("BTCUSDT", "ETHUSDT"),
        source_kind="RECONCILED_PAPER",
        source_ref="mismatched-report",
        reconciliation_hash="b" * 64,
    )
    with pytest.raises(ShadowStateConflictError, match="does not match"):
        ledger.commit_reconciled_state(cycle.cycle_id, report, state)


def test_reconciled_state_must_cover_every_tracked_asset_in_report() -> None:
    ledger = ShadowStateLedger(_genesis())
    cycle = _cycle("cycle-coverage")
    ledger.append_cycle(cycle)
    report = _reconciliation()
    state = _reconciled_state(report, covered_assets=("BTCUSDT",))
    with pytest.raises(ShadowStateConflictError, match="does not cover"):
        ledger.commit_reconciled_state(cycle.cycle_id, report, state)


def test_state_timestamp_cannot_move_backwards() -> None:
    ledger = ShadowStateLedger(_genesis())
    cycle = _cycle("cycle-time")
    ledger.append_cycle(cycle)
    report = _reconciliation()
    state = _reconciled_state(report, observed_at=99.0)
    with pytest.raises(ShadowStateConflictError, match="backwards"):
        ledger.commit_reconciled_state(cycle.cycle_id, report, state)


def test_duplicate_cycle_and_commit_are_idempotent_but_conflicts_fail() -> None:
    ledger = ShadowStateLedger(_genesis())
    cycle = _cycle("cycle-idem")
    first = ledger.append_cycle(cycle)
    duplicate = ledger.append_cycle(cycle)
    assert first.transition_id == duplicate.transition_id
    assert duplicate.duplicate is True

    conflicting = _cycle("cycle-idem", remaining_cash=500.0)
    with pytest.raises(ShadowStateConflictError, match="different execution evidence"):
        ledger.append_cycle(conflicting)

    report = _reconciliation()
    state = _reconciled_state(report)
    commit = ledger.commit_reconciled_state(cycle.cycle_id, report, state)
    duplicate_commit = ledger.commit_reconciled_state(cycle.cycle_id, report, state)
    assert duplicate_commit.duplicate is True
    assert duplicate_commit.transition_id == commit.transition_id


def test_stale_expected_state_id_is_rejected() -> None:
    ledger = ShadowStateLedger(_genesis())
    with pytest.raises(ShadowStateConflictError, match="stale expected_state_id"):
        ledger.append_cycle(_cycle("cycle-stale"), expected_state_id="0" * 64)


def test_genesis_cannot_claim_reconciliation_and_reconciled_state_requires_hash() -> None:
    with pytest.raises(ValueError, match="genesis state cannot claim"):
        ShadowAccountState(
            account_id="paper-acct",
            observed_at=100.0,
            equity_usd=1000.0,
            available_cash_usd=700.0,
            position_weights=(),
            covered_assets=("BTCUSDT",),
            source_kind="GENESIS",
            source_ref="bad-genesis",
            reconciliation_hash="a" * 64,
        )

    with pytest.raises(ValueError, match="requires a reconciliation"):
        ShadowAccountState(
            account_id="paper-acct",
            observed_at=100.0,
            equity_usd=1000.0,
            available_cash_usd=700.0,
            position_weights=(),
            covered_assets=("BTCUSDT",),
            source_kind="RECONCILED_PAPER",
            source_ref="missing-hash",
            reconciliation_hash=None,
        )
