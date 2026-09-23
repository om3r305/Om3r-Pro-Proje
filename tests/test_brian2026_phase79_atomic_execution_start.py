from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase70_durable_runtime_store import RuntimeLease
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from brian2026.phase77_execution_claim_lifecycle import (
    ClaimCompletionReceipt,
    ExecutionClaimError,
    ExecutionClaimReceipt,
)
from brian2026.phase78_execution_kill_switch import ExecutionKillSwitchDecision
from brian2026.phase79_atomic_execution_start import (
    AtomicExecutionStartError,
    AtomicExecutionStartReceipt,
    AtomicExecutionStartStore,
    PersistedAtomicStartedRuntimeSupervisor,
)


def _lease() -> RuntimeLease:
    return RuntimeLease(
        runtime_id="runtime-79",
        owner_token="owner-a",
        fencing_token=17,
        version=8,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )


def _claim() -> ExecutionClaimReceipt:
    return ExecutionClaimReceipt(
        runtime_id="runtime-79",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=8,
        fencing_token=17,
        claim_fencing_token=5,
        status="CLAIMED",
        claimed=True,
        cancelled=False,
        terminal=False,
        risk_version=12,
        risk_receipt_id="r" * 64,
        journal_stage="CYCLE_CREATED",
        resume_only=False,
        cancel_reason=None,
        claim_until="2026-09-23T14:00:00Z",
        completion_checkpoint_id=None,
    )


class FakeRpc:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, name, params):
        self.calls.append((name, dict(params)))
        value = self.response
        return value(dict(params)) if callable(value) else dict(value)


def _start_row(
    *,
    status="STARTED",
    started=True,
    duplicate=False,
    terminal=False,
    cancel=False,
    phase78_status="PROCEED",
    stage="CYCLE_CREATED",
    reason=None,
    resume_only=False,
):
    return {
        "started": started,
        "duplicate": duplicate,
        "terminal": terminal,
        "cancel_requested": cancel,
        "status": status,
        "runtime_id": "runtime-79",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "runtime_version": 8,
        "fencing_token": 17,
        "claim_fencing_token": 5,
        "risk_version": 13,
        "risk_receipt_id": "s" * 64,
        "risk_state": "HALTED" if cancel else "ACTIVE",
        "journal_stage": stage,
        "phase78_status": phase78_status,
        "reason": reason,
        "resume_only": resume_only,
    }


def test_store_marks_started_with_exact_claim_fence() -> None:
    rpc = FakeRpc(_start_row())
    receipt = AtomicExecutionStartStore(rpc).mark_started(
        _lease(),
        _claim(),
        worker_token="worker-a",
    )

    assert receipt.started is True
    assert receipt.status == "STARTED"
    assert receipt.phase78_status == "PROCEED"
    assert receipt.resume_only is False
    assert rpc.calls == [(
        "brian_mark_shadow_execution_started",
        {
            "p_runtime_id": "runtime-79",
            "p_owner_token": "owner-a",
            "p_fencing_token": 17,
            "p_cycle_id": "c" * 64,
            "p_worker_token": "worker-a",
            "p_claim_fencing_token": 5,
        },
    )]


def test_store_parses_resume_and_duplicate_start_without_losing_cancel_evidence() -> None:
    resume = AtomicExecutionStartStore(
        FakeRpc(
            _start_row(
                status="STARTED_RESUME",
                phase78_status="CANCEL_REQUESTED",
                stage="PAPER_APPLIED",
                cancel=True,
                reason="HALTED",
                resume_only=True,
            )
        )
    ).mark_started(_lease(), _claim(), worker_token="worker-recover")
    assert resume.started is True
    assert resume.resume_only is True
    assert resume.cancel_requested is True
    assert resume.reason == "HALTED"

    duplicate = AtomicExecutionStartStore(
        FakeRpc(
            _start_row(
                status="STARTED_ALREADY",
                duplicate=True,
            )
        )
    ).mark_started(_lease(), _claim(), worker_token="worker-a")
    assert duplicate.started is True
    assert duplicate.duplicate is True


def test_store_rejects_non_owned_claim_before_rpc() -> None:
    claim = _claim()
    claim = ExecutionClaimReceipt(
        runtime_id=claim.runtime_id,
        cycle_id=claim.cycle_id,
        dispatch_id=claim.dispatch_id,
        runtime_version=claim.runtime_version,
        fencing_token=claim.fencing_token,
        claim_fencing_token=0,
        status="COMPLETED",
        claimed=False,
        cancelled=False,
        terminal=True,
        risk_version=claim.risk_version,
        risk_receipt_id=claim.risk_receipt_id,
        journal_stage="COMMITTED",
        resume_only=True,
        cancel_reason=None,
        claim_until=None,
        completion_checkpoint_id="z" * 64,
    )
    rpc = FakeRpc({})
    with pytest.raises(AtomicExecutionStartError, match="active owned claim"):
        AtomicExecutionStartStore(rpc).mark_started(
            _lease(),
            claim,
            worker_token="worker-a",
        )
    assert rpc.calls == []


def test_store_rejects_inconsistent_started_and_terminal_flags() -> None:
    row = _start_row()
    row["terminal"] = True
    with pytest.raises(ValueError, match="cannot already be terminal"):
        AtomicExecutionStartStore(FakeRpc(row)).mark_started(
            _lease(),
            _claim(),
            worker_token="worker-a",
        )


class _Journal:
    def __init__(self):
        self.stage = "CYCLE_CREATED"

    def latest_stage(self, cycle_id):
        del cycle_id
        return self.stage


class _Checkpoint:
    checkpoint_id = "z" * 64


class _Runtime:
    def __init__(self):
        self.journal = _Journal()

    def checkpoint(self):
        return _Checkpoint()


class _RuntimeSupervisor:
    def __init__(self):
        self.lease = _lease()
        self.persisted_version = 8
        self._valid = True
        self.runtime = _Runtime()

    @property
    def valid(self):
        return self._valid


class _Phase75:
    def __init__(self):
        self.runtime_supervisor = _RuntimeSupervisor()
        self.aborted = False

    def abort_authorized_cycle(self, *, cycle_id, reason):
        assert cycle_id == "c" * 64
        assert reason.startswith("phase79:")
        self.aborted = True
        self.runtime_supervisor.runtime.journal.stage = "ABORTED"
        self.runtime_supervisor.persisted_version += 1


class _Phase76:
    def __init__(self):
        self.governed_supervisor = _Phase75()


class _Claims:
    def __init__(self):
        self.complete_calls = 0

    def complete(self, lease, claim, *, worker_token):
        del lease, claim, worker_token
        self.complete_calls += 1
        return ClaimCompletionReceipt(
            runtime_id="runtime-79",
            cycle_id="c" * 64,
            dispatch_id="d" * 64,
            claim_fencing_token=5,
            status="COMPLETED",
            completed=True,
            duplicate=False,
            completion_checkpoint_id="z" * 64,
        )


class _Phase77:
    def __init__(self, *, outcome="COMMITTED", terminal=None):
        self.dispatched_supervisor = _Phase76()
        self.claims = _Claims()
        self.outcome = outcome
        self.terminal = terminal
        self.advanced = False

    def authorize_submit_and_claim(self, governed, *, worker_token, claim_seconds):
        del governed, worker_token, claim_seconds
        return object(), object(), _claim(), self.terminal

    def advance_claimed(
        self,
        claim,
        *,
        worker_token,
        marks,
        observed_at,
        source_ref,
        complete_on_commit,
    ):
        del claim, worker_token, marks, observed_at, source_ref
        assert complete_on_commit is False
        self.advanced = True
        runtime = self.dispatched_supervisor.governed_supervisor.runtime_supervisor
        runtime.persisted_version += 1
        runtime.runtime.journal.stage = (
            "COMMITTED" if self.outcome == "COMMITTED" else "PAPER_APPLIED"
        )
        return (
            SimpleNamespace(durable_receipt=SimpleNamespace(status=self.outcome)),
            self.outcome,
            None,
        )


class _Starts:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = 0

    def mark_started(self, lease, claim, *, worker_token):
        del lease, claim, worker_token
        self.calls += 1
        return self.receipt


class _Kill:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def check(self, lease, claim, *, worker_token):
        del lease, claim, worker_token
        self.calls += 1
        return self.decision


def _start_receipt(
    *,
    status="STARTED",
    started=True,
    terminal=False,
    cancel=False,
    reason=None,
    resume_only=False,
):
    return AtomicExecutionStartReceipt(
        runtime_id="runtime-79",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=8,
        fencing_token=17,
        claim_fencing_token=5,
        status=status,
        started=started,
        duplicate=False,
        terminal=terminal,
        cancel_requested=cancel,
        risk_version=13 if status not in {"LEASE_LOST", "CLAIM_LOST", "RISK_STATE_UNAVAILABLE"} else None,
        risk_receipt_id=("s" * 64) if status not in {"LEASE_LOST", "CLAIM_LOST", "RISK_STATE_UNAVAILABLE"} else None,
        risk_state=("HALTED" if cancel else "ACTIVE") if status not in {"LEASE_LOST", "CLAIM_LOST", "RISK_STATE_UNAVAILABLE"} else None,
        journal_stage="CYCLE_CREATED",
        phase78_status=(
            "CANCELLED_BEFORE_EXECUTION"
            if status == "CANCELLED_BEFORE_EXECUTION"
            else "PROCEED"
        ),
        reason=reason,
        resume_only=resume_only,
    )


def _decision(
    status="COMPLETED",
    *,
    proceed=False,
    cancel=False,
    terminal=True,
    stage="COMMITTED",
    reason=None,
):
    return ExecutionKillSwitchDecision(
        runtime_id="runtime-79",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=9,
        status=status,
        proceed=proceed,
        cancel_requested=cancel,
        terminal=terminal,
        risk_version=14,
        risk_receipt_id="t" * 64,
        journal_stage=stage,
        reason=reason,
        phase="AFTER_START" if cancel else None,
        completion_checkpoint_id=None,
    )


def test_supervisor_never_advances_when_phase78_atomically_cancels_start() -> None:
    phase77 = _Phase77()
    starts = _Starts(
        _start_receipt(
            status="CANCELLED_BEFORE_EXECUTION",
            started=False,
            terminal=True,
            cancel=True,
            reason="HALTED",
        )
    )
    wrapper = PersistedAtomicStartedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=starts,
        kill_switch=_Kill(_decision()),
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase79-cancel",
    )
    assert step.outcome == "CANCELLED_BEFORE_EXECUTION"
    assert phase77.advanced is False
    assert phase77.dispatched_supervisor.governed_supervisor.aborted is True


def test_supervisor_orders_started_before_advance_and_completes_after_commit() -> None:
    phase77 = _Phase77(outcome="COMMITTED")
    starts = _Starts(_start_receipt())
    kill = _Kill(_decision())
    wrapper = PersistedAtomicStartedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=starts,
        kill_switch=kill,
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase79-commit",
    )
    assert starts.calls == 1
    assert phase77.advanced is True
    assert kill.calls == 1
    assert phase77.claims.complete_calls == 1
    assert step.outcome == "COMMITTED"


def test_post_start_cancel_request_never_aborts_or_rewrites_started_history() -> None:
    phase77 = _Phase77(outcome="MARKS_REQUIRED")
    starts = _Starts(_start_receipt())
    kill = _Kill(
        _decision(
            "CANCEL_REQUESTED",
            proceed=True,
            cancel=True,
            terminal=False,
            stage="RECONCILED",
            reason="HALTED",
        )
    )
    wrapper = PersistedAtomicStartedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=starts,
        kill_switch=kill,
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase79-post-start-kill",
    )
    assert step.outcome == "MARKS_REQUIRED_CANCEL_REQUESTED"
    assert phase77.advanced is True
    assert phase77.dispatched_supervisor.governed_supervisor.aborted is False


def test_post_start_pre_execution_cancel_is_a_history_contradiction() -> None:
    phase77 = _Phase77(outcome="MARKS_REQUIRED")
    wrapper = PersistedAtomicStartedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=_Starts(_start_receipt()),
        kill_switch=_Kill(
            _decision(
                "CANCELLED_BEFORE_EXECUTION",
                proceed=False,
                cancel=True,
                terminal=True,
                stage="CYCLE_CREATED",
                reason="HALTED",
            )
        ),
    )
    with pytest.raises(PersistedRuntimeStaleError, match="contradict"):
        wrapper.process_governed_cycle(
            object(),
            worker_token="worker-a",
            claim_seconds=30,
            marks={},
            observed_at=1.0,
            source_ref="phase79-contradiction",
        )
    runtime = phase77.dispatched_supervisor.governed_supervisor.runtime_supervisor
    assert runtime.valid is False


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("LEASE_LOST", PersistedRuntimeLeaseError),
        ("RISK_STATE_UNAVAILABLE", PersistedRuntimeStaleError),
        ("CLAIM_LOST", ExecutionClaimError),
    ],
)
def test_fail_closed_start_statuses_never_advance(status, error) -> None:
    phase77 = _Phase77()
    start = _start_receipt(
        status=status,
        started=False,
        terminal=False,
    )
    wrapper = PersistedAtomicStartedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=_Starts(start),
        kill_switch=_Kill(_decision()),
    )
    with pytest.raises(error):
        wrapper.process_governed_cycle(
            object(),
            worker_token="worker-a",
            claim_seconds=30,
            marks={},
            observed_at=1.0,
            source_ref="phase79-fail",
        )
    assert phase77.advanced is False
