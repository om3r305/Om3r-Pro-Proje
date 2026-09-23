from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase70_durable_runtime_store import RuntimeLease
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from brian2026.phase77_execution_claim_lifecycle import (
    ExecutionClaimReceipt,
)
from brian2026.phase78_execution_kill_switch import (
    ExecutionKillSwitchDecision,
    ExecutionKillSwitchError,
    ExecutionKillSwitchStore,
    PersistedKillSwitchRuntimeSupervisor,
)


def _lease():
    return RuntimeLease(
        runtime_id="runtime-78",
        owner_token="owner-a",
        fencing_token=12,
        version=6,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )


def _claim():
    return ExecutionClaimReceipt(
        runtime_id="runtime-78",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=6,
        fencing_token=12,
        claim_fencing_token=4,
        status="CLAIMED",
        claimed=True,
        cancelled=False,
        terminal=False,
        risk_version=8,
        risk_receipt_id="r" * 64,
        journal_stage="CYCLE_CREATED",
        resume_only=False,
        cancel_reason=None,
        claim_until=None,
        completion_checkpoint_id=None,
    )


class FakeRpc:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, name, params):
        self.calls.append((name, dict(params)))
        value = self.responses.pop(0)
        return value(dict(params)) if callable(value) else dict(value)


def _decision_row(
    status="PROCEED",
    *,
    proceed=True,
    cancel=False,
    terminal=False,
    stage="CYCLE_CREATED",
    reason=None,
):
    return {
        "status": status,
        "proceed": proceed,
        "cancel_requested": cancel,
        "terminal": terminal,
        "runtime_id": "runtime-78",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "runtime_version": 6,
        "risk_version": 9,
        "risk_receipt_id": "x" * 64,
        "journal_stage": stage,
        "reason": reason,
        "phase": (
            "BEFORE_EXECUTION"
            if status == "CANCELLED_BEFORE_EXECUTION"
            else "AFTER_START" if status == "CANCEL_REQUESTED" else None
        ),
    }


def test_kill_switch_store_parses_proceed_and_cancel_request_contracts() -> None:
    rpc = FakeRpc([
        _decision_row(),
        _decision_row(
            "CANCEL_REQUESTED",
            proceed=True,
            cancel=True,
            stage="PAPER_APPLIED",
            reason="HALTED",
        ),
    ])
    store = ExecutionKillSwitchStore(rpc)

    proceed = store.check(_lease(), _claim(), worker_token="worker-a")
    cancel = store.check(_lease(), _claim(), worker_token="worker-a")

    assert proceed.status == "PROCEED"
    assert proceed.proceed is True
    assert cancel.status == "CANCEL_REQUESTED"
    assert cancel.cancel_requested is True
    assert cancel.proceed is True
    assert cancel.terminal is False
    assert cancel.reason == "HALTED"
    assert rpc.calls[0][1]["p_claim_fencing_token"] == 4


def test_kill_switch_rejects_inconsistent_database_flags() -> None:
    row = _decision_row("CANCEL_REQUESTED", proceed=False, cancel=True)
    with pytest.raises(ExecutionKillSwitchError, match="flags"):
        ExecutionKillSwitchStore(FakeRpc([row])).check(
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
        self.persisted_version = 6
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
        assert reason.startswith("phase78:")
        self.aborted = True
        self.runtime_supervisor.runtime.journal.stage = "ABORTED"
        self.runtime_supervisor.persisted_version += 1


class _Phase76:
    def __init__(self):
        self.governed_supervisor = _Phase75()


class _Claims:
    def __init__(self):
        self.completed = 0

    def complete(self, lease, claim, *, worker_token):
        del lease, claim, worker_token
        self.completed += 1
        return SimpleNamespace(
            completed=True,
            status="COMPLETED",
        )


class _Phase77:
    def __init__(self, *, advance_stage="COMMITTED", advance_outcome="COMMITTED"):
        self.dispatched_supervisor = _Phase76()
        self.claims = _Claims()
        self.advance_stage = advance_stage
        self.advance_outcome = advance_outcome
        self.advanced = False

    def authorize_submit_and_claim(self, governed, *, worker_token, claim_seconds):
        del governed, worker_token, claim_seconds
        return object(), object(), _claim(), None

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
        self.dispatched_supervisor.governed_supervisor.runtime_supervisor.runtime.journal.stage = self.advance_stage
        return (
            SimpleNamespace(durable_receipt=SimpleNamespace(status=self.advance_outcome)),
            self.advance_outcome,
            None,
        )


class _KillStore:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.calls = 0

    def check(self, lease, claim, *, worker_token):
        del lease, claim, worker_token
        self.calls += 1
        return self.decisions.pop(0)


def _decision(
    status,
    *,
    proceed,
    cancel=False,
    terminal=False,
    stage="CYCLE_CREATED",
    reason=None,
):
    return ExecutionKillSwitchDecision(
        runtime_id="runtime-78",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=6,
        status=status,
        proceed=proceed,
        cancel_requested=cancel,
        terminal=terminal,
        risk_version=9,
        risk_receipt_id="x" * 64,
        journal_stage=stage,
        reason=reason,
        phase=(
            "BEFORE_EXECUTION" if status == "CANCELLED_BEFORE_EXECUTION"
            else "AFTER_START" if status == "CANCEL_REQUESTED"
            else None
        ),
        completion_checkpoint_id=None,
    )


def test_pre_execution_halt_aborts_without_advancing_paper() -> None:
    phase77 = _Phase77()
    kill = _KillStore(
        _decision(
            "CANCELLED_BEFORE_EXECUTION",
            proceed=False,
            cancel=True,
            terminal=True,
            reason="HALTED",
        )
    )
    supervisor = PersistedKillSwitchRuntimeSupervisor(
        claimed_supervisor=phase77,
        kill_switch=kill,
    )
    step = supervisor.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase78-pre-halt",
    )
    assert step.outcome == "CANCELLED_BEFORE_EXECUTION"
    assert phase77.advanced is False
    assert phase77.dispatched_supervisor.governed_supervisor.aborted is True
    assert phase77.claims.completed == 0


def test_healthy_precheck_then_commit_completes_claim() -> None:
    phase77 = _Phase77()
    kill = _KillStore(
        _decision("PROCEED", proceed=True),
        _decision(
            "COMPLETED",
            proceed=False,
            terminal=True,
            stage="COMMITTED",
        ),
    )
    supervisor = PersistedKillSwitchRuntimeSupervisor(
        claimed_supervisor=phase77,
        kill_switch=kill,
    )
    step = supervisor.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase78-healthy",
    )
    assert phase77.advanced is True
    assert phase77.claims.completed == 1
    assert step.outcome == "COMMITTED"
    assert step.postcheck.status == "COMPLETED"


def test_risk_change_after_paper_start_records_cancel_request_but_keeps_recovery() -> None:
    phase77 = _Phase77(
        advance_stage="RECONCILED",
        advance_outcome="MARKS_REQUIRED",
    )
    kill = _KillStore(
        _decision("PROCEED", proceed=True),
        _decision(
            "CANCEL_REQUESTED",
            proceed=True,
            cancel=True,
            stage="RECONCILED",
            reason="HALTED",
        ),
    )
    supervisor = PersistedKillSwitchRuntimeSupervisor(
        claimed_supervisor=phase77,
        kill_switch=kill,
    )
    step = supervisor.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase78-post-start",
    )
    assert phase77.advanced is True
    assert phase77.claims.completed == 0
    assert step.outcome == "MARKS_REQUIRED_CANCEL_REQUESTED"
    assert step.postcheck.cancel_requested is True


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("LEASE_LOST", PersistedRuntimeLeaseError),
        ("CLAIM_LOST", PersistedRuntimeStaleError),
        ("RISK_STATE_UNAVAILABLE", PersistedRuntimeStaleError),
    ],
)
def test_fail_closed_precheck_invalidates_supervisor(status, error) -> None:
    phase77 = _Phase77()
    kill = _KillStore(
        _decision(status, proceed=False)
    )
    supervisor = PersistedKillSwitchRuntimeSupervisor(
        claimed_supervisor=phase77,
        kill_switch=kill,
    )
    with pytest.raises(error):
        supervisor.process_governed_cycle(
            object(),
            worker_token="worker-a",
            claim_seconds=30,
            marks={},
            observed_at=1.0,
            source_ref="phase78-fail",
        )
    runtime_supervisor = (
        phase77.dispatched_supervisor.governed_supervisor.runtime_supervisor
    )
    assert runtime_supervisor.valid is False
    assert phase77.advanced is False


def test_resume_only_precheck_is_allowed_to_finish_existing_side_effects() -> None:
    phase77 = _Phase77()
    kill = _KillStore(
        _decision(
            "RESUME_ONLY",
            proceed=True,
            stage="PAPER_APPLIED",
        ),
        _decision(
            "COMPLETED",
            proceed=False,
            terminal=True,
            stage="COMMITTED",
        ),
    )
    supervisor = PersistedKillSwitchRuntimeSupervisor(
        claimed_supervisor=phase77,
        kill_switch=kill,
    )
    step = supervisor.process_governed_cycle(
        object(),
        worker_token="worker-recover",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase78-resume",
    )
    assert step.precheck.status == "RESUME_ONLY"
    assert phase77.advanced is True
    assert step.outcome == "COMMITTED"
