from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase70_durable_runtime_store import RuntimeLease
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from brian2026.phase75_atomic_governed_writeahead import AtomicGovernedWriteAheadReceipt
from brian2026.phase76_shadow_execution_outbox import ShadowExecutionDispatchReceipt
from brian2026.phase77_execution_claim_lifecycle import (
    ExecutionClaimBusyError,
    ExecutionClaimError,
    ExecutionClaimReceipt,
    ExecutionClaimStore,
    PersistedClaimedRuntimeSupervisor,
)


def _lease() -> RuntimeLease:
    return RuntimeLease(
        runtime_id="runtime-77",
        owner_token="owner-a",
        fencing_token=11,
        version=5,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )


def _auth() -> AtomicGovernedWriteAheadReceipt:
    return AtomicGovernedWriteAheadReceipt(
        runtime_id="runtime-77",
        cycle_id="c" * 64,
        checkpoint_id="k" * 64,
        runtime_version_before=4,
        runtime_version_after=5,
        current_runtime_version=5,
        risk_version=7,
        risk_ledger_hash="l" * 64,
        risk_receipt_id="r" * 64,
        governed_result_id="g" * 64,
        policy_fingerprint="p" * 64,
        fencing_token=11,
        status="AUTHORIZED_AND_PERSISTED",
        authorized=True,
        duplicate=False,
    )


def _dispatch() -> ShadowExecutionDispatchReceipt:
    auth = _auth()
    return ShadowExecutionDispatchReceipt(
        runtime_id=auth.runtime_id,
        cycle_id=auth.cycle_id,
        dispatch_id="d" * 64,
        governed_result_id=auth.governed_result_id,
        policy_fingerprint=auth.policy_fingerprint,
        authorization_checkpoint_id=auth.checkpoint_id,
        authorization_runtime_version=auth.runtime_version_after,
        current_runtime_version=auth.runtime_version_after,
        authorization_risk_version=auth.risk_version,
        current_risk_version=auth.risk_version,
        risk_ledger_hash=auth.risk_ledger_hash,
        risk_receipt_id=auth.risk_receipt_id,
        fencing_token=auth.fencing_token,
        status="SUBMITTED",
        submitted=True,
        duplicate=False,
    )


class FakeRpc:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.calls = []

    def __call__(self, name, params):
        self.calls.append((name, dict(params)))
        value = self.responses[name]
        return value(dict(params)) if callable(value) else dict(value)


def _claimed_row(*, status="CLAIMED", resume_only=False):
    return {
        "claimed": True,
        "cancelled": False,
        "terminal": False,
        "status": status,
        "runtime_id": "runtime-77",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "runtime_version": 5,
        "fencing_token": 11,
        "claim_fencing_token": 3,
        "claim_until": "2026-09-23T13:00:00Z",
        "risk_version": 8,
        "risk_receipt_id": "x" * 64,
        "journal_stage": "PAPER_APPLIED" if resume_only else "CYCLE_CREATED",
        "resume_only": resume_only,
    }


def test_claim_parses_fenced_worker_ownership() -> None:
    rpc = FakeRpc({"brian_claim_shadow_execution_dispatch": _claimed_row()})
    claim = ExecutionClaimStore(rpc).claim(
        _lease(),
        cycle_id="c" * 64,
        worker_token="worker-a",
        claim_seconds=30,
    )
    assert claim.claimed is True
    assert claim.claim_fencing_token == 3
    assert claim.risk_version == 8
    assert claim.resume_only is False
    assert rpc.calls[0][1]["p_worker_token"] == "worker-a"


def test_claim_parses_risk_cancellation_as_terminal() -> None:
    rpc = FakeRpc({
        "brian_claim_shadow_execution_dispatch": {
            "claimed": False,
            "cancelled": True,
            "terminal": True,
            "status": "CANCELLED_BEFORE_EXECUTION",
            "cancel_reason": "HALTED",
            "runtime_id": "runtime-77",
            "dispatch_id": "d" * 64,
            "cycle_id": "c" * 64,
            "runtime_version": 5,
            "fencing_token": 11,
            "claim_fencing_token": 0,
            "risk_version": 9,
            "risk_receipt_id": "h" * 64,
            "journal_stage": "CYCLE_CREATED",
            "resume_only": False,
        }
    })
    claim = ExecutionClaimStore(rpc).claim(
        _lease(),
        cycle_id="c" * 64,
        worker_token="worker-a",
        claim_seconds=30,
    )
    assert claim.claimed is False
    assert claim.cancelled is True
    assert claim.terminal is True
    assert claim.cancel_reason == "HALTED"


def test_claim_renew_and_completion_preserve_claim_fence() -> None:
    claim_rpc = FakeRpc({
        "brian_claim_shadow_execution_dispatch": _claimed_row(),
        "brian_renew_shadow_execution_claim": {
            "renewed": True,
            "status": "RENEWED",
            "runtime_id": "runtime-77",
            "dispatch_id": "d" * 64,
            "cycle_id": "c" * 64,
            "fencing_token": 11,
            "claim_fencing_token": 3,
            "claim_until": "2026-09-23T13:01:00Z",
        },
        "brian_complete_shadow_execution_claim": {
            "completed": True,
            "duplicate": False,
            "status": "COMPLETED",
            "runtime_id": "runtime-77",
            "dispatch_id": "d" * 64,
            "cycle_id": "c" * 64,
            "claim_fencing_token": 3,
            "completion_checkpoint_id": "z" * 64,
        },
    })
    store = ExecutionClaimStore(claim_rpc)
    claim = store.claim(
        _lease(),
        cycle_id="c" * 64,
        worker_token="worker-a",
        claim_seconds=30,
    )
    renewed = store.renew(
        _lease(),
        claim,
        worker_token="worker-a",
        claim_seconds=60,
    )
    completed = store.complete(
        _lease(),
        claim,
        worker_token="worker-a",
    )
    assert renewed.renewed is True
    assert renewed.claim_fencing_token == 3
    assert completed.completed is True
    assert completed.completion_checkpoint_id == "z" * 64
    assert claim_rpc.calls[1][1]["p_claim_fencing_token"] == 3
    assert claim_rpc.calls[2][1]["p_claim_fencing_token"] == 3


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
        self.persisted_version = 5
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
        assert reason.startswith("phase77:")
        self.aborted = True
        self.runtime_supervisor.runtime.journal.stage = "ABORTED"
        self.runtime_supervisor.persisted_version = 6


class _Phase76:
    def __init__(self):
        self.governed_supervisor = _Phase75()
        self.advanced = False

    def authorize_and_submit(self, governed):
        del governed
        return _auth(), _dispatch(), None

    def advance_submitted(self, *, marks, observed_at, source_ref):
        del marks, observed_at, source_ref
        self.advanced = True
        self.governed_supervisor.runtime_supervisor.runtime.journal.stage = "COMMITTED"
        self.governed_supervisor.runtime_supervisor.persisted_version = 6
        return SimpleNamespace(
            durable_receipt=SimpleNamespace(status="COMMITTED"),
        )


class _ClaimStore:
    def __init__(self, claim, completion=None):
        self.claim_receipt = claim
        self.completion = completion
        self.complete_calls = 0

    def claim(self, lease, *, cycle_id, worker_token, claim_seconds):
        del lease, worker_token, claim_seconds
        assert cycle_id == "c" * 64
        return self.claim_receipt

    def complete(self, lease, claim, *, worker_token):
        del lease, claim, worker_token
        self.complete_calls += 1
        return self.completion


def _claim_receipt(
    *,
    status="CLAIMED",
    claimed=True,
    cancelled=False,
    terminal=False,
    reason=None,
):
    return ExecutionClaimReceipt(
        runtime_id="runtime-77",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=5,
        fencing_token=11,
        claim_fencing_token=3 if claimed else 0,
        status=status,
        claimed=claimed,
        cancelled=cancelled,
        terminal=terminal,
        risk_version=8,
        risk_receipt_id="x" * 64,
        journal_stage="CYCLE_CREATED",
        resume_only=False,
        cancel_reason=reason,
        claim_until=None,
        completion_checkpoint_id=None,
    )


def test_supervisor_cancels_before_execution_without_advancing() -> None:
    phase76 = _Phase76()
    claims = _ClaimStore(
        _claim_receipt(
            status="CANCELLED_BEFORE_EXECUTION",
            claimed=False,
            cancelled=True,
            terminal=True,
            reason="HALTED",
        )
    )
    supervisor = PersistedClaimedRuntimeSupervisor(
        dispatched_supervisor=phase76,
        claims=claims,
    )
    step = supervisor.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase77-cancel",
    )
    assert step.outcome == "CANCELLED_BEFORE_EXECUTION"
    assert phase76.governed_supervisor.aborted is True
    assert phase76.advanced is False
    assert claims.complete_calls == 0


def test_supervisor_blocks_second_worker_without_invalidating_runtime() -> None:
    phase76 = _Phase76()
    claims = _ClaimStore(
        _claim_receipt(
            status="BLOCKED_ACTIVE",
            claimed=False,
        )
    )
    supervisor = PersistedClaimedRuntimeSupervisor(
        dispatched_supervisor=phase76,
        claims=claims,
    )
    with pytest.raises(ExecutionClaimBusyError, match="another worker"):
        supervisor.process_governed_cycle(
            object(),
            worker_token="worker-b",
            claim_seconds=30,
            marks={},
            observed_at=1.0,
            source_ref="phase77-busy",
        )
    assert phase76.governed_supervisor.runtime_supervisor.valid is True
    assert phase76.advanced is False


def test_supervisor_executes_only_after_claim_and_completes_after_commit() -> None:
    phase76 = _Phase76()
    completion = SimpleNamespace(
        completed=True,
        status="COMPLETED",
    )
    claims = _ClaimStore(_claim_receipt(), completion=completion)
    supervisor = PersistedClaimedRuntimeSupervisor(
        dispatched_supervisor=phase76,
        claims=claims,
    )
    step = supervisor.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase77-success",
    )
    assert phase76.advanced is True
    assert claims.complete_calls == 1
    assert step.outcome == "COMMITTED"
    assert step.completion is completion


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("LEASE_LOST", PersistedRuntimeLeaseError),
        ("DISPATCH_MISSING", PersistedRuntimeStaleError),
        ("RISK_STATE_UNAVAILABLE", PersistedRuntimeStaleError),
    ],
)
def test_fail_closed_claim_status_invalidates_supervisor(status, error) -> None:
    phase76 = _Phase76()
    claims = _ClaimStore(
        _claim_receipt(status=status, claimed=False)
    )
    supervisor = PersistedClaimedRuntimeSupervisor(
        dispatched_supervisor=phase76,
        claims=claims,
    )
    with pytest.raises(error):
        supervisor.process_governed_cycle(
            object(),
            worker_token="worker-a",
            claim_seconds=30,
            marks={},
            observed_at=1.0,
            source_ref="phase77-fail",
        )
    assert phase76.governed_supervisor.runtime_supervisor.valid is False
    assert phase76.advanced is False


def test_runtime_committed_recovery_does_not_execute_again() -> None:
    phase76 = _Phase76()
    phase76.governed_supervisor.runtime_supervisor.runtime.journal.stage = "COMMITTED"
    claims = _ClaimStore(
        _claim_receipt(
            status="COMPLETED",
            claimed=False,
            terminal=True,
        )
    )
    supervisor = PersistedClaimedRuntimeSupervisor(
        dispatched_supervisor=phase76,
        claims=claims,
    )
    step = supervisor.process_governed_cycle(
        object(),
        worker_token="worker-recover",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase77-recovered",
    )
    assert step.outcome == "ALREADY_COMPLETED"
    assert phase76.advanced is False
