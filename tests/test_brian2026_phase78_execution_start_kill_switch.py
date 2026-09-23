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
    ClaimCompletionReceipt,
    ExecutionClaimError,
    ExecutionClaimReceipt,
)
from brian2026.phase78_execution_start_kill_switch import (
    ExecutionStartError,
    ExecutionStartKillSwitchStore,
    ExecutionStartReceipt,
    ExecutionKillReceipt,
    PersistedStartedRuntimeSupervisor,
)


def _lease() -> RuntimeLease:
    return RuntimeLease(
        runtime_id="runtime-78",
        owner_token="owner-a",
        fencing_token=13,
        version=6,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )


def _claim() -> ExecutionClaimReceipt:
    return ExecutionClaimReceipt(
        runtime_id="runtime-78",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=6,
        fencing_token=13,
        claim_fencing_token=4,
        status="CLAIMED",
        claimed=True,
        cancelled=False,
        terminal=False,
        risk_version=9,
        risk_receipt_id="r" * 64,
        journal_stage="CYCLE_CREATED",
        resume_only=False,
        cancel_reason=None,
        claim_until="2026-09-23T14:00:00Z",
        completion_checkpoint_id=None,
    )


def _dispatch() -> ShadowExecutionDispatchReceipt:
    return ShadowExecutionDispatchReceipt(
        runtime_id="runtime-78",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        governed_result_id="g" * 64,
        policy_fingerprint="p" * 64,
        authorization_checkpoint_id="k" * 64,
        authorization_runtime_version=6,
        current_runtime_version=6,
        authorization_risk_version=9,
        current_risk_version=9,
        risk_ledger_hash="l" * 64,
        risk_receipt_id="r" * 64,
        fencing_token=13,
        status="SUBMITTED",
        submitted=True,
        duplicate=False,
    )


def _authorization() -> AtomicGovernedWriteAheadReceipt:
    return AtomicGovernedWriteAheadReceipt(
        runtime_id="runtime-78",
        cycle_id="c" * 64,
        checkpoint_id="k" * 64,
        runtime_version_before=5,
        runtime_version_after=6,
        current_runtime_version=6,
        risk_version=9,
        risk_ledger_hash="l" * 64,
        risk_receipt_id="r" * 64,
        governed_result_id="g" * 64,
        policy_fingerprint="p" * 64,
        fencing_token=13,
        status="AUTHORIZED_AND_PERSISTED",
        authorized=True,
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


def _start_row(*, status="STARTED", cancelled=False, duplicate=False):
    return {
        "started": not cancelled,
        "cancelled": cancelled,
        "duplicate": duplicate,
        "status": status,
        "runtime_id": "runtime-78",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "runtime_version": 6,
        "fencing_token": 13,
        "claim_fencing_token": 4,
        "risk_version": 9,
        "risk_receipt_id": "s" * 64,
        "risk_state": "ACTIVE" if not cancelled else "HALTED",
        "journal_stage": "CYCLE_CREATED",
        "resume_only": False,
        "cancel_reason": "HALTED" if cancelled else None,
    }


def _kill_row(*, requested=False, status="CONTINUE"):
    return {
        "kill_requested": requested,
        "status": status,
        "reason": "HALTED" if requested else None,
        "request_sequence": 1 if requested else None,
        "runtime_id": "runtime-78",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "runtime_version": 6,
        "fencing_token": 13,
        "claim_fencing_token": 4,
        "risk_version": 10,
        "risk_receipt_id": "t" * 64,
        "risk_state": "HALTED" if requested else "ACTIVE",
        "journal_stage": "PAPER_APPLIED",
        "start_risk_version": 9,
        "start_risk_receipt_id": "s" * 64,
    }


def test_start_store_parses_point_of_no_return_and_exact_rpc_contract() -> None:
    rpc = FakeRpc({"brian_start_shadow_execution_claim": _start_row()})
    start = ExecutionStartKillSwitchStore(rpc).start(
        _lease(),
        _claim(),
        worker_token="worker-a",
    )

    assert start.started is True
    assert start.cancelled is False
    assert start.status == "STARTED"
    assert start.risk_version == 9
    assert start.risk_receipt_id == "s" * 64
    assert rpc.calls[0][1]["p_claim_fencing_token"] == 4
    assert rpc.calls[0][1]["p_worker_token"] == "worker-a"


def test_start_store_parses_pre_start_risk_cancellation() -> None:
    rpc = FakeRpc({
        "brian_start_shadow_execution_claim": _start_row(
            status="CANCELLED_BEFORE_START",
            cancelled=True,
        )
    })
    start = ExecutionStartKillSwitchStore(rpc).start(
        _lease(),
        _claim(),
        worker_token="worker-a",
    )
    assert start.started is False
    assert start.cancelled is True
    assert start.cancel_reason == "HALTED"


def test_start_store_rejects_duplicate_flag_without_started_state() -> None:
    row = _start_row(status="STARTED_ALREADY", duplicate=True)
    row["started"] = False
    with pytest.raises(ExecutionStartError):
        ExecutionStartKillSwitchStore(
            FakeRpc({"brian_start_shadow_execution_claim": row})
        ).start(
            _lease(),
            _claim(),
            worker_token="worker-a",
        )


def test_kill_check_is_contentful_and_idempotent_identity_is_db_owned() -> None:
    rpc = FakeRpc({
        "brian_check_shadow_execution_kill_switch": _kill_row(
            requested=True,
            status="KILL_REQUESTED",
        )
    })
    kill = ExecutionStartKillSwitchStore(rpc).check_kill(
        _lease(),
        _claim(),
        worker_token="worker-a",
    )
    assert kill.kill_requested is True
    assert kill.reason == "HALTED"
    assert kill.request_sequence == 1
    assert kill.start_risk_version == 9
    assert kill.risk_version == 10


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
    def __init__(self, completion=None):
        self.completion = completion
        self.complete_calls = 0

    def complete(self, lease, claim, *, worker_token):
        del lease, claim, worker_token
        self.complete_calls += 1
        return self.completion


class _Phase77:
    def __init__(self, *, outcome="RECONCILIATION_BLOCKED", terminal=None):
        self.dispatched_supervisor = _Phase76()
        self.claims = _Claims(
            ClaimCompletionReceipt(
                runtime_id="runtime-78",
                cycle_id="c" * 64,
                dispatch_id="d" * 64,
                claim_fencing_token=4,
                status="COMPLETED",
                completed=True,
                duplicate=False,
                completion_checkpoint_id="z" * 64,
            )
        )
        self.outcome = outcome
        self.terminal = terminal
        self.advanced = False

    def authorize_submit_and_claim(self, governed, *, worker_token, claim_seconds):
        del governed, worker_token, claim_seconds
        return _authorization(), _dispatch(), _claim(), self.terminal

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
        if self.outcome == "COMMITTED":
            self.dispatched_supervisor.governed_supervisor.runtime_supervisor.runtime.journal.stage = "COMMITTED"
            self.dispatched_supervisor.governed_supervisor.runtime_supervisor.persisted_version += 1
        else:
            self.dispatched_supervisor.governed_supervisor.runtime_supervisor.runtime.journal.stage = "PAPER_APPLIED"
            self.dispatched_supervisor.governed_supervisor.runtime_supervisor.persisted_version += 1
        return SimpleNamespace(durable_receipt=SimpleNamespace(status=self.outcome)), self.outcome, None


class _Starts:
    def __init__(self, start, kill):
        self.start_receipt = start
        self.kill_receipt = kill
        self.start_calls = 0
        self.kill_calls = 0

    def start(self, lease, claim, *, worker_token):
        del lease, claim, worker_token
        self.start_calls += 1
        return self.start_receipt

    def check_kill(self, lease, claim, *, worker_token):
        del lease, claim, worker_token
        self.kill_calls += 1
        return self.kill_receipt


def _start_receipt(*, cancelled=False, status=None):
    return ExecutionStartReceipt(
        runtime_id="runtime-78",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=6,
        fencing_token=13,
        claim_fencing_token=4,
        status=status or ("CANCELLED_BEFORE_START" if cancelled else "STARTED"),
        started=not cancelled,
        cancelled=cancelled,
        duplicate=False,
        risk_version=10,
        risk_receipt_id="s" * 64,
        risk_state="HALTED" if cancelled else "ACTIVE",
        journal_stage="CYCLE_CREATED",
        resume_only=False,
        cancel_reason="HALTED" if cancelled else None,
    )


def _kill_receipt(*, requested=False, status=None, journal_stage="PAPER_APPLIED"):
    return ExecutionKillReceipt(
        runtime_id="runtime-78",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=7,
        fencing_token=13,
        claim_fencing_token=4,
        status=status or ("KILL_REQUESTED" if requested else "CONTINUE"),
        kill_requested=requested,
        reason="HALTED" if requested else None,
        request_sequence=1 if requested else None,
        risk_version=11,
        risk_receipt_id="t" * 64,
        risk_state="HALTED" if requested else "ACTIVE",
        journal_stage=journal_stage,
        start_risk_version=10,
        start_risk_receipt_id="s" * 64,
    )


def test_supervisor_cancels_at_final_start_gate_without_advancing_side_effects() -> None:
    phase77 = _Phase77()
    starts = _Starts(
        _start_receipt(cancelled=True),
        _kill_receipt(),
    )
    supervisor = PersistedStartedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=starts,
    )

    step = supervisor.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase78-prestart-cancel",
    )
    assert step.outcome == "CANCELLED_BEFORE_START"
    assert phase77.advanced is False
    assert phase77.dispatched_supervisor.governed_supervisor.aborted is True
    assert starts.kill_calls == 0


def test_post_start_halt_becomes_kill_request_not_fake_rollback() -> None:
    phase77 = _Phase77(outcome="RECONCILIATION_BLOCKED")
    starts = _Starts(
        _start_receipt(),
        _kill_receipt(requested=True),
    )
    supervisor = PersistedStartedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=starts,
    )

    step = supervisor.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase78-kill",
    )
    assert step.outcome == "KILL_REQUESTED_RECONCILIATION_BLOCKED"
    assert phase77.advanced is True
    assert phase77.dispatched_supervisor.governed_supervisor.aborted is False
    assert step.kill is not None and step.kill.kill_requested is True
    assert phase77.claims.complete_calls == 0


def test_committed_cycle_completes_claim_and_does_not_turn_post_commit_risk_into_rollback() -> None:
    phase77 = _Phase77(outcome="COMMITTED")
    starts = _Starts(
        _start_receipt(),
        _kill_receipt(
            requested=False,
            status="COMMITTED",
            journal_stage="COMMITTED",
        ),
    )
    supervisor = PersistedStartedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=starts,
    )

    step = supervisor.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase78-commit",
    )
    assert step.outcome == "COMMITTED"
    assert step.completion is not None and step.completion.completed is True
    assert phase77.claims.complete_calls == 1
    assert phase77.dispatched_supervisor.governed_supervisor.aborted is False


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("LEASE_LOST", PersistedRuntimeLeaseError),
        ("DISPATCH_MISSING", PersistedRuntimeStaleError),
        ("RISK_STATE_UNAVAILABLE", PersistedRuntimeStaleError),
    ],
)
def test_fail_closed_start_statuses_never_advance(status, error) -> None:
    phase77 = _Phase77()
    start = _start_receipt()
    start = ExecutionStartReceipt(
        runtime_id=start.runtime_id,
        cycle_id=start.cycle_id,
        dispatch_id=start.dispatch_id if status != "LEASE_LOST" else "",
        runtime_version=start.runtime_version,
        fencing_token=start.fencing_token,
        claim_fencing_token=start.claim_fencing_token,
        status=status,
        started=False,
        cancelled=False,
        duplicate=False,
        risk_version=None,
        risk_receipt_id=None,
        risk_state=None,
        journal_stage="CYCLE_CREATED",
        resume_only=False,
        cancel_reason=None,
    )
    supervisor = PersistedStartedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=_Starts(start, _kill_receipt()),
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
    assert phase77.advanced is False


def test_claim_loss_at_start_requires_reclaim_but_does_not_forge_runtime_staleness() -> None:
    phase77 = _Phase77()
    base = _start_receipt()
    lost = ExecutionStartReceipt(
        runtime_id=base.runtime_id,
        cycle_id=base.cycle_id,
        dispatch_id=base.dispatch_id,
        runtime_version=base.runtime_version,
        fencing_token=base.fencing_token,
        claim_fencing_token=base.claim_fencing_token,
        status="CLAIM_LOST",
        started=False,
        cancelled=False,
        duplicate=False,
        risk_version=None,
        risk_receipt_id=None,
        risk_state=None,
        journal_stage="CYCLE_CREATED",
        resume_only=False,
        cancel_reason=None,
    )
    supervisor = PersistedStartedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=_Starts(lost, _kill_receipt()),
    )
    with pytest.raises(ExecutionClaimError, match="acquire a new claim"):
        supervisor.process_governed_cycle(
            object(),
            worker_token="worker-a",
            claim_seconds=30,
            marks={},
            observed_at=1.0,
            source_ref="phase78-claim-lost",
        )
    runtime_supervisor = (
        phase77.dispatched_supervisor.governed_supervisor.runtime_supervisor
    )
    assert runtime_supervisor.valid is True
    assert phase77.advanced is False
