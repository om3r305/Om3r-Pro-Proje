from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase70_durable_runtime_store import RuntimeLease
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedRuntimeStaleError,
)
from brian2026.phase81_cancel_recovery_directive import (
    CancelRecoveryDirectiveReceipt,
    CancelRecoveryLeg,
)
from brian2026.phase82_recovery_claim_fencing import RecoveryClaimReceipt
from brian2026.phase83_atomic_recovery_start import AtomicRecoveryStartReceipt
from brian2026.phase84_recovery_execution_checkpoint import (
    RecoveryCheckpointReceipt,
    RecoveryExecutionCoreStep,
)
from brian2026.phase85_recovery_completion_audit import (
    RecoveryCompletionAuditError,
    RecoveryCompletionAuditReceipt,
)
from brian2026.phase87_recovery_restart_resume import RecoveryRestartWorkItem
from brian2026.phase88_recovery_restart_orchestrator import (
    RecoveryRestartOrchestrationError,
    RecoveryRestartOrchestrator,
)


RUNTIME_ID = "runtime-88"
ORIGINAL = "o" * 64
DISPATCH = "d" * 64
CANCEL = "r" * 64
CHECKPOINT = "k" * 64
HEAD = "h" * 64
RECOVERY = "y" * 64
PROGRESS = "p" * 64


def _lease() -> RuntimeLease:
    return RuntimeLease(
        runtime_id=RUNTIME_ID,
        owner_token="owner-a",
        fencing_token=71,
        version=10,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )


class _Checkpoint:
    checkpoint_id = CHECKPOINT


class _Runtime:
    def __init__(self):
        self.ledger = SimpleNamespace(
            head_state=SimpleNamespace(state_id=HEAD)
        )

    def checkpoint(self):
        return _Checkpoint()


class _RuntimeSupervisor:
    def __init__(self):
        self.runtime_id = RUNTIME_ID
        self.lease = _lease()
        self.persisted_version = 10
        self.runtime = _Runtime()
        self._valid = True

    @property
    def valid(self):
        return self._valid


def _work(state="NEEDS_CLAIM", **overrides):
    values = dict(
        runtime_id=RUNTIME_ID,
        has_work=True,
        status="WORK",
        work_state=state,
        original_cycle_id=ORIGINAL,
        dispatch_id=DISPATCH,
        cancel_risk_version=7,
        cancel_risk_receipt_id=CANCEL,
        cancel_reason="REDUCING_NEW_RISK",
        requested_at="2026-09-23T12:00:00Z",
        runtime_version=10,
        runtime_checkpoint_id=CHECKPOINT,
        runtime_head_state_id=HEAD,
        directive_exists=True,
        recovery_status="READY_REDUCE_ONLY",
        directive_runtime_version=10,
        directive_state_id=HEAD,
        directive_prepared_at="2026-09-23T12:00:01Z",
        claim_status=None,
        claim_worker_token=None,
        claim_fencing_token=None,
        claim_until=None,
        recovery_cycle_id=None,
        progress_runtime_version=None,
        progress_head_state_id=None,
        progress_checkpoint_id=None,
        started=False,
        started_at=None,
        recovery_journal_stage=None,
        phase84_terminal_event=False,
    )
    if state == "NEEDS_DIRECTIVE":
        values.update(
            directive_exists=False,
            recovery_status=None,
            directive_runtime_version=None,
            directive_state_id=None,
            directive_prepared_at=None,
        )
    elif state == "MANUAL_REVIEW":
        values["recovery_status"] = "MANUAL_REVIEW"
    elif state == "COMPLETED_WITHOUT_CERTIFICATE":
        values.update(
            claim_status="COMPLETED",
            claim_worker_token="old-worker",
            claim_fencing_token=3,
        )
    elif state == "CLAIM_EXPIRED":
        values.update(
            claim_status="CLAIMED",
            claim_worker_token="old-worker",
            claim_fencing_token=3,
            claim_until="2026-09-23T11:00:00Z",
        )
    elif state == "NEEDS_START":
        values.update(
            claim_status="CLAIMED",
            claim_worker_token="recovery-a",
            claim_fencing_token=3,
            claim_until="2026-09-23T13:00:00Z",
        )
    elif state == "STARTED_NEEDS_EXECUTION":
        values.update(
            claim_status="CLAIMED",
            claim_worker_token="recovery-a",
            claim_fencing_token=3,
            claim_until="2026-09-23T13:00:00Z",
            started=True,
            started_at="2026-09-23T12:01:00Z",
        )
    elif state in {"RECOVERY_PROGRESS", "NEEDS_AUDIT"}:
        values.update(
            claim_status="CLAIMED",
            claim_worker_token="recovery-a",
            claim_fencing_token=3,
            claim_until="2026-09-23T13:00:00Z",
            started=True,
            started_at="2026-09-23T12:01:00Z",
            recovery_cycle_id=RECOVERY,
            progress_runtime_version=10,
            progress_head_state_id=HEAD,
            progress_checkpoint_id=PROGRESS,
            recovery_journal_stage=(
                "COMMITTED" if state == "NEEDS_AUDIT" else "PAPER_APPLIED"
            ),
            phase84_terminal_event=state == "NEEDS_AUDIT",
        )
    values.update(overrides)
    return RecoveryRestartWorkItem(**values)


def _idle():
    return RecoveryRestartWorkItem(
        runtime_id=RUNTIME_ID,
        has_work=False,
        status="IDLE",
        work_state="IDLE",
    )


def _leg():
    return CancelRecoveryLeg(
        asset_id="BTCUSDT",
        before_weight=0.10,
        current_weight=0.25,
        target_weight=0.10,
        reduce_weight=0.15,
        current_direction=1,
        order_direction=-1,
        reduce_only=True,
    )


def _directive(*, recovery_status="READY_REDUCE_ONLY", status="PREPARED", prepared=True, **kw):
    return CancelRecoveryDirectiveReceipt(
        runtime_id=RUNTIME_ID,
        cycle_id=ORIGINAL,
        dispatch_id=DISPATCH,
        runtime_version=10,
        fencing_token=71,
        status=status,
        prepared=prepared,
        duplicate=status == "DUPLICATE",
        cancel_risk_version=7 if prepared else None,
        cancel_risk_receipt_id=CANCEL if prepared else None,
        cancel_reason="REDUCING_NEW_RISK" if prepared else None,
        pre_state_id="a" * 64 if prepared else None,
        current_state_id=HEAD if prepared else None,
        current_risk_version=8 if prepared else None,
        current_risk_receipt_id="s" * 64 if prepared else None,
        current_risk_state=(
            "HALTED" if recovery_status == "WAIT_RISK_RELEASE" else "REDUCING"
        ) if prepared else None,
        recovery_status=recovery_status if prepared else None,
        recovery_legs=(
            ()
            if recovery_status == "NO_RECOVERY_REQUIRED" or not prepared
            else (_leg(),)
        ),
        unsafe_assets=(
            ({"asset_id": "BTCUSDT", "reason": "ROLLBACK_NOT_REDUCE_ONLY"},)
            if recovery_status == "MANUAL_REVIEW"
            else ()
        ),
        **kw,
    )


def _claim(*, status="CLAIMED", claimed=True, terminal=False, worker="recovery-a"):
    return RecoveryClaimReceipt(
        runtime_id=RUNTIME_ID,
        cycle_id=ORIGINAL,
        dispatch_id=DISPATCH,
        cancel_risk_receipt_id=CANCEL,
        runtime_version=10,
        head_state_id=HEAD,
        fencing_token=71,
        claim_fencing_token=4 if claimed else None,
        status=status,
        claimed=claimed,
        terminal=terminal,
        worker_token=worker if claimed else None,
        claim_until="2026-09-23T13:00:00Z" if claimed else None,
        risk_version=9,
        risk_receipt_id="t" * 64,
        risk_state="REDUCING" if claimed else (
            "HALTED" if status == "WAIT_RISK_RELEASE" else "REDUCING"
        ),
        recovery_status="READY_REDUCE_ONLY",
        recovery_legs=(_leg(),) if claimed else (),
    )


def _start(*, status="STARTED", started=True, duplicate=False, resume=False):
    return AtomicRecoveryStartReceipt(
        runtime_id=RUNTIME_ID,
        cycle_id=ORIGINAL,
        dispatch_id=DISPATCH,
        cancel_risk_receipt_id=CANCEL,
        runtime_version=10,
        head_state_id=HEAD,
        fencing_token=71,
        recovery_claim_fencing_token=4,
        status=status,
        started=started,
        duplicate=duplicate,
        resume_only=resume,
        risk_version=9 if started else None,
        risk_receipt_id="t" * 64 if started else None,
        risk_state="REDUCING" if started else None,
        recovery_legs=(_leg(),) if started else (),
    )


def _progress(*, terminal=True):
    return RecoveryCheckpointReceipt(
        runtime_id=RUNTIME_ID,
        original_cycle_id=ORIGINAL,
        recovery_cycle_id=RECOVERY,
        dispatch_id=DISPATCH,
        checkpoint_id=PROGRESS,
        journal_stage="COMMITTED" if terminal else "PAPER_APPLIED",
        version=10,
        current_version=10,
        fencing_token=71,
        recovery_claim_fencing_token=4,
        status="DUPLICATE_CURRENT" if terminal else "COMMITTED",
        committed=True,
        duplicate=terminal,
        terminal=terminal,
        head_state_id=HEAD,
    )


def _execution(*, terminal=True):
    return RecoveryExecutionCoreStep(
        recovery_cycle_id=RECOVERY,
        write_ahead=None,
        progress=_progress(terminal=terminal),
        durable_status="COMMITTED" if terminal else "RECONCILIATION_BLOCKED",
        outcome=(
            "RESTART_RECOVERY_RECOVERY_COMMITTED_PENDING_AUDIT"
            if terminal
            else "RESTART_RECOVERY_RECOVERY_RECONCILIATION_BLOCKED"
        ),
        persisted_version=10,
        checkpoint_id=CHECKPOINT,
    )


def _audit(*, status="CERTIFIED"):
    certified = status in {"CERTIFIED", "DUPLICATE"}
    failed = status == "AUDIT_FAILED"
    return RecoveryCompletionAuditReceipt(
        runtime_id=RUNTIME_ID,
        original_cycle_id=ORIGINAL,
        recovery_cycle_id=RECOVERY,
        dispatch_id=DISPATCH,
        completion_checkpoint_id=PROGRESS,
        runtime_version=10,
        fencing_token=71,
        status=status,
        certified=certified,
        duplicate=status == "DUPLICATE",
        cancel_risk_receipt_id=CANCEL if certified else None,
        start_head_state_id=HEAD if certified else None,
        final_head_state_id="f" * 64 if certified else None,
        paper_checkpoint_id="q" * 64 if certified else None,
        recovery_claim_fencing_token=4 if certified else None,
        recovery_fill_count=1 if certified else None,
        leg_audits=({"asset_id": "BTCUSDT", "safe": True},) if certified else (),
        failures=({"reason": "POSITION_FLIPPED"},) if failed else (),
    )


class _WorkStore:
    def __init__(self, *items):
        self.items = list(items)
        self.calls = 0

    def read_next(self, *, runtime_id):
        assert runtime_id == RUNTIME_ID
        self.calls += 1
        if not self.items:
            return _idle()
        if len(self.items) == 1:
            return self.items[0]
        return self.items.pop(0)


class _DirectiveStore:
    def __init__(self, *receipts):
        self.receipts = list(receipts)
        self.calls = 0

    def prepare(self, lease, *, cycle_id, expected_runtime_version):
        assert lease.runtime_id == RUNTIME_ID
        assert cycle_id == ORIGINAL
        assert expected_runtime_version == 10
        self.calls += 1
        return self.receipts.pop(0)


class _ClaimStore:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = 0

    def claim(self, lease, *, cycle_id, worker_token, claim_seconds):
        assert lease.runtime_id == RUNTIME_ID
        assert cycle_id == ORIGINAL
        assert worker_token == "recovery-a"
        assert claim_seconds == 30
        self.calls += 1
        return self.receipt


class _StartStore:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = 0

    def mark_started(self, lease, claim, *, worker_token):
        assert lease.runtime_id == RUNTIME_ID
        assert claim.cycle_id == ORIGINAL
        assert worker_token == "recovery-a"
        self.calls += 1
        return self.receipt


class _Execution:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = 0

    def execute_started_recovery(self, **kwargs):
        assert kwargs["start"].cycle_id == ORIGINAL
        assert kwargs["claim"].cycle_id == ORIGINAL
        assert kwargs["recovery_worker_token"] == "recovery-a"
        assert kwargs["base_outcome"] == "RESTART_RECOVERY"
        self.calls += 1
        return self.receipt


class _Audits:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = 0

    def certify(self, lease, *, original_cycle_id, recovery_cycle_id, expected_checkpoint_id):
        assert lease.runtime_id == RUNTIME_ID
        assert original_cycle_id == ORIGINAL
        assert recovery_cycle_id == RECOVERY
        assert expected_checkpoint_id == PROGRESS
        self.calls += 1
        return self.receipt


class _Aborter:
    def __init__(self):
        self.calls = []

    def abort_authorized_cycle(self, *, cycle_id, reason):
        self.calls.append((cycle_id, reason))


def _orchestrator(
    work,
    *,
    directive=None,
    claim=None,
    start=None,
    execution=None,
    audit=None,
    supervisor=None,
    aborter=None,
):
    return RecoveryRestartOrchestrator(
        runtime_supervisor=supervisor or _RuntimeSupervisor(),
        work=work,
        directives=directive or _DirectiveStore(),
        claims=claim or _ClaimStore(_claim()),
        starts=start or _StartStore(_start()),
        recovery_execution=execution or _Execution(_execution(terminal=False)),
        audits=audit or _Audits(_audit()),
        foreign_cycle_aborter=aborter,
    )


def _resume(orchestrator):
    return orchestrator.resume_next(
        recovery_worker_token="recovery-a",
        recovery_claim_seconds=30,
        recovery_markets={},
        recovery_risk_limits_by_asset={},
        recovery_ttl_seconds=60,
        marks={},
        observed_at=1.0,
        source_ref="phase88-test",
    )


def test_idle_does_not_touch_recovery_boundaries() -> None:
    work = _WorkStore(_idle())
    directives = _DirectiveStore()
    claims = _ClaimStore(_claim())
    starts = _StartStore(_start())
    execution = _Execution(_execution())
    audits = _Audits(_audit())
    orchestrator = RecoveryRestartOrchestrator(
        runtime_supervisor=_RuntimeSupervisor(),
        work=work,
        directives=directives,
        claims=claims,
        starts=starts,
        recovery_execution=execution,
        audits=audits,
    )
    step = _resume(orchestrator)
    assert step.outcome == "IDLE"
    assert directives.calls == claims.calls == starts.calls == execution.calls == audits.calls == 0


def test_stale_phase87_anchor_invalidates_local_runtime_before_any_action() -> None:
    supervisor = _RuntimeSupervisor()
    work = _WorkStore(_work(runtime_version=11))
    orchestrator = _orchestrator(work, supervisor=supervisor)
    with pytest.raises(PersistedRuntimeStaleError, match="runtime version"):
        _resume(orchestrator)
    assert supervisor.valid is False


def test_needs_directive_can_resolve_as_no_recovery_required_without_claim() -> None:
    directives = _DirectiveStore(_directive(recovery_status="NO_RECOVERY_REQUIRED"))
    claims = _ClaimStore(_claim())
    orchestrator = _orchestrator(
        _WorkStore(_work("NEEDS_DIRECTIVE")),
        directive=directives,
        claim=claims,
    )
    step = _resume(orchestrator)
    assert step.outcome == "NO_RECOVERY_REQUIRED"
    assert step.directive is not None
    assert claims.calls == 0


def test_manual_review_is_never_auto_claimed() -> None:
    claims = _ClaimStore(_claim())
    step = _resume(_orchestrator(
        _WorkStore(_work("MANUAL_REVIEW")),
        claim=claims,
    ))
    assert step.outcome == "MANUAL_REVIEW_REQUIRED"
    assert claims.calls == 0


def test_active_other_recovery_owner_is_not_stolen_before_expiry() -> None:
    blocked = _claim(status="BLOCKED_ACTIVE", claimed=False)
    claims = _ClaimStore(blocked)
    starts = _StartStore(_start())
    step = _resume(_orchestrator(
        _WorkStore(_work()),
        claim=claims,
        start=starts,
    ))
    assert step.outcome == "WAIT_ACTIVE_RECOVERY_OWNER"
    assert starts.calls == 0


def test_expired_or_unowned_work_claims_starts_and_resumes_phase84() -> None:
    claim = _claim(status="EXPIRED_RECOVERY")
    start = _start(status="STARTED_RESUME", duplicate=True, resume=True)
    execution = _Execution(_execution(terminal=False))
    step = _resume(_orchestrator(
        _WorkStore(_work("CLAIM_EXPIRED")),
        claim=_ClaimStore(claim),
        start=_StartStore(start),
        execution=execution,
    ))
    assert step.execution is not None
    assert step.outcome.endswith("RECOVERY_RECONCILIATION_BLOCKED")
    assert execution.calls == 1


def test_terminal_phase84_is_certified_and_same_backlog_item_must_disappear() -> None:
    work = _WorkStore(_work("STARTED_NEEDS_EXECUTION"), _idle())
    execution = _Execution(_execution(terminal=True))
    audits = _Audits(_audit())
    step = _resume(_orchestrator(
        work,
        execution=execution,
        audit=audits,
    ))
    assert step.outcome == "RECOVERY_COMPLETED"
    assert step.audit is not None and step.audit.certified is True
    assert audits.calls == 1
    assert work.calls == 2


def test_needs_audit_skips_claim_and_execution_and_certifies_directly() -> None:
    work = _WorkStore(_work("NEEDS_AUDIT"), _idle())
    claims = _ClaimStore(_claim())
    execution = _Execution(_execution())
    audits = _Audits(_audit())
    step = _resume(_orchestrator(
        work,
        claim=claims,
        execution=execution,
        audit=audits,
    ))
    assert step.outcome == "RECOVERY_COMPLETED"
    assert claims.calls == 0
    assert execution.calls == 0
    assert audits.calls == 1


def test_certificate_that_leaves_same_backlog_item_visible_fails_closed() -> None:
    supervisor = _RuntimeSupervisor()
    same = _work("NEEDS_AUDIT")
    work = _WorkStore(same, same)
    orchestrator = _orchestrator(
        work,
        audit=_Audits(_audit()),
        supervisor=supervisor,
    )
    with pytest.raises(PersistedRuntimeStaleError, match="remains unresolved"):
        _resume(orchestrator)
    assert supervisor.valid is False


def test_completed_without_certificate_is_explicit_fail_closed_state() -> None:
    supervisor = _RuntimeSupervisor()
    orchestrator = _orchestrator(
        _WorkStore(_work("COMPLETED_WITHOUT_CERTIFICATE")),
        supervisor=supervisor,
    )
    with pytest.raises(
        RecoveryRestartOrchestrationError,
        match="without Phase85 certificate",
    ):
        _resume(orchestrator)
    assert supervisor.valid is False


def test_audit_failure_invalidates_runtime_and_preserves_failure_reason() -> None:
    supervisor = _RuntimeSupervisor()
    work = _WorkStore(_work("NEEDS_AUDIT"))
    orchestrator = _orchestrator(
        work,
        audit=_Audits(_audit(status="AUDIT_FAILED")),
        supervisor=supervisor,
    )
    with pytest.raises(RecoveryCompletionAuditError, match="POSITION_FLIPPED"):
        _resume(orchestrator)
    assert supervisor.valid is False


def test_foreign_prepaper_cycle_can_be_quarantined_then_directive_retried() -> None:
    foreign = CancelRecoveryDirectiveReceipt(
        runtime_id=RUNTIME_ID,
        cycle_id=ORIGINAL,
        dispatch_id=DISPATCH,
        runtime_version=10,
        fencing_token=71,
        status="FOREIGN_CYCLE_ACTIVE",
        prepared=False,
        duplicate=False,
        foreign_cycle_id="f" * 64,
        foreign_cycle_stage="CYCLE_CREATED",
    )
    ready = _directive(recovery_status="NO_RECOVERY_REQUIRED")
    aborter = _Aborter()
    directives = _DirectiveStore(foreign, ready)
    step = _resume(_orchestrator(
        _WorkStore(_work("NEEDS_DIRECTIVE")),
        directive=directives,
        aborter=aborter,
    ))
    assert step.outcome == "NO_RECOVERY_REQUIRED"
    assert aborter.calls == [
        ("f" * 64, "phase88:restart-recovery-quarantine")
    ]
    assert directives.calls == 2
