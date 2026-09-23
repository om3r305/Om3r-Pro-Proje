from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .phase56_pretrade_risk_engine import InstrumentRiskLimits
from .phase57_shadow_execution_cycle import ExecutionMarketInput
from .phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from .phase81_cancel_recovery_directive import (
    CancelRecoveryDirectiveReceipt,
    CancelRecoveryDirectiveStore,
)
from .phase82_recovery_claim_fencing import (
    RecoveryClaimError,
    RecoveryClaimReceipt,
    RecoveryClaimStore,
)
from .phase83_atomic_recovery_start import (
    AtomicRecoveryStartReceipt,
    AtomicRecoveryStartStore,
)
from .phase84_recovery_execution_checkpoint import (
    PersistedRecoveryExecutionSupervisor,
    RecoveryExecutionCoreStep,
)
from .phase85_recovery_completion_audit import (
    RecoveryCompletionAuditError,
    RecoveryCompletionAuditReceipt,
    RecoveryCompletionAuditStore,
)
from .phase87_recovery_restart_resume import (
    RecoveryRestartWorkItem,
    RecoveryRestartWorkStore,
)

PHASE88_SCHEMA_VERSION = "brian.phase88-recovery-restart-orchestrator.v1"


class RecoveryRestartOrchestrationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryRestartStep:
    work: RecoveryRestartWorkItem
    directive: CancelRecoveryDirectiveReceipt | None
    claim: RecoveryClaimReceipt | None
    start: AtomicRecoveryStartReceipt | None
    execution: RecoveryExecutionCoreStep | None
    audit: RecoveryCompletionAuditReceipt | None
    outcome: str
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE88_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if self.persisted_version < 0:
            raise ValueError("persisted_version must be non-negative")
        if len(self.checkpoint_id) != 64:
            raise ValueError("checkpoint_id must be a content hash")
        if not self.outcome:
            raise ValueError("outcome is required")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase88 restart orchestration must remain shadow-only")


class RecoveryRestartOrchestrator:
    """Resume one DB-authoritative recovery backlog item after process restart.

    Phase88 deliberately reuses the existing Phase81-85 stores and Phase84
    direct-resume core. It never invents a parallel recovery/execution path.
    One call processes at most the oldest Phase87 backlog item so a broken item
    cannot create an unbounded recovery loop.
    """

    def __init__(
        self,
        *,
        runtime_supervisor,
        work: RecoveryRestartWorkStore,
        directives: CancelRecoveryDirectiveStore,
        claims: RecoveryClaimStore,
        starts: AtomicRecoveryStartStore,
        recovery_execution: PersistedRecoveryExecutionSupervisor,
        audits: RecoveryCompletionAuditStore,
        foreign_cycle_aborter=None,
    ) -> None:
        self.runtime_supervisor = runtime_supervisor
        self.work = work
        self.directives = directives
        self.claims = claims
        self.starts = starts
        self.recovery_execution = recovery_execution
        self.audits = audits
        self.foreign_cycle_aborter = foreign_cycle_aborter

    def _checkpoint_id(self) -> str:
        return self.runtime_supervisor.runtime.checkpoint().checkpoint_id

    def _head_state_id(self) -> str:
        return self.runtime_supervisor.runtime.ledger.head_state.state_id

    def _step(
        self,
        work: RecoveryRestartWorkItem,
        *,
        outcome: str,
        directive: CancelRecoveryDirectiveReceipt | None = None,
        claim: RecoveryClaimReceipt | None = None,
        start: AtomicRecoveryStartReceipt | None = None,
        execution: RecoveryExecutionCoreStep | None = None,
        audit: RecoveryCompletionAuditReceipt | None = None,
    ) -> RecoveryRestartStep:
        return RecoveryRestartStep(
            work=work,
            directive=directive,
            claim=claim,
            start=start,
            execution=execution,
            audit=audit,
            outcome=outcome,
            persisted_version=self.runtime_supervisor.persisted_version,
            checkpoint_id=self._checkpoint_id(),
        )

    def _assert_work_anchor(self, work: RecoveryRestartWorkItem) -> None:
        supervisor = self.runtime_supervisor
        if not supervisor.valid:
            raise PersistedRuntimeStaleError(
                "runtime supervisor is stale before recovery restart"
            )
        if not work.has_work:
            return
        if work.runtime_id != supervisor.runtime_id:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "recovery backlog runtime does not match local supervisor"
            )
        if work.runtime_version != supervisor.persisted_version:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "recovery backlog runtime version differs from local supervisor"
            )
        if work.runtime_checkpoint_id != self._checkpoint_id():
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "recovery backlog checkpoint differs from local supervisor"
            )
        if work.runtime_head_state_id != self._head_state_id():
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "recovery backlog Phase60 head differs from local supervisor"
            )

    def _handle_directive_error(
        self,
        receipt: CancelRecoveryDirectiveReceipt,
    ) -> None:
        supervisor = self.runtime_supervisor
        if receipt.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost while restart worker prepared recovery"
            )
        if receipt.status in {
            "RUNTIME_VERSION_CONFLICT",
            "HEAD_MOVED",
            "RISK_STATE_UNAVAILABLE",
            "EVIDENCE_INVALID",
        }:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"restart recovery directive rejected with {receipt.status}"
            )
        if receipt.status == "NO_CANCEL_REQUEST":
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "Phase87 reported recovery work but Phase81 found no cancel request"
            )

    def _prepare_directive(
        self,
        work: RecoveryRestartWorkItem,
    ) -> CancelRecoveryDirectiveReceipt:
        receipt = self.directives.prepare(
            self.runtime_supervisor.lease,
            cycle_id=str(work.original_cycle_id),
            expected_runtime_version=self.runtime_supervisor.persisted_version,
        )
        self._handle_directive_error(receipt)

        if (
            receipt.status == "FOREIGN_CYCLE_ACTIVE"
            and receipt.foreign_cycle_stage == "CYCLE_CREATED"
            and receipt.foreign_cycle_id is not None
            and self.foreign_cycle_aborter is not None
        ):
            self.foreign_cycle_aborter.abort_authorized_cycle(
                cycle_id=receipt.foreign_cycle_id,
                reason="phase88:restart-recovery-quarantine",
            )
            receipt = self.directives.prepare(
                self.runtime_supervisor.lease,
                cycle_id=str(work.original_cycle_id),
                expected_runtime_version=self.runtime_supervisor.persisted_version,
            )
            self._handle_directive_error(receipt)
        return receipt

    def _handle_claim_error(self, claim: RecoveryClaimReceipt) -> None:
        supervisor = self.runtime_supervisor
        if claim.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost while restart worker claimed recovery"
            )
        if claim.status in {
            "HEAD_MOVED",
            "DIRECTIVE_MISSING",
            "RISK_STATE_UNAVAILABLE",
            "EVIDENCE_INVALID",
        }:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"restart recovery claim rejected with {claim.status}"
            )
        if claim.status == "COMPLETED":
            supervisor._valid = False
            raise RecoveryRestartOrchestrationError(
                "recovery claim is COMPLETED without Phase85 backlog resolution"
            )

    def _handle_start_error(self, start: AtomicRecoveryStartReceipt) -> None:
        supervisor = self.runtime_supervisor
        if start.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost at restart recovery STARTED boundary"
            )
        if start.status in {
            "HEAD_MOVED",
            "DIRECTIVE_MISSING",
            "RISK_STATE_UNAVAILABLE",
            "EVIDENCE_INVALID",
        }:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"restart recovery STARTED rejected with {start.status}"
            )
        if start.status == "CLAIM_LOST":
            raise RecoveryClaimError(
                "recovery claim was lost before restart STARTED boundary"
            )

    def _certify(
        self,
        work: RecoveryRestartWorkItem,
        *,
        original_cycle_id: str,
        recovery_cycle_id: str,
        checkpoint_id: str,
    ) -> RecoveryCompletionAuditReceipt:
        audit = self.audits.certify(
            self.runtime_supervisor.lease,
            original_cycle_id=original_cycle_id,
            recovery_cycle_id=recovery_cycle_id,
            expected_checkpoint_id=checkpoint_id,
        )
        supervisor = self.runtime_supervisor
        if audit.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost during restart recovery audit"
            )
        if audit.status in {
            "HEAD_MOVED",
            "CLAIM_STATE_INVALID",
            "EVIDENCE_INVALID",
        }:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"restart recovery audit rejected with {audit.status}"
            )
        if audit.status == "AUDIT_FAILED":
            supervisor._valid = False
            reasons = tuple(
                str(row.get("reason", "UNKNOWN"))
                for row in audit.failures
            )
            raise RecoveryCompletionAuditError(
                "restart recovery audit failed: " + ",".join(reasons)
            )
        if audit.status not in {"CERTIFIED", "DUPLICATE"} or not audit.certified:
            supervisor._valid = False
            raise RecoveryCompletionAuditError(
                f"unsupported restart recovery audit status {audit.status}"
            )

        # A successful certificate must remove this exact item from the
        # unresolved backlog. Another older/newer recovery item may still exist.
        next_work = self.work.read_next(runtime_id=supervisor.runtime_id)
        if (
            next_work.has_work
            and next_work.original_cycle_id == work.original_cycle_id
            and next_work.cancel_risk_receipt_id == work.cancel_risk_receipt_id
        ):
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "certified recovery remains unresolved in Phase87 backlog"
            )
        return audit

    def resume_next(
        self,
        *,
        recovery_worker_token: str,
        recovery_claim_seconds: int,
        recovery_markets: Mapping[str, ExecutionMarketInput],
        recovery_risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
        recovery_ttl_seconds: int,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> RecoveryRestartStep:
        if not recovery_worker_token.strip():
            raise ValueError("recovery_worker_token is required")
        if recovery_claim_seconds <= 0 or recovery_ttl_seconds <= 0:
            raise ValueError("recovery claim/intent TTL must be positive")

        supervisor = self.runtime_supervisor
        work = self.work.read_next(runtime_id=supervisor.runtime_id)
        self._assert_work_anchor(work)

        if not work.has_work:
            return self._step(work, outcome="IDLE")
        if work.work_state == "MANUAL_REVIEW":
            return self._step(work, outcome="MANUAL_REVIEW_REQUIRED")
        if work.work_state == "COMPLETED_WITHOUT_CERTIFICATE":
            supervisor._valid = False
            raise RecoveryRestartOrchestrationError(
                "recovery claim completed without Phase85 certificate"
            )

        if work.work_state == "NEEDS_AUDIT":
            assert work.original_cycle_id is not None
            assert work.recovery_cycle_id is not None
            assert work.progress_checkpoint_id is not None
            audit = self._certify(
                work,
                original_cycle_id=work.original_cycle_id,
                recovery_cycle_id=work.recovery_cycle_id,
                checkpoint_id=work.progress_checkpoint_id,
            )
            return self._step(
                work,
                audit=audit,
                outcome="RECOVERY_COMPLETED",
            )

        directive: CancelRecoveryDirectiveReceipt | None = None
        if work.work_state == "NEEDS_DIRECTIVE":
            directive = self._prepare_directive(work)
            if directive.status == "WAIT_ORIGINAL_COMMIT":
                return self._step(
                    work,
                    directive=directive,
                    outcome="WAIT_ORIGINAL_COMMIT",
                )
            if directive.status == "FOREIGN_CYCLE_ACTIVE":
                return self._step(
                    work,
                    directive=directive,
                    outcome="WAIT_FOREIGN_CYCLE",
                )
            if not directive.prepared:
                return self._step(
                    work,
                    directive=directive,
                    outcome=f"WAIT_{directive.status}",
                )
            if directive.recovery_status == "NO_RECOVERY_REQUIRED":
                return self._step(
                    work,
                    directive=directive,
                    outcome="NO_RECOVERY_REQUIRED",
                )
            if directive.recovery_status == "MANUAL_REVIEW":
                return self._step(
                    work,
                    directive=directive,
                    outcome="MANUAL_REVIEW_REQUIRED",
                )

        assert work.original_cycle_id is not None
        claim = self.claims.claim(
            supervisor.lease,
            cycle_id=work.original_cycle_id,
            worker_token=recovery_worker_token,
            claim_seconds=recovery_claim_seconds,
        )
        self._handle_claim_error(claim)

        if claim.status == "WAIT_RISK_RELEASE":
            return self._step(
                work,
                directive=directive,
                claim=claim,
                outcome="WAIT_RISK_RELEASE",
            )
        if claim.status == "BLOCKED_ACTIVE":
            return self._step(
                work,
                directive=directive,
                claim=claim,
                outcome="WAIT_ACTIVE_RECOVERY_OWNER",
            )
        if not claim.claimed:
            return self._step(
                work,
                directive=directive,
                claim=claim,
                outcome=f"WAIT_{claim.status}",
            )

        start = self.starts.mark_started(
            supervisor.lease,
            claim,
            worker_token=recovery_worker_token,
        )
        self._handle_start_error(start)

        if start.status == "WAIT_RISK_RELEASE":
            return self._step(
                work,
                directive=directive,
                claim=claim,
                start=start,
                outcome="WAIT_RISK_RELEASE",
            )
        if not start.started:
            return self._step(
                work,
                directive=directive,
                claim=claim,
                start=start,
                outcome=f"WAIT_{start.status}",
            )

        execution = self.recovery_execution.execute_started_recovery(
            start=start,
            claim=claim,
            recovery_worker_token=recovery_worker_token,
            recovery_claim_seconds=recovery_claim_seconds,
            recovery_markets=recovery_markets,
            recovery_risk_limits_by_asset=recovery_risk_limits_by_asset,
            recovery_ttl_seconds=recovery_ttl_seconds,
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
            base_outcome="RESTART_RECOVERY",
        )

        progress = execution.progress
        if (
            progress is None
            or not progress.terminal
            or execution.recovery_cycle_id is None
        ):
            return self._step(
                work,
                directive=directive,
                claim=claim,
                start=start,
                execution=execution,
                outcome=execution.outcome,
            )

        audit = self._certify(
            work,
            original_cycle_id=progress.original_cycle_id,
            recovery_cycle_id=progress.recovery_cycle_id,
            checkpoint_id=progress.checkpoint_id,
        )
        return self._step(
            work,
            directive=directive,
            claim=claim,
            start=start,
            execution=execution,
            audit=audit,
            outcome="RECOVERY_COMPLETED",
        )
