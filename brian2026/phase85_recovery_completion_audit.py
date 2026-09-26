from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from .phase84_recovery_execution_checkpoint import (
    PersistedRecoveryExecutionSupervisor,
    RecoveryExecutionStep,
)

PHASE85_SCHEMA_VERSION = "brian.phase85-recovery-completion-audit.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class RecoveryCompletionAuditError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryCompletionAuditReceipt:
    runtime_id: str
    original_cycle_id: str
    recovery_cycle_id: str
    dispatch_id: str
    completion_checkpoint_id: str
    runtime_version: int
    fencing_token: int
    status: str
    certified: bool
    duplicate: bool
    cancel_risk_receipt_id: str | None = None
    start_head_state_id: str | None = None
    final_head_state_id: str | None = None
    paper_checkpoint_id: str | None = None
    recovery_claim_fencing_token: int | None = None
    recovery_fill_count: int | None = None
    leg_audits: tuple[Mapping[str, object], ...] = ()
    failures: tuple[Mapping[str, object], ...] = ()
    schema_version: str = PHASE85_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        for label, value in (
            ("original_cycle_id", self.original_cycle_id),
            ("recovery_cycle_id", self.recovery_cycle_id),
            ("completion_checkpoint_id", self.completion_checkpoint_id),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be a content hash")
        if self.original_cycle_id == self.recovery_cycle_id:
            raise ValueError("recovery cycle must differ from original cycle")
        if self.dispatch_id and len(self.dispatch_id) != 64:
            raise ValueError("dispatch_id must be a content hash")
        if self.runtime_version < 0 or self.fencing_token <= 0:
            raise ValueError("runtime version/fence is invalid")
        if self.certified and self.status not in {"CERTIFIED", "DUPLICATE"}:
            raise ValueError("certified audit has unsupported status")
        if self.status == "CERTIFIED" and (not self.certified or self.duplicate):
            raise ValueError("CERTIFIED flags are inconsistent")
        if self.status == "DUPLICATE" and not (self.certified and self.duplicate):
            raise ValueError("DUPLICATE flags are inconsistent")
        if self.status == "AUDIT_FAILED" and (self.certified or not self.failures):
            raise ValueError("AUDIT_FAILED must carry failure evidence")
        if self.certified:
            for label, value in (
                ("cancel_risk_receipt_id", self.cancel_risk_receipt_id),
                ("start_head_state_id", self.start_head_state_id),
                ("final_head_state_id", self.final_head_state_id),
                ("paper_checkpoint_id", self.paper_checkpoint_id),
            ):
                if value is None or len(value) != 64:
                    raise ValueError(f"certified audit requires {label}")
            if (
                self.recovery_claim_fencing_token is None
                or self.recovery_claim_fencing_token <= 0
            ):
                raise ValueError("certified audit requires recovery claim fence")
            if self.recovery_fill_count is None or self.recovery_fill_count <= 0:
                raise ValueError("certified audit requires recovery fills")
            if not self.leg_audits:
                raise ValueError("certified audit requires leg audits")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase85 audit must remain shadow-only")


@dataclass(frozen=True, slots=True)
class CertifiedRecoveryExecutionStep:
    recovery_step: RecoveryExecutionStep
    audit: RecoveryCompletionAuditReceipt | None
    outcome: str
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE85_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RecoveryCompletionAuditError(f"{label} returned non-object payload")
    return {str(key): item for key, item in value.items()}


def _integer(
    value: object,
    label: str,
    *,
    default: int | None = None,
) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        raise RecoveryCompletionAuditError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RecoveryCompletionAuditError(f"{label} must be integer") from exc


def _optional_hash(value: object, label: str) -> str | None:
    if value is None:
        return None
    result = str(value)
    if len(result) != 64:
        raise RecoveryCompletionAuditError(f"{label} must be a content hash")
    return result


def _rows(value: object, label: str) -> tuple[Mapping[str, object], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise RecoveryCompletionAuditError(f"{label} must be an array")
    return tuple(_mapping(row, label) for row in value)


class RecoveryCompletionAuditStore:
    """Certify the final authoritative paper state before marking recovery complete."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def certify(
        self,
        lease,
        *,
        original_cycle_id: str,
        recovery_cycle_id: str,
        expected_checkpoint_id: str,
    ) -> RecoveryCompletionAuditReceipt:
        if not lease.acquired:
            raise RecoveryCompletionAuditError("runtime lease is not acquired")
        for label, value in (
            ("original_cycle_id", original_cycle_id),
            ("recovery_cycle_id", recovery_cycle_id),
            ("expected_checkpoint_id", expected_checkpoint_id),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be a content hash")
        if original_cycle_id == recovery_cycle_id:
            raise ValueError("recovery cycle must differ from original cycle")

        row = _mapping(
            self._rpc(
                "brian_certify_shadow_recovery_completion",
                {
                    "p_runtime_id": lease.runtime_id,
                    "p_owner_token": lease.owner_token,
                    "p_fencing_token": lease.fencing_token,
                    "p_original_cycle_id": original_cycle_id,
                    "p_recovery_cycle_id": recovery_cycle_id,
                    "p_expected_checkpoint_id": expected_checkpoint_id,
                },
            ),
            "brian_certify_shadow_recovery_completion",
        )

        if str(row.get("runtime_id", "")) != lease.runtime_id:
            raise RecoveryCompletionAuditError("audit runtime_id mismatch")
        if str(row.get("original_cycle_id", "")) != original_cycle_id:
            raise RecoveryCompletionAuditError("audit original cycle mismatch")
        if str(row.get("recovery_cycle_id", "")) != recovery_cycle_id:
            raise RecoveryCompletionAuditError("audit recovery cycle mismatch")

        certified_raw = row.get("certified")
        duplicate_raw = row.get("duplicate")
        if not isinstance(certified_raw, bool) or not isinstance(duplicate_raw, bool):
            raise RecoveryCompletionAuditError("certified/duplicate must be boolean")

        receipt = RecoveryCompletionAuditReceipt(
            runtime_id=lease.runtime_id,
            original_cycle_id=original_cycle_id,
            recovery_cycle_id=recovery_cycle_id,
            dispatch_id=str(row.get("dispatch_id", "")),
            completion_checkpoint_id=str(
                row.get("completion_checkpoint_id", expected_checkpoint_id)
            ),
            runtime_version=_integer(
                row.get("runtime_version"),
                "runtime_version",
                default=lease.version,
            ),
            fencing_token=_integer(
                row.get("fencing_token"),
                "fencing_token",
                default=lease.fencing_token,
            ),
            status=str(row.get("status", "")),
            certified=certified_raw,
            duplicate=duplicate_raw,
            cancel_risk_receipt_id=_optional_hash(
                row.get("cancel_risk_receipt_id"),
                "cancel_risk_receipt_id",
            ),
            start_head_state_id=_optional_hash(
                row.get("start_head_state_id"),
                "start_head_state_id",
            ),
            final_head_state_id=_optional_hash(
                row.get("final_head_state_id"),
                "final_head_state_id",
            ),
            paper_checkpoint_id=_optional_hash(
                row.get("paper_checkpoint_id"),
                "paper_checkpoint_id",
            ),
            recovery_claim_fencing_token=(
                None
                if row.get("recovery_claim_fencing_token") is None
                else _integer(
                    row.get("recovery_claim_fencing_token"),
                    "recovery_claim_fencing_token",
                )
            ),
            recovery_fill_count=(
                None
                if row.get("recovery_fill_count") is None
                else _integer(row.get("recovery_fill_count"), "recovery_fill_count")
            ),
            leg_audits=_rows(row.get("leg_audits"), "leg_audits"),
            failures=_rows(row.get("failures"), "failures"),
        )

        if receipt.completion_checkpoint_id != expected_checkpoint_id:
            raise RecoveryCompletionAuditError("audit completion checkpoint drift")
        if receipt.fencing_token != lease.fencing_token:
            raise RecoveryCompletionAuditError("audit runtime fence drift")
        return receipt


class PersistedRecoveryCompletionAuditSupervisor:
    """Phase84 terminal commit followed by Phase85 authoritative exposure audit."""

    def __init__(
        self,
        *,
        recovery_supervisor: PersistedRecoveryExecutionSupervisor,
        audits: RecoveryCompletionAuditStore,
    ) -> None:
        self.recovery_supervisor = recovery_supervisor
        self.audits = audits

    def _runtime_supervisor(self):
        return self.recovery_supervisor._runtime_supervisor()

    def process_governed_cycle(self, governed, **kwargs) -> CertifiedRecoveryExecutionStep:
        recovery_step = self.recovery_supervisor.process_governed_cycle(
            governed,
            **kwargs,
        )
        supervisor = self._runtime_supervisor()
        progress = recovery_step.progress

        if (
            progress is None
            or not progress.terminal
            or recovery_step.recovery_cycle_id is None
        ):
            return CertifiedRecoveryExecutionStep(
                recovery_step=recovery_step,
                audit=None,
                outcome=recovery_step.outcome,
                persisted_version=recovery_step.persisted_version,
                checkpoint_id=recovery_step.checkpoint_id,
            )

        audit = self.audits.certify(
            supervisor.lease,
            original_cycle_id=progress.original_cycle_id,
            recovery_cycle_id=progress.recovery_cycle_id,
            expected_checkpoint_id=progress.checkpoint_id,
        )

        if audit.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost during recovery completion audit"
            )
        if audit.status in {
            "HEAD_MOVED",
            "CLAIM_STATE_INVALID",
            "EVIDENCE_INVALID",
        }:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"recovery completion audit rejected with {audit.status}"
            )
        if audit.status == "AUDIT_FAILED":
            supervisor._valid = False
            reasons = tuple(
                str(row.get("reason", "UNKNOWN"))
                for row in audit.failures
            )
            raise RecoveryCompletionAuditError(
                "recovery completion invariant failed: " + ",".join(reasons)
            )
        if audit.status not in {"CERTIFIED", "DUPLICATE"} or not audit.certified:
            supervisor._valid = False
            raise RecoveryCompletionAuditError(
                f"unsupported recovery audit status {audit.status}"
            )

        return CertifiedRecoveryExecutionStep(
            recovery_step=recovery_step,
            audit=audit,
            outcome=f"{recovery_step.outcome}_RECOVERY_COMPLETED",
            persisted_version=supervisor.persisted_version,
            checkpoint_id=supervisor.runtime.checkpoint().checkpoint_id,
        )
