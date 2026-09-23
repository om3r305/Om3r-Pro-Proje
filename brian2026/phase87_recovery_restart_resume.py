from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

PHASE87_SCHEMA_VERSION = "brian.phase87-recovery-restart-resume.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]

WORK_STATES = {
    "IDLE",
    "NEEDS_DIRECTIVE",
    "MANUAL_REVIEW",
    "COMPLETED_WITHOUT_CERTIFICATE",
    "NEEDS_AUDIT",
    "NEEDS_CLAIM",
    "CLAIM_EXPIRED",
    "NEEDS_START",
    "STARTED_NEEDS_EXECUTION",
    "RECOVERY_PROGRESS",
}


class RecoveryRestartWorkError(RuntimeError):
    pass


def _optional_hash(value: object, label: str) -> str | None:
    if value is None:
        return None
    result = str(value)
    if len(result) != 64:
        raise RecoveryRestartWorkError(f"{label} must be a content hash")
    return result


def _optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RecoveryRestartWorkError(f"{label} must be integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RecoveryRestartWorkError(f"{label} must be integer") from exc
    if result < 0:
        raise RecoveryRestartWorkError(f"{label} cannot be negative")
    return result


@dataclass(frozen=True, slots=True)
class RecoveryRestartWorkItem:
    runtime_id: str
    has_work: bool
    status: str
    work_state: str
    original_cycle_id: str | None = None
    dispatch_id: str | None = None
    cancel_risk_version: int | None = None
    cancel_risk_receipt_id: str | None = None
    cancel_reason: str | None = None
    requested_at: object | None = None
    runtime_version: int | None = None
    runtime_checkpoint_id: str | None = None
    runtime_head_state_id: str | None = None
    directive_exists: bool = False
    recovery_status: str | None = None
    directive_runtime_version: int | None = None
    directive_state_id: str | None = None
    directive_prepared_at: object | None = None
    claim_status: str | None = None
    claim_worker_token: str | None = None
    claim_fencing_token: int | None = None
    claim_until: object | None = None
    recovery_cycle_id: str | None = None
    progress_runtime_version: int | None = None
    progress_head_state_id: str | None = None
    progress_checkpoint_id: str | None = None
    started: bool = False
    started_at: object | None = None
    recovery_journal_stage: str | None = None
    phase84_terminal_event: bool = False
    schema_version: str = PHASE87_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if self.work_state not in WORK_STATES:
            raise ValueError(f"unsupported recovery work_state {self.work_state}")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase87 work item must remain shadow-only")

        if not self.has_work:
            if self.status != "IDLE" or self.work_state != "IDLE":
                raise ValueError("no-work result must be IDLE")
            if any(
                value is not None
                for value in (
                    self.original_cycle_id,
                    self.dispatch_id,
                    self.cancel_risk_receipt_id,
                    self.recovery_cycle_id,
                )
            ):
                raise ValueError("IDLE recovery work cannot carry work identity")
            return

        if self.status != "WORK":
            raise ValueError("active recovery backlog must use WORK status")
        for label, value in (
            ("original_cycle_id", self.original_cycle_id),
            ("dispatch_id", self.dispatch_id),
            ("cancel_risk_receipt_id", self.cancel_risk_receipt_id),
            ("runtime_checkpoint_id", self.runtime_checkpoint_id),
            ("runtime_head_state_id", self.runtime_head_state_id),
        ):
            if value is None or len(value) != 64:
                raise ValueError(f"{label} must be a content hash for active work")
        if self.cancel_risk_version is None or self.cancel_risk_version <= 0:
            raise ValueError("active recovery work requires cancel_risk_version")
        if self.runtime_version is None or self.runtime_version <= 0:
            raise ValueError("active recovery work requires runtime_version")
        if not self.cancel_reason:
            raise ValueError("active recovery work requires cancel_reason")

        if self.work_state == "NEEDS_DIRECTIVE":
            if self.directive_exists:
                raise ValueError("NEEDS_DIRECTIVE cannot already have directive")
        else:
            if not self.directive_exists:
                raise ValueError(f"{self.work_state} requires durable directive")

        if self.directive_exists:
            if self.directive_runtime_version is None or self.directive_runtime_version <= 0:
                raise ValueError("durable directive requires runtime version")
            if self.directive_state_id is None or len(self.directive_state_id) != 64:
                raise ValueError("durable directive requires state id")
            if not self.recovery_status:
                raise ValueError("durable directive requires recovery_status")

        if self.work_state == "MANUAL_REVIEW" and self.recovery_status != "MANUAL_REVIEW":
            raise ValueError("MANUAL_REVIEW work must come from MANUAL_REVIEW directive")
        if self.work_state == "COMPLETED_WITHOUT_CERTIFICATE" and self.claim_status != "COMPLETED":
            raise ValueError("COMPLETED_WITHOUT_CERTIFICATE requires completed claim")

        if self.claim_status is not None:
            if self.claim_fencing_token is None or self.claim_fencing_token <= 0:
                raise ValueError("claim state requires positive claim fence")
        if self.claim_status == "CLAIMED" and not self.claim_worker_token:
            raise ValueError("CLAIMED recovery requires worker token")

        if self.work_state == "CLAIM_EXPIRED" and self.claim_status != "CLAIMED":
            raise ValueError("CLAIM_EXPIRED requires CLAIMED state")
        if self.started and self.started_at is None:
            raise ValueError("started recovery requires started_at")
        if self.work_state == "NEEDS_START" and self.started:
            raise ValueError("NEEDS_START cannot already be STARTED")
        if self.work_state == "STARTED_NEEDS_EXECUTION":
            if not self.started or self.recovery_cycle_id is not None:
                raise ValueError(
                    "STARTED_NEEDS_EXECUTION requires STARTED and no recovery cycle"
                )

        if self.recovery_cycle_id is not None:
            if len(self.recovery_cycle_id) != 64:
                raise ValueError("recovery_cycle_id must be a content hash")
            if not self.started:
                raise ValueError("durable recovery cycle requires STARTED evidence")
            if self.progress_runtime_version is None or self.progress_runtime_version <= 0:
                raise ValueError("durable recovery cycle requires progress runtime version")
            if self.progress_checkpoint_id is None or len(self.progress_checkpoint_id) != 64:
                raise ValueError("durable recovery cycle requires progress checkpoint")
            if self.progress_head_state_id is None or len(self.progress_head_state_id) != 64:
                raise ValueError("durable recovery cycle requires progress head state")

        if self.work_state == "RECOVERY_PROGRESS" and self.recovery_cycle_id is None:
            raise ValueError("RECOVERY_PROGRESS requires recovery_cycle_id")
        if self.work_state == "NEEDS_AUDIT":
            if (
                self.recovery_cycle_id is None
                or self.recovery_journal_stage != "COMMITTED"
                or not self.phase84_terminal_event
                or self.progress_checkpoint_id is None
            ):
                raise ValueError(
                    "NEEDS_AUDIT requires terminal Phase84 recovery evidence"
                )

    @property
    def action(self) -> str:
        return {
            "IDLE": "NONE",
            "NEEDS_DIRECTIVE": "PREPARE_DIRECTIVE",
            "MANUAL_REVIEW": "MANUAL_REVIEW",
            "COMPLETED_WITHOUT_CERTIFICATE": "FAIL_CLOSED",
            "NEEDS_AUDIT": "AUDIT",
            "NEEDS_CLAIM": "ACQUIRE_CLAIM",
            "CLAIM_EXPIRED": "TAKE_OVER_CLAIM",
            "NEEDS_START": "MARK_STARTED",
            "STARTED_NEEDS_EXECUTION": "EXECUTE_RECOVERY",
            "RECOVERY_PROGRESS": "RESUME_RECOVERY",
        }[self.work_state]


class RecoveryRestartWorkStore:
    """Read one DB-authoritative unresolved recovery item after process restart."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def read_next(self, *, runtime_id: str) -> RecoveryRestartWorkItem:
        if not runtime_id.strip():
            raise ValueError("runtime_id is required")
        raw = self._rpc(
            "brian_read_next_shadow_recovery_work",
            {"p_runtime_id": runtime_id},
        )
        if not isinstance(raw, Mapping):
            raise RecoveryRestartWorkError(
                "brian_read_next_shadow_recovery_work returned non-object payload"
            )
        row = {str(key): value for key, value in raw.items()}
        if str(row.get("runtime_id", "")) != runtime_id:
            raise RecoveryRestartWorkError("recovery backlog runtime_id mismatch")

        has_work = row.get("has_work")
        if not isinstance(has_work, bool):
            raise RecoveryRestartWorkError("has_work must be boolean")

        if not has_work:
            return RecoveryRestartWorkItem(
                runtime_id=runtime_id,
                has_work=False,
                status=str(row.get("status", "")),
                work_state="IDLE",
            )

        directive_exists = row.get("directive_exists")
        started = row.get("started")
        terminal_event = row.get("phase84_terminal_event")
        if not isinstance(directive_exists, bool):
            raise RecoveryRestartWorkError("directive_exists must be boolean")
        if not isinstance(started, bool):
            raise RecoveryRestartWorkError("started must be boolean")
        if not isinstance(terminal_event, bool):
            raise RecoveryRestartWorkError("phase84_terminal_event must be boolean")

        return RecoveryRestartWorkItem(
            runtime_id=runtime_id,
            has_work=True,
            status=str(row.get("status", "")),
            work_state=str(row.get("work_state", "")),
            original_cycle_id=_optional_hash(
                row.get("original_cycle_id"),
                "original_cycle_id",
            ),
            dispatch_id=_optional_hash(row.get("dispatch_id"), "dispatch_id"),
            cancel_risk_version=_optional_int(
                row.get("cancel_risk_version"),
                "cancel_risk_version",
            ),
            cancel_risk_receipt_id=_optional_hash(
                row.get("cancel_risk_receipt_id"),
                "cancel_risk_receipt_id",
            ),
            cancel_reason=(
                None if row.get("cancel_reason") is None else str(row.get("cancel_reason"))
            ),
            requested_at=row.get("requested_at"),
            runtime_version=_optional_int(row.get("runtime_version"), "runtime_version"),
            runtime_checkpoint_id=_optional_hash(
                row.get("runtime_checkpoint_id"),
                "runtime_checkpoint_id",
            ),
            runtime_head_state_id=_optional_hash(
                row.get("runtime_head_state_id"),
                "runtime_head_state_id",
            ),
            directive_exists=directive_exists,
            recovery_status=(
                None if row.get("recovery_status") is None else str(row.get("recovery_status"))
            ),
            directive_runtime_version=_optional_int(
                row.get("directive_runtime_version"),
                "directive_runtime_version",
            ),
            directive_state_id=_optional_hash(
                row.get("directive_state_id"),
                "directive_state_id",
            ),
            directive_prepared_at=row.get("directive_prepared_at"),
            claim_status=(
                None if row.get("claim_status") is None else str(row.get("claim_status"))
            ),
            claim_worker_token=(
                None
                if row.get("claim_worker_token") is None
                else str(row.get("claim_worker_token"))
            ),
            claim_fencing_token=_optional_int(
                row.get("claim_fencing_token"),
                "claim_fencing_token",
            ),
            claim_until=row.get("claim_until"),
            recovery_cycle_id=_optional_hash(
                row.get("recovery_cycle_id"),
                "recovery_cycle_id",
            ),
            progress_runtime_version=_optional_int(
                row.get("progress_runtime_version"),
                "progress_runtime_version",
            ),
            progress_head_state_id=_optional_hash(
                row.get("progress_head_state_id"),
                "progress_head_state_id",
            ),
            progress_checkpoint_id=_optional_hash(
                row.get("progress_checkpoint_id"),
                "progress_checkpoint_id",
            ),
            started=started,
            started_at=row.get("started_at"),
            recovery_journal_stage=(
                None
                if row.get("recovery_journal_stage") is None
                else str(row.get("recovery_journal_stage"))
            ),
            phase84_terminal_event=terminal_event,
        )
