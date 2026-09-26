from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .phase70_durable_runtime_store import RuntimeLease
from .phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from .phase77_execution_claim_lifecycle import (
    ClaimCompletionReceipt,
    ExecutionClaimError,
    ExecutionClaimReceipt,
    PersistedClaimedRuntimeSupervisor,
)
from .phase78_execution_kill_switch import (
    ExecutionKillSwitchDecision,
    ExecutionKillSwitchStore,
)

PHASE79_SCHEMA_VERSION = "brian.phase79-atomic-execution-start.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class AtomicExecutionStartError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AtomicExecutionStartReceipt:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    runtime_version: int
    fencing_token: int
    claim_fencing_token: int
    status: str
    started: bool
    duplicate: bool
    terminal: bool
    cancel_requested: bool
    risk_version: int | None
    risk_receipt_id: str | None
    risk_state: str | None
    journal_stage: str | None
    phase78_status: str | None
    reason: str | None
    resume_only: bool
    schema_version: str = PHASE79_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(self.cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")
        if self.dispatch_id and len(self.dispatch_id) != 64:
            raise ValueError("dispatch_id must be a content hash when present")
        if self.runtime_version < 0:
            raise ValueError("runtime_version cannot be negative")
        if self.fencing_token <= 0 or self.claim_fencing_token <= 0:
            raise ValueError("runtime/claim fencing token must be positive")
        if self.risk_version is not None and self.risk_version <= 0:
            raise ValueError("risk_version must be positive")
        if self.risk_receipt_id is not None and len(self.risk_receipt_id) != 64:
            raise ValueError("risk_receipt_id must be a content hash")
        if self.risk_state is not None and self.risk_state not in (
            "ACTIVE",
            "REDUCING",
            "HALTED",
        ):
            raise ValueError("invalid risk_state")
        if self.started and self.terminal:
            raise ValueError("started receipt cannot already be terminal")
        if self.duplicate and not self.started:
            raise ValueError("duplicate execution start must be started")
        if self.status in {"STARTED", "STARTED_RESUME", "STARTED_ALREADY"} and not self.started:
            raise ValueError(f"{self.status} requires started=true")
        if self.status == "STARTED_ALREADY" and not self.duplicate:
            raise ValueError("STARTED_ALREADY requires duplicate=true")
        if self.status == "STARTED_RESUME" and not self.resume_only:
            raise ValueError("STARTED_RESUME requires resume_only=true")
        if self.status == "STARTED" and self.resume_only:
            raise ValueError("fresh STARTED cannot be resume_only")
        if self.status in {
            "CANCELLED_BEFORE_EXECUTION",
            "ABORTED",
            "COMPLETED",
        } and not self.terminal:
            raise ValueError(f"{self.status} must be terminal")
        if self.started and (
            self.risk_version is None
            or self.risk_receipt_id is None
            or self.risk_state is None
            or self.journal_stage is None
            or self.phase78_status is None
        ):
            raise ValueError("STARTED receipt requires complete persisted risk/journal evidence")
        if self.cancel_requested and not self.reason:
            raise ValueError("cancel_requested requires a reason")
        if not self.shadow_only or self.live_execution:
            raise ValueError("execution-start receipt must remain shadow-only")


@dataclass(frozen=True, slots=True)
class StoredExecutionStart:
    runtime_id: str
    dispatch_id: str
    cycle_id: str
    worker_token: str
    claim_fencing_token: int
    runtime_fencing_token: int
    runtime_version_at_start: int
    risk_version_at_start: int
    risk_receipt_id_at_start: str
    risk_state_at_start: str
    journal_stage_at_start: str
    phase78_status_at_start: str
    cancel_requested_at_start: bool
    cancel_reason_at_start: str | None
    resume_only: bool
    started_at: object | None
    schema_version: str = PHASE79_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip() or not self.worker_token.strip():
            raise ValueError("runtime_id/worker_token are required")
        for label, value in (
            ("dispatch_id", self.dispatch_id),
            ("cycle_id", self.cycle_id),
            ("risk_receipt_id_at_start", self.risk_receipt_id_at_start),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be a content hash")
        if (
            self.claim_fencing_token <= 0
            or self.runtime_fencing_token <= 0
            or self.runtime_version_at_start <= 0
            or self.risk_version_at_start <= 0
        ):
            raise ValueError("stored start versions/fences must be positive")
        if self.risk_state_at_start not in ("ACTIVE", "REDUCING", "HALTED"):
            raise ValueError("invalid stored risk state")
        if self.cancel_requested_at_start and not self.cancel_reason_at_start:
            raise ValueError("cancel_requested_at_start requires reason")
        if not self.shadow_only or self.live_execution:
            raise ValueError("stored execution start crossed live boundary")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AtomicExecutionStartError(f"{label} returned non-object payload")
    return {str(key): item for key, item in value.items()}


def _integer(value: object, label: str, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        raise AtomicExecutionStartError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AtomicExecutionStartError(f"{label} must be integer") from exc


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise AtomicExecutionStartError(f"{label} must be boolean")
    return value


def _optional_hash(value: object, label: str) -> str | None:
    if value is None:
        return None
    result = str(value)
    if len(result) != 64:
        raise AtomicExecutionStartError(f"{label} must be a content hash")
    return result


class AtomicExecutionStartStore:
    """Transport adapter for Phase79's atomic Phase78-check + STARTED insert."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def mark_started(
        self,
        lease: RuntimeLease,
        claim: ExecutionClaimReceipt,
        *,
        worker_token: str,
    ) -> AtomicExecutionStartReceipt:
        if not lease.acquired:
            raise AtomicExecutionStartError("runtime lease is not acquired")
        if claim.runtime_id != lease.runtime_id:
            raise AtomicExecutionStartError("claim runtime does not match lease")
        if claim.fencing_token != lease.fencing_token:
            raise AtomicExecutionStartError("claim runtime fence does not match lease")
        if not claim.claimed or claim.claim_fencing_token <= 0:
            raise AtomicExecutionStartError("active owned claim is required")
        if not worker_token.strip():
            raise ValueError("worker_token is required")

        row = _mapping(
            self._rpc(
                "brian_mark_shadow_execution_started",
                {
                    "p_runtime_id": lease.runtime_id,
                    "p_owner_token": lease.owner_token,
                    "p_fencing_token": lease.fencing_token,
                    "p_cycle_id": claim.cycle_id,
                    "p_worker_token": worker_token,
                    "p_claim_fencing_token": claim.claim_fencing_token,
                },
            ),
            "brian_mark_shadow_execution_started",
        )

        if str(row.get("runtime_id", "")) != lease.runtime_id:
            raise AtomicExecutionStartError("start runtime_id mismatch")
        if str(row.get("cycle_id", "")) != claim.cycle_id:
            raise AtomicExecutionStartError("start cycle_id mismatch")

        started = _boolean(row.get("started"), "started")
        duplicate = bool(row.get("duplicate", False))
        terminal = _boolean(row.get("terminal"), "terminal")
        cancel_requested = _boolean(
            row.get("cancel_requested"),
            "cancel_requested",
        )
        status = str(row.get("status", ""))

        receipt = AtomicExecutionStartReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=claim.cycle_id,
            dispatch_id=str(row.get("dispatch_id", claim.dispatch_id)),
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
            claim_fencing_token=_integer(
                row.get("claim_fencing_token"),
                "claim_fencing_token",
                default=claim.claim_fencing_token,
            ),
            status=status,
            started=started,
            duplicate=duplicate,
            terminal=terminal,
            cancel_requested=cancel_requested,
            risk_version=(
                None
                if row.get("risk_version") is None
                else _integer(row.get("risk_version"), "risk_version")
            ),
            risk_receipt_id=_optional_hash(
                row.get("risk_receipt_id"),
                "risk_receipt_id",
            ),
            risk_state=(
                None
                if row.get("risk_state") is None
                else str(row.get("risk_state"))
            ),
            journal_stage=(
                None
                if row.get("journal_stage") is None
                else str(row.get("journal_stage"))
            ),
            phase78_status=(
                None
                if row.get("phase78_status") is None
                else str(row.get("phase78_status"))
            ),
            reason=None if row.get("reason") is None else str(row.get("reason")),
            resume_only=bool(row.get("resume_only", False)),
        )
        if receipt.dispatch_id != claim.dispatch_id:
            raise AtomicExecutionStartError(
                "database STARTED dispatch_id does not match owned claim"
            )
        if receipt.fencing_token != lease.fencing_token:
            raise AtomicExecutionStartError(
                "database STARTED runtime fencing token does not match lease"
            )
        if receipt.claim_fencing_token != claim.claim_fencing_token:
            raise AtomicExecutionStartError(
                "database STARTED claim fencing token does not match owned claim"
            )
        return receipt

    def load(
        self,
        *,
        runtime_id: str,
        cycle_id: str,
    ) -> StoredExecutionStart | None:
        if not runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")

        result = self._rpc(
            "brian_read_shadow_execution_start",
            {
                "p_runtime_id": runtime_id,
                "p_cycle_id": cycle_id,
            },
        )
        if result is None:
            return None
        row = _mapping(result, "brian_read_shadow_execution_start")
        if str(row.get("runtime_id", "")) != runtime_id:
            raise AtomicExecutionStartError("stored start runtime_id mismatch")
        if str(row.get("cycle_id", "")) != cycle_id:
            raise AtomicExecutionStartError("stored start cycle_id mismatch")
        if row.get("shadow_only") is not True or row.get("live_execution") is not False:
            raise AtomicExecutionStartError("stored execution start crossed live boundary")

        return StoredExecutionStart(
            runtime_id=runtime_id,
            dispatch_id=str(row.get("dispatch_id", "")),
            cycle_id=cycle_id,
            worker_token=str(row.get("worker_token", "")),
            claim_fencing_token=_integer(
                row.get("claim_fencing_token"),
                "claim_fencing_token",
            ),
            runtime_fencing_token=_integer(
                row.get("runtime_fencing_token"),
                "runtime_fencing_token",
            ),
            runtime_version_at_start=_integer(
                row.get("runtime_version_at_start"),
                "runtime_version_at_start",
            ),
            risk_version_at_start=_integer(
                row.get("risk_version_at_start"),
                "risk_version_at_start",
            ),
            risk_receipt_id_at_start=str(
                row.get("risk_receipt_id_at_start", "")
            ),
            risk_state_at_start=str(row.get("risk_state_at_start", "")),
            journal_stage_at_start=str(row.get("journal_stage_at_start", "")),
            phase78_status_at_start=str(
                row.get("phase78_status_at_start", "")
            ),
            cancel_requested_at_start=_boolean(
                row.get("cancel_requested_at_start"),
                "cancel_requested_at_start",
            ),
            cancel_reason_at_start=(
                None
                if row.get("cancel_reason_at_start") is None
                else str(row.get("cancel_reason_at_start"))
            ),
            resume_only=bool(row.get("resume_only", False)),
            started_at=row.get("started_at"),
            shadow_only=True,
            live_execution=False,
        )


@dataclass(frozen=True, slots=True)
class AtomicStartedExecutionStep:
    claim: ExecutionClaimReceipt
    start: AtomicExecutionStartReceipt | None
    postcheck: ExecutionKillSwitchDecision | None
    outcome: str
    completion: ClaimCompletionReceipt | None
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE79_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


class PersistedAtomicStartedRuntimeSupervisor:
    """Require Phase79 STARTED before any Phase77 paper/recovery advance."""

    def __init__(
        self,
        *,
        claimed_supervisor: PersistedClaimedRuntimeSupervisor,
        starts: AtomicExecutionStartStore,
        kill_switch: ExecutionKillSwitchStore,
    ) -> None:
        self.claimed_supervisor = claimed_supervisor
        self.starts = starts
        self.kill_switch = kill_switch

    def _runtime_supervisor(self):
        return (
            self.claimed_supervisor
            .dispatched_supervisor
            .governed_supervisor
            .runtime_supervisor
        )

    def process_governed_cycle(
        self,
        governed,
        *,
        worker_token: str,
        claim_seconds: int,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> AtomicStartedExecutionStep:
        phase77 = self.claimed_supervisor
        phase76 = phase77.dispatched_supervisor
        phase75 = phase76.governed_supervisor
        supervisor = self._runtime_supervisor()

        _, _, claim, terminal = phase77.authorize_submit_and_claim(
            governed,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
        )

        if terminal is not None:
            checkpoint = supervisor.runtime.checkpoint()
            return AtomicStartedExecutionStep(
                claim=claim,
                start=None,
                postcheck=None,
                outcome=terminal,
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        start = self.starts.mark_started(
            supervisor.lease,
            claim,
            worker_token=worker_token,
        )

        if start.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost at atomic execution-start boundary"
            )
        if start.status == "RISK_STATE_UNAVAILABLE":
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "risk state unavailable at atomic execution-start boundary"
            )
        if start.status == "CLAIM_LOST":
            raise ExecutionClaimError(
                "execution claim lost before STARTED; acquire a fresh claim"
            )

        if start.status == "CANCELLED_BEFORE_EXECUTION":
            if supervisor.runtime.journal.latest_stage(claim.cycle_id) == "CYCLE_CREATED":
                phase75.abort_authorized_cycle(
                    cycle_id=claim.cycle_id,
                    reason=f"phase79:{start.reason or 'phase78_cancel'}",
                )
            checkpoint = supervisor.runtime.checkpoint()
            return AtomicStartedExecutionStep(
                claim=claim,
                start=start,
                postcheck=None,
                outcome="CANCELLED_BEFORE_EXECUTION",
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        if start.status in {"ABORTED", "COMPLETED"}:
            checkpoint = supervisor.runtime.checkpoint()
            return AtomicStartedExecutionStep(
                claim=claim,
                start=start,
                postcheck=None,
                outcome=(
                    "ALREADY_COMPLETED"
                    if start.status == "COMPLETED"
                    else "ABORTED"
                ),
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        if not start.started:
            raise AtomicExecutionStartError(
                f"unexpected non-started status {start.status}"
            )
        if start.fencing_token != supervisor.lease.fencing_token:
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "STARTED runtime fence differs from current lease"
            )
        if start.claim_fencing_token != claim.claim_fencing_token:
            raise ExecutionClaimError(
                "STARTED claim fence differs from owned claim"
            )

        _, outcome, _ = phase77.advance_claimed(
            claim,
            worker_token=worker_token,
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
            complete_on_commit=False,
        )

        postcheck = self.kill_switch.check(
            supervisor.lease,
            claim,
            worker_token=worker_token,
        )

        if postcheck.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost during Phase79 post-start check"
            )
        if postcheck.status in {"CLAIM_LOST", "RISK_STATE_UNAVAILABLE"}:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"post-start check failed with {postcheck.status}"
            )

        if postcheck.status == "CANCEL_REQUESTED":
            outcome = f"{outcome}_CANCEL_REQUESTED"
        elif postcheck.status == "CANCELLED_BEFORE_EXECUTION":
            # STARTED proves the point-of-no-return was crossed. A later
            # pre-execution cancellation response would contradict durable
            # history, so fail closed rather than rewriting it.
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "post-start evidence contradiction: kill switch attempted "
                "pre-execution cancellation"
            )

        completion = None
        if outcome.startswith("COMMITTED") or postcheck.status == "COMPLETED":
            completion = phase77.claims.complete(
                supervisor.lease,
                claim,
                worker_token=worker_token,
            )
            if not completion.completed:
                outcome = f"COMMITTED_{completion.status}"
            elif postcheck.cancel_requested:
                outcome = "COMMITTED_CANCEL_REQUESTED"
            else:
                outcome = "COMMITTED"

        checkpoint = supervisor.runtime.checkpoint()
        return AtomicStartedExecutionStep(
            claim=claim,
            start=start,
            postcheck=postcheck,
            outcome=outcome,
            completion=completion,
            persisted_version=supervisor.persisted_version,
            checkpoint_id=checkpoint.checkpoint_id,
        )
