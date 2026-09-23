from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .phase70_durable_runtime_store import RuntimeLease
from .phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from .phase76_shadow_execution_outbox import (
    PersistedDispatchedRuntimeSupervisor,
    ShadowExecutionDispatchReceipt,
)

PHASE77_SCHEMA_VERSION = "brian.phase77-execution-claim-lifecycle.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class ExecutionClaimError(RuntimeError):
    pass


class ExecutionClaimBusyError(ExecutionClaimError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutionClaimReceipt:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    runtime_version: int
    fencing_token: int
    claim_fencing_token: int
    status: str
    claimed: bool
    cancelled: bool
    terminal: bool
    risk_version: int | None
    risk_receipt_id: str | None
    journal_stage: str | None
    resume_only: bool
    cancel_reason: str | None
    claim_until: object | None
    completion_checkpoint_id: str | None
    schema_version: str = PHASE77_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(self.cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")
        if self.dispatch_id and len(self.dispatch_id) != 64:
            raise ValueError("dispatch_id must be a content hash when present")
        if self.runtime_version < 0 or self.fencing_token <= 0:
            raise ValueError("invalid runtime version/fencing token")
        if self.claim_fencing_token < 0:
            raise ValueError("claim_fencing_token cannot be negative")
        if self.risk_version is not None and self.risk_version <= 0:
            raise ValueError("risk_version must be positive when present")
        if self.risk_receipt_id is not None and len(self.risk_receipt_id) != 64:
            raise ValueError("risk_receipt_id must be a content hash")
        if self.completion_checkpoint_id is not None and len(self.completion_checkpoint_id) != 64:
            raise ValueError("completion_checkpoint_id must be a content hash")
        if not self.shadow_only or self.live_execution:
            raise ValueError("claim receipt must remain shadow-only")


@dataclass(frozen=True, slots=True)
class ClaimRenewReceipt:
    runtime_id: str
    cycle_id: str
    dispatch_id: str | None
    claim_fencing_token: int
    status: str
    renewed: bool
    claim_until: object | None
    schema_version: str = PHASE77_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ClaimCompletionReceipt:
    runtime_id: str
    cycle_id: str
    dispatch_id: str | None
    claim_fencing_token: int
    status: str
    completed: bool
    duplicate: bool
    completion_checkpoint_id: str | None
    schema_version: str = PHASE77_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class StoredExecutionClaim:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    status: str
    worker_token: str | None
    claim_fencing_token: int
    claim_until: object | None
    risk_version_at_decision: int | None
    risk_receipt_id_at_decision: str | None
    journal_stage_at_decision: str | None
    resume_only: bool
    cancel_reason: str | None
    completion_checkpoint_id: str | None
    schema_version: str = PHASE77_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ExecutionClaimError(f"{label} returned non-object payload")
    return {str(key): item for key, item in value.items()}


def _integer(value: object, label: str, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        raise ExecutionClaimError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ExecutionClaimError(f"{label} must be integer") from exc


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ExecutionClaimError(f"{label} must be boolean")
    return value


class ExecutionClaimStore:
    """Fenced Phase77 worker-claim client over the Phase70 runtime lease."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def claim(
        self,
        lease: RuntimeLease,
        *,
        cycle_id: str,
        worker_token: str,
        claim_seconds: int,
    ) -> ExecutionClaimReceipt:
        if not lease.acquired:
            raise ExecutionClaimError("runtime lease is not acquired")
        if len(cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")
        if not worker_token.strip():
            raise ValueError("worker_token is required")
        if claim_seconds <= 0:
            raise ValueError("claim_seconds must be positive")

        row = _mapping(
            self._rpc(
                "brian_claim_shadow_execution_dispatch",
                {
                    "p_runtime_id": lease.runtime_id,
                    "p_owner_token": lease.owner_token,
                    "p_fencing_token": lease.fencing_token,
                    "p_cycle_id": cycle_id,
                    "p_worker_token": worker_token,
                    "p_claim_seconds": int(claim_seconds),
                },
            ),
            "brian_claim_shadow_execution_dispatch",
        )
        if str(row.get("runtime_id", "")) != lease.runtime_id:
            raise ExecutionClaimError("claim runtime_id mismatch")
        if str(row.get("cycle_id", "")) != cycle_id:
            raise ExecutionClaimError("claim cycle_id mismatch")

        claimed = _boolean(row.get("claimed"), "claimed")
        cancelled = _boolean(row.get("cancelled"), "cancelled")
        terminal = _boolean(row.get("terminal"), "terminal")
        status = str(row.get("status", ""))
        dispatch_id = str(row.get("dispatch_id", ""))

        if claimed and (cancelled or terminal):
            raise ExecutionClaimError("claimed receipt cannot be cancelled/terminal")
        if status in {"CLAIMED", "CLAIMED_RESUME", "ALREADY_CLAIMED"} and not claimed:
            raise ExecutionClaimError(f"{status} must return claimed=true")
        if status == "CANCELLED_BEFORE_EXECUTION" and not (cancelled and terminal):
            raise ExecutionClaimError("cancelled claim receipt flags are inconsistent")
        if status == "COMPLETED" and not terminal:
            raise ExecutionClaimError("completed claim must be terminal")
        if status in {"BLOCKED_ACTIVE", "LEASE_LOST", "DISPATCH_MISSING", "RISK_STATE_UNAVAILABLE"} and claimed:
            raise ExecutionClaimError(f"{status} cannot claim execution")

        return ExecutionClaimReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=cycle_id,
            dispatch_id=dispatch_id,
            runtime_version=_integer(row.get("runtime_version"), "runtime_version", default=lease.version),
            fencing_token=_integer(row.get("fencing_token"), "fencing_token", default=lease.fencing_token),
            claim_fencing_token=_integer(
                row.get("claim_fencing_token"),
                "claim_fencing_token",
                default=0,
            ),
            status=status,
            claimed=claimed,
            cancelled=cancelled,
            terminal=terminal,
            risk_version=(
                None
                if row.get("risk_version") is None
                else _integer(row.get("risk_version"), "risk_version")
            ),
            risk_receipt_id=(
                None
                if row.get("risk_receipt_id") is None
                else str(row.get("risk_receipt_id"))
            ),
            journal_stage=(
                None if row.get("journal_stage") is None else str(row.get("journal_stage"))
            ),
            resume_only=bool(row.get("resume_only", False)),
            cancel_reason=(
                None if row.get("cancel_reason") is None else str(row.get("cancel_reason"))
            ),
            claim_until=row.get("claim_until"),
            completion_checkpoint_id=(
                None
                if row.get("completion_checkpoint_id") is None
                else str(row.get("completion_checkpoint_id"))
            ),
        )

    def renew(
        self,
        lease: RuntimeLease,
        claim: ExecutionClaimReceipt,
        *,
        worker_token: str,
        claim_seconds: int,
    ) -> ClaimRenewReceipt:
        if not claim.claimed or claim.claim_fencing_token <= 0:
            raise ExecutionClaimError("only an active claim can be renewed")
        row = _mapping(
            self._rpc(
                "brian_renew_shadow_execution_claim",
                {
                    "p_runtime_id": lease.runtime_id,
                    "p_owner_token": lease.owner_token,
                    "p_fencing_token": lease.fencing_token,
                    "p_cycle_id": claim.cycle_id,
                    "p_worker_token": worker_token,
                    "p_claim_fencing_token": claim.claim_fencing_token,
                    "p_claim_seconds": int(claim_seconds),
                },
            ),
            "brian_renew_shadow_execution_claim",
        )
        return ClaimRenewReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=claim.cycle_id,
            dispatch_id=(
                None if row.get("dispatch_id") is None else str(row.get("dispatch_id"))
            ),
            claim_fencing_token=claim.claim_fencing_token,
            status=str(row.get("status", "")),
            renewed=_boolean(row.get("renewed"), "renewed"),
            claim_until=row.get("claim_until"),
        )

    def complete(
        self,
        lease: RuntimeLease,
        claim: ExecutionClaimReceipt,
        *,
        worker_token: str,
    ) -> ClaimCompletionReceipt:
        if not claim.claimed or claim.claim_fencing_token <= 0:
            raise ExecutionClaimError("only an owned claim can be completed")
        row = _mapping(
            self._rpc(
                "brian_complete_shadow_execution_claim",
                {
                    "p_runtime_id": lease.runtime_id,
                    "p_owner_token": lease.owner_token,
                    "p_fencing_token": lease.fencing_token,
                    "p_cycle_id": claim.cycle_id,
                    "p_worker_token": worker_token,
                    "p_claim_fencing_token": claim.claim_fencing_token,
                },
            ),
            "brian_complete_shadow_execution_claim",
        )
        return ClaimCompletionReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=claim.cycle_id,
            dispatch_id=(
                None if row.get("dispatch_id") is None else str(row.get("dispatch_id"))
            ),
            claim_fencing_token=claim.claim_fencing_token,
            status=str(row.get("status", "")),
            completed=_boolean(row.get("completed"), "completed"),
            duplicate=bool(row.get("duplicate", False)),
            completion_checkpoint_id=(
                None
                if row.get("completion_checkpoint_id") is None
                else str(row.get("completion_checkpoint_id"))
            ),
        )

    def load(
        self,
        *,
        runtime_id: str,
        cycle_id: str,
    ) -> StoredExecutionClaim | None:
        result = self._rpc(
            "brian_read_shadow_execution_claim",
            {"p_runtime_id": runtime_id, "p_cycle_id": cycle_id},
        )
        if result is None:
            return None
        row = _mapping(result, "brian_read_shadow_execution_claim")
        if row.get("shadow_only") is not True or row.get("live_execution") is not False:
            raise ExecutionClaimError("stored claim crossed live boundary")
        return StoredExecutionClaim(
            runtime_id=runtime_id,
            cycle_id=cycle_id,
            dispatch_id=str(row.get("dispatch_id", "")),
            status=str(row.get("status", "")),
            worker_token=(
                None if row.get("worker_token") is None else str(row.get("worker_token"))
            ),
            claim_fencing_token=_integer(
                row.get("claim_fencing_token"),
                "claim_fencing_token",
                default=0,
            ),
            claim_until=row.get("claim_until"),
            risk_version_at_decision=(
                None
                if row.get("risk_version_at_decision") is None
                else _integer(row.get("risk_version_at_decision"), "risk_version_at_decision")
            ),
            risk_receipt_id_at_decision=(
                None
                if row.get("risk_receipt_id_at_decision") is None
                else str(row.get("risk_receipt_id_at_decision"))
            ),
            journal_stage_at_decision=(
                None
                if row.get("journal_stage_at_decision") is None
                else str(row.get("journal_stage_at_decision"))
            ),
            resume_only=bool(row.get("resume_only", False)),
            cancel_reason=(
                None if row.get("cancel_reason") is None else str(row.get("cancel_reason"))
            ),
            completion_checkpoint_id=(
                None
                if row.get("completion_checkpoint_id") is None
                else str(row.get("completion_checkpoint_id"))
            ),
        )


@dataclass(frozen=True, slots=True)
class ClaimedExecutionStep:
    dispatch: ShadowExecutionDispatchReceipt
    claim: ExecutionClaimReceipt
    outcome: str
    completion: ClaimCompletionReceipt | None
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE77_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


class PersistedClaimedRuntimeSupervisor:
    """Require a fenced Phase77 claim before Phase67/71 execution can advance."""

    def __init__(
        self,
        *,
        dispatched_supervisor: PersistedDispatchedRuntimeSupervisor,
        claims: ExecutionClaimStore,
    ) -> None:
        self.dispatched_supervisor = dispatched_supervisor
        self.claims = claims

    def process_governed_cycle(
        self,
        governed,
        *,
        worker_token: str,
        claim_seconds: int,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> ClaimedExecutionStep:
        phase76 = self.dispatched_supervisor
        phase75 = phase76.governed_supervisor
        supervisor = phase75.runtime_supervisor

        authorization, dispatch, pre_submit_terminal = phase76.authorize_and_submit(governed)
        if pre_submit_terminal is not None:
            raise ExecutionClaimError(
                f"cycle terminated before dispatch claim: {pre_submit_terminal}"
            )

        claim = self.claims.claim(
            supervisor.lease,
            cycle_id=authorization.cycle_id,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
        )

        if claim.status == "BLOCKED_ACTIVE":
            raise ExecutionClaimBusyError("execution dispatch is claimed by another worker")
        if claim.status in {"LEASE_LOST", "DISPATCH_MISSING", "RISK_STATE_UNAVAILABLE"}:
            supervisor._valid = False
            if claim.status == "LEASE_LOST":
                raise PersistedRuntimeLeaseError("runtime lease lost while claiming dispatch")
            raise PersistedRuntimeStaleError(
                f"execution claim rejected with {claim.status}; reload required"
            )

        if claim.cancelled:
            if supervisor.runtime.journal.latest_stage(authorization.cycle_id) == "CYCLE_CREATED":
                phase75.abort_authorized_cycle(
                    cycle_id=authorization.cycle_id,
                    reason=f"phase77:{claim.cancel_reason or 'risk_cancel'}",
                )
            checkpoint = supervisor.runtime.checkpoint()
            return ClaimedExecutionStep(
                dispatch=dispatch,
                claim=claim,
                outcome="CANCELLED_BEFORE_EXECUTION",
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        if claim.status == "COMPLETED":
            checkpoint = supervisor.runtime.checkpoint()
            return ClaimedExecutionStep(
                dispatch=dispatch,
                claim=claim,
                outcome="ALREADY_COMPLETED",
                completion=None,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        if not claim.claimed:
            raise ExecutionClaimError(f"unexpected non-claimed status {claim.status}")

        advanced = phase76.advance_submitted(
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )
        checkpoint = supervisor.runtime.checkpoint()
        outcome = (
            "NONE"
            if advanced.durable_receipt is None
            else advanced.durable_receipt.status
        )

        completion = None
        if outcome == "COMMITTED":
            completion = self.claims.complete(
                supervisor.lease,
                claim,
                worker_token=worker_token,
            )
            if not completion.completed:
                # Runtime state is already durable/committed. A later claim call
                # will recover COMPLETED from the journal; never re-run execution.
                outcome = f"COMMITTED_{completion.status}"

        return ClaimedExecutionStep(
            dispatch=dispatch,
            claim=claim,
            outcome=outcome,
            completion=completion,
            persisted_version=supervisor.persisted_version,
            checkpoint_id=checkpoint.checkpoint_id,
        )
