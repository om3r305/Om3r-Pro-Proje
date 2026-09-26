from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from .phase81_cancel_recovery_directive import CancelRecoveryLeg
from .phase82_recovery_claim_fencing import (
    PersistedRecoveryClaimSupervisor,
    RecoveryClaimError,
    RecoveryClaimExecutionStep,
    RecoveryClaimReceipt,
)

PHASE83_SCHEMA_VERSION = "brian.phase83-atomic-recovery-start.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class AtomicRecoveryStartError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AtomicRecoveryStartReceipt:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    cancel_risk_receipt_id: str
    runtime_version: int
    head_state_id: str | None
    fencing_token: int
    recovery_claim_fencing_token: int
    status: str
    started: bool
    duplicate: bool
    resume_only: bool
    risk_version: int | None = None
    risk_receipt_id: str | None = None
    risk_state: str | None = None
    recovery_legs: tuple[CancelRecoveryLeg, ...] = ()
    schema_version: str = PHASE83_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(self.cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")
        if self.dispatch_id and len(self.dispatch_id) != 64:
            raise ValueError("dispatch_id must be a content hash")
        if self.cancel_risk_receipt_id and len(self.cancel_risk_receipt_id) != 64:
            raise ValueError("cancel_risk_receipt_id must be a content hash")
        if self.runtime_version < 0:
            raise ValueError("runtime_version cannot be negative")
        if self.head_state_id is not None and len(self.head_state_id) != 64:
            raise ValueError("head_state_id must be a content hash")
        if self.fencing_token <= 0 or self.recovery_claim_fencing_token <= 0:
            raise ValueError("runtime/recovery claim fences must be positive")
        if self.started:
            if self.status not in {"STARTED", "STARTED_ALREADY", "STARTED_RESUME"}:
                raise ValueError("started recovery has unsupported status")
            if not self.recovery_legs:
                raise ValueError("started recovery requires immutable recovery legs")
            if self.risk_state not in {"ACTIVE", "REDUCING"}:
                raise ValueError("started recovery requires ACTIVE/REDUCING risk")
            if self.risk_version is None or self.risk_version <= 0:
                raise ValueError("started recovery requires risk version")
            if self.risk_receipt_id is None or len(self.risk_receipt_id) != 64:
                raise ValueError("started recovery requires risk receipt")
        if self.status == "STARTED" and (self.duplicate or self.resume_only):
            raise ValueError("fresh STARTED flags are inconsistent")
        if self.status == "STARTED_ALREADY" and not (
            self.started and self.duplicate and not self.resume_only
        ):
            raise ValueError("STARTED_ALREADY flags are inconsistent")
        if self.status == "STARTED_RESUME" and not (
            self.started and self.duplicate and self.resume_only
        ):
            raise ValueError("STARTED_RESUME flags are inconsistent")
        if self.status == "WAIT_RISK_RELEASE" and self.started:
            raise ValueError("HALTED recovery cannot cross STARTED")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase83 start must remain shadow-only")


@dataclass(frozen=True, slots=True)
class AtomicRecoveryStartExecutionStep:
    claim_step: RecoveryClaimExecutionStep
    start: AtomicRecoveryStartReceipt | None
    outcome: str
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE83_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AtomicRecoveryStartError(f"{label} returned non-object payload")
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
        raise AtomicRecoveryStartError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AtomicRecoveryStartError(f"{label} must be integer") from exc


def _optional_hash(value: object, label: str) -> str | None:
    if value is None:
        return None
    result = str(value)
    if len(result) != 64:
        raise AtomicRecoveryStartError(f"{label} must be a content hash")
    return result


def _leg(value: object) -> CancelRecoveryLeg:
    row = _mapping(value, "recovery start leg")
    return CancelRecoveryLeg(
        asset_id=str(row.get("asset_id", "")),
        before_weight=float(row.get("before_weight")),
        current_weight=float(row.get("current_weight")),
        target_weight=float(row.get("target_weight")),
        reduce_weight=float(row.get("reduce_weight")),
        current_direction=_integer(
            row.get("current_direction"),
            "current_direction",
        ),
        order_direction=_integer(
            row.get("order_direction"),
            "order_direction",
        ),
        reduce_only=bool(row.get("reduce_only", False)),
    )


class AtomicRecoveryStartStore:
    """Final atomic point-of-no-return before any recovery paper side effect."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def mark_started(
        self,
        lease,
        claim: RecoveryClaimReceipt,
        *,
        worker_token: str,
    ) -> AtomicRecoveryStartReceipt:
        if not lease.acquired:
            raise AtomicRecoveryStartError("runtime lease is not acquired")
        if not claim.claimed or claim.claim_fencing_token is None:
            raise AtomicRecoveryStartError("active recovery claim is required")
        if claim.runtime_id != lease.runtime_id:
            raise AtomicRecoveryStartError("recovery claim runtime does not match lease")
        if claim.fencing_token != lease.fencing_token:
            raise AtomicRecoveryStartError("recovery claim runtime fence does not match lease")
        if claim.worker_token != worker_token:
            raise AtomicRecoveryStartError("recovery claim worker does not match requester")

        row = _mapping(
            self._rpc(
                "brian_mark_shadow_recovery_started",
                {
                    "p_runtime_id": lease.runtime_id,
                    "p_owner_token": lease.owner_token,
                    "p_fencing_token": lease.fencing_token,
                    "p_cycle_id": claim.cycle_id,
                    "p_worker_token": worker_token,
                    "p_recovery_claim_fencing_token": claim.claim_fencing_token,
                },
            ),
            "brian_mark_shadow_recovery_started",
        )

        if str(row.get("runtime_id", "")) != lease.runtime_id:
            raise AtomicRecoveryStartError("recovery STARTED runtime_id mismatch")
        if str(row.get("cycle_id", "")) != claim.cycle_id:
            raise AtomicRecoveryStartError("recovery STARTED cycle_id mismatch")

        started_raw = row.get("started")
        if not isinstance(started_raw, bool):
            raise AtomicRecoveryStartError("started must be boolean")
        started = started_raw
        duplicate = bool(row.get("duplicate", False))
        resume_only = bool(row.get("resume_only", False))
        status = str(row.get("status", ""))

        raw_legs = row.get("recovery_legs", ())
        if raw_legs is None:
            raw_legs = ()
        if not isinstance(raw_legs, (list, tuple)):
            raise AtomicRecoveryStartError("recovery_legs must be an array")

        receipt = AtomicRecoveryStartReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=claim.cycle_id,
            dispatch_id=str(row.get("dispatch_id", claim.dispatch_id)),
            cancel_risk_receipt_id=str(
                row.get("cancel_risk_receipt_id", claim.cancel_risk_receipt_id)
            ),
            runtime_version=_integer(
                row.get("runtime_version"),
                "runtime_version",
                default=claim.runtime_version,
            ),
            head_state_id=_optional_hash(
                row.get("head_state_id"),
                "head_state_id",
            ),
            fencing_token=_integer(
                row.get("fencing_token"),
                "fencing_token",
                default=lease.fencing_token,
            ),
            recovery_claim_fencing_token=_integer(
                row.get("recovery_claim_fencing_token"),
                "recovery_claim_fencing_token",
                default=claim.claim_fencing_token,
            ),
            status=status,
            started=started,
            duplicate=duplicate,
            resume_only=resume_only,
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
                None if row.get("risk_state") is None else str(row.get("risk_state"))
            ),
            recovery_legs=tuple(_leg(value) for value in raw_legs),
        )

        if receipt.dispatch_id and receipt.dispatch_id != claim.dispatch_id:
            raise AtomicRecoveryStartError("recovery STARTED dispatch_id drift")
        if (
            receipt.cancel_risk_receipt_id
            and receipt.cancel_risk_receipt_id != claim.cancel_risk_receipt_id
        ):
            raise AtomicRecoveryStartError(
                "recovery STARTED cancel-risk receipt drift"
            )
        if receipt.fencing_token != lease.fencing_token:
            raise AtomicRecoveryStartError("recovery STARTED runtime fence drift")
        if receipt.recovery_claim_fencing_token != claim.claim_fencing_token:
            raise AtomicRecoveryStartError("recovery STARTED claim fence drift")

        if receipt.started:
            if tuple(leg.to_dict() for leg in receipt.recovery_legs) != tuple(
                leg.to_dict() for leg in claim.recovery_legs
            ):
                raise AtomicRecoveryStartError(
                    "recovery STARTED legs differ from claimed directive"
                )
        return receipt


class PersistedAtomicRecoveryStartSupervisor:
    """Phase82 single-owner recovery claim plus Phase83 atomic STARTED barrier."""

    def __init__(
        self,
        *,
        claim_supervisor: PersistedRecoveryClaimSupervisor,
        starts: AtomicRecoveryStartStore,
    ) -> None:
        self.claim_supervisor = claim_supervisor
        self.starts = starts

    def _runtime_supervisor(self):
        return (
            self.claim_supervisor
            .recovery_supervisor
            .execution_supervisor
            .claimed_supervisor
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
        recovery_worker_token: str,
        recovery_claim_seconds: int,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> AtomicRecoveryStartExecutionStep:
        claim_step = self.claim_supervisor.process_governed_cycle(
            governed,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
            recovery_worker_token=recovery_worker_token,
            recovery_claim_seconds=recovery_claim_seconds,
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )
        supervisor = self._runtime_supervisor()
        claim = claim_step.claim

        if claim is None or not claim.claimed:
            return AtomicRecoveryStartExecutionStep(
                claim_step=claim_step,
                start=None,
                outcome=claim_step.outcome,
                persisted_version=claim_step.persisted_version,
                checkpoint_id=claim_step.checkpoint_id,
            )

        start = self.starts.mark_started(
            supervisor.lease,
            claim,
            worker_token=recovery_worker_token,
        )

        if start.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost at recovery STARTED boundary"
            )
        if start.status in {
            "HEAD_MOVED",
            "DIRECTIVE_MISSING",
            "RISK_STATE_UNAVAILABLE",
            "EVIDENCE_INVALID",
        }:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"recovery STARTED rejected with {start.status}"
            )
        if start.status == "CLAIM_LOST":
            raise RecoveryClaimError(
                "recovery claim lost before STARTED; acquire a fresh recovery claim"
            )

        outcome = claim_step.outcome
        if start.status == "WAIT_RISK_RELEASE":
            outcome = f"{outcome}_START_WAIT_RISK_RELEASE"
        elif start.status in {"STARTED", "STARTED_ALREADY"}:
            outcome = f"{outcome}_RECOVERY_STARTED"
        elif start.status == "STARTED_RESUME":
            outcome = f"{outcome}_RECOVERY_RESUME"

        return AtomicRecoveryStartExecutionStep(
            claim_step=claim_step,
            start=start,
            outcome=outcome,
            persisted_version=supervisor.persisted_version,
            checkpoint_id=supervisor.runtime.checkpoint().checkpoint_id,
        )
