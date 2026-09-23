from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from .phase81_cancel_recovery_directive import (
    CancelRecoveryLeg,
    PersistedRecoveryObligationSupervisor,
    RecoveryObligationExecutionStep,
)

PHASE82_SCHEMA_VERSION = "brian.phase82-recovery-claim-fencing.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class RecoveryClaimError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryClaimReceipt:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    cancel_risk_receipt_id: str
    runtime_version: int
    head_state_id: str | None
    fencing_token: int
    claim_fencing_token: int | None
    status: str
    claimed: bool
    terminal: bool
    worker_token: str | None = None
    claim_until: object | None = None
    risk_version: int | None = None
    risk_receipt_id: str | None = None
    risk_state: str | None = None
    recovery_status: str | None = None
    recovery_legs: tuple[CancelRecoveryLeg, ...] = ()
    completion_ref: str | None = None
    schema_version: str = PHASE82_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(self.cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")
        if self.dispatch_id and len(self.dispatch_id) != 64:
            raise ValueError("dispatch_id must be a content hash when present")
        if self.cancel_risk_receipt_id and len(self.cancel_risk_receipt_id) != 64:
            raise ValueError("cancel_risk_receipt_id must be a content hash")
        if self.runtime_version < 0:
            raise ValueError("runtime_version must be non-negative")
        if self.head_state_id is not None and len(self.head_state_id) != 64:
            raise ValueError("head_state_id must be a content hash")
        if self.fencing_token <= 0:
            raise ValueError("fencing_token must be positive")
        if self.claimed:
            if self.claim_fencing_token is None or self.claim_fencing_token <= 0:
                raise ValueError("claimed recovery requires positive claim fence")
            if not self.worker_token:
                raise ValueError("claimed recovery requires worker_token")
            if self.terminal:
                raise ValueError("claimed recovery cannot already be terminal")
            if self.status not in {"CLAIMED", "ALREADY_OWNED", "EXPIRED_RECOVERY"}:
                raise ValueError("claimed recovery returned unsupported status")
            if not self.recovery_legs:
                raise ValueError("claimed recovery requires reduce-only legs")
            if self.risk_state == "HALTED":
                raise ValueError("HALTED recovery cannot be claimed for execution")
        if self.status in {"NO_RECOVERY_REQUIRED", "MANUAL_REVIEW", "COMPLETED"}:
            if not self.terminal:
                raise ValueError(f"{self.status} must be terminal")
            if self.claimed:
                raise ValueError(f"{self.status} cannot be actively claimed")
        if self.status == "WAIT_RISK_RELEASE" and (self.claimed or self.terminal):
            raise ValueError("WAIT_RISK_RELEASE must remain non-terminal/unclaimed")
        if self.risk_version is not None and self.risk_version <= 0:
            raise ValueError("risk_version must be positive")
        if self.risk_receipt_id is not None and len(self.risk_receipt_id) != 64:
            raise ValueError("risk_receipt_id must be a content hash")
        if self.risk_state is not None and self.risk_state not in {
            "ACTIVE",
            "REDUCING",
            "HALTED",
        }:
            raise ValueError("invalid recovery claim risk_state")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase82 claim must remain shadow-only")


@dataclass(frozen=True, slots=True)
class RecoveryClaimRenewal:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    runtime_version: int
    head_state_id: str | None
    fencing_token: int
    claim_fencing_token: int
    status: str
    renewed: bool
    claim_until: object | None = None
    risk_version: int | None = None
    risk_receipt_id: str | None = None
    risk_state: str | None = None
    schema_version: str = PHASE82_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip() or len(self.cycle_id) != 64:
            raise ValueError("renewal runtime/cycle identity is invalid")
        if self.dispatch_id and len(self.dispatch_id) != 64:
            raise ValueError("renewal dispatch_id must be a content hash")
        if self.runtime_version < 0 or self.fencing_token <= 0:
            raise ValueError("renewal runtime version/fence is invalid")
        if self.claim_fencing_token <= 0:
            raise ValueError("renewal claim fence must be positive")
        if self.renewed and self.status != "RENEWED":
            raise ValueError("renewed receipt must use RENEWED status")
        if self.status == "RENEWED" and not self.renewed:
            raise ValueError("RENEWED status must set renewed=true")
        if self.risk_state == "HALTED" and self.renewed:
            raise ValueError("HALTED recovery claim cannot be renewed")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase82 renewal must remain shadow-only")


@dataclass(frozen=True, slots=True)
class RecoveryClaimExecutionStep:
    recovery_step: RecoveryObligationExecutionStep
    claim: RecoveryClaimReceipt | None
    outcome: str
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE82_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RecoveryClaimError(f"{label} returned non-object payload")
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
        raise RecoveryClaimError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RecoveryClaimError(f"{label} must be integer") from exc


def _optional_hash(value: object, label: str) -> str | None:
    if value is None:
        return None
    result = str(value)
    if len(result) != 64:
        raise RecoveryClaimError(f"{label} must be a content hash")
    return result


def _leg(value: object) -> CancelRecoveryLeg:
    row = _mapping(value, "recovery claim leg")
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


class RecoveryClaimStore:
    """Claim/renew a Phase81 recovery obligation under an independent fence."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def claim(
        self,
        lease,
        *,
        cycle_id: str,
        worker_token: str,
        claim_seconds: int,
    ) -> RecoveryClaimReceipt:
        if not lease.acquired:
            raise RecoveryClaimError("runtime lease is not acquired")
        if len(cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")
        if not worker_token.strip():
            raise ValueError("worker_token is required")
        if claim_seconds <= 0:
            raise ValueError("claim_seconds must be positive")

        row = _mapping(
            self._rpc(
                "brian_claim_shadow_cancel_recovery",
                {
                    "p_runtime_id": lease.runtime_id,
                    "p_owner_token": lease.owner_token,
                    "p_fencing_token": lease.fencing_token,
                    "p_cycle_id": cycle_id,
                    "p_worker_token": worker_token,
                    "p_claim_seconds": int(claim_seconds),
                },
            ),
            "brian_claim_shadow_cancel_recovery",
        )

        if str(row.get("runtime_id", "")) != lease.runtime_id:
            raise RecoveryClaimError("recovery claim runtime_id mismatch")
        if str(row.get("cycle_id", "")) != cycle_id:
            raise RecoveryClaimError("recovery claim cycle_id mismatch")

        claimed_raw = row.get("claimed")
        terminal_raw = row.get("terminal")
        if not isinstance(claimed_raw, bool) or not isinstance(terminal_raw, bool):
            raise RecoveryClaimError("claimed/terminal must be boolean")

        raw_legs = row.get("recovery_legs", ())
        if raw_legs is None:
            raw_legs = ()
        if not isinstance(raw_legs, (list, tuple)):
            raise RecoveryClaimError("recovery_legs must be an array")

        receipt = RecoveryClaimReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=cycle_id,
            dispatch_id=str(row.get("dispatch_id", "")),
            cancel_risk_receipt_id=str(
                row.get("cancel_risk_receipt_id", "")
            ),
            runtime_version=_integer(
                row.get("runtime_version"),
                "runtime_version",
                default=lease.version,
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
            claim_fencing_token=(
                None
                if row.get("claim_fencing_token") is None
                else _integer(
                    row.get("claim_fencing_token"),
                    "claim_fencing_token",
                )
            ),
            status=str(row.get("status", "")),
            claimed=claimed_raw,
            terminal=terminal_raw,
            worker_token=(
                None if row.get("worker_token") is None else str(row.get("worker_token"))
            ),
            claim_until=row.get("claim_until"),
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
            recovery_status=(
                None
                if row.get("recovery_status") is None
                else str(row.get("recovery_status"))
            ),
            recovery_legs=tuple(_leg(value) for value in raw_legs),
            completion_ref=(
                None if row.get("completion_ref") is None else str(row.get("completion_ref"))
            ),
        )

        if receipt.fencing_token != lease.fencing_token:
            raise RecoveryClaimError(
                "database recovery claim runtime fence does not match lease"
            )
        if receipt.claimed and receipt.worker_token != worker_token:
            raise RecoveryClaimError(
                "database recovery claim worker does not match requester"
            )
        return receipt

    def renew(
        self,
        lease,
        claim: RecoveryClaimReceipt,
        *,
        worker_token: str,
        claim_seconds: int,
    ) -> RecoveryClaimRenewal:
        if not claim.claimed or claim.claim_fencing_token is None:
            raise RecoveryClaimError("active recovery claim is required")
        if claim.runtime_id != lease.runtime_id:
            raise RecoveryClaimError("recovery claim runtime does not match lease")
        if claim.fencing_token != lease.fencing_token:
            raise RecoveryClaimError("recovery claim runtime fence does not match lease")
        if claim.worker_token != worker_token:
            raise RecoveryClaimError("recovery claim worker does not match requester")
        if claim_seconds <= 0:
            raise ValueError("claim_seconds must be positive")

        row = _mapping(
            self._rpc(
                "brian_renew_shadow_cancel_recovery_claim",
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
            "brian_renew_shadow_cancel_recovery_claim",
        )
        if str(row.get("runtime_id", "")) != lease.runtime_id:
            raise RecoveryClaimError("recovery renewal runtime_id mismatch")
        if str(row.get("cycle_id", "")) != claim.cycle_id:
            raise RecoveryClaimError("recovery renewal cycle_id mismatch")

        renewed_raw = row.get("renewed")
        if not isinstance(renewed_raw, bool):
            raise RecoveryClaimError("renewed must be boolean")

        receipt = RecoveryClaimRenewal(
            runtime_id=lease.runtime_id,
            cycle_id=claim.cycle_id,
            dispatch_id=str(row.get("dispatch_id", claim.dispatch_id)),
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
            claim_fencing_token=_integer(
                row.get("claim_fencing_token"),
                "claim_fencing_token",
                default=claim.claim_fencing_token,
            ),
            status=str(row.get("status", "")),
            renewed=renewed_raw,
            claim_until=row.get("claim_until"),
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
        )
        if receipt.fencing_token != lease.fencing_token:
            raise RecoveryClaimError("recovery renewal runtime fence drift")
        if receipt.claim_fencing_token != claim.claim_fencing_token:
            raise RecoveryClaimError("recovery renewal claim fence drift")
        return receipt


class PersistedRecoveryClaimSupervisor:
    """Phase81 obligation followed by single-owner Phase82 recovery claim."""

    def __init__(
        self,
        *,
        recovery_supervisor: PersistedRecoveryObligationSupervisor,
        claims: RecoveryClaimStore,
    ) -> None:
        self.recovery_supervisor = recovery_supervisor
        self.claims = claims

    def _runtime_supervisor(self):
        return (
            self.recovery_supervisor
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
    ) -> RecoveryClaimExecutionStep:
        recovery_step = self.recovery_supervisor.process_governed_cycle(
            governed,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )
        supervisor = self._runtime_supervisor()
        recovery = recovery_step.recovery

        if (
            recovery is None
            or not recovery.prepared
            or recovery.recovery_status in {
                "NO_RECOVERY_REQUIRED",
                "MANUAL_REVIEW",
            }
        ):
            return RecoveryClaimExecutionStep(
                recovery_step=recovery_step,
                claim=None,
                outcome=recovery_step.outcome,
                persisted_version=recovery_step.persisted_version,
                checkpoint_id=recovery_step.checkpoint_id,
            )

        claim = self.claims.claim(
            supervisor.lease,
            cycle_id=recovery.cycle_id,
            worker_token=recovery_worker_token,
            claim_seconds=recovery_claim_seconds,
        )

        if claim.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost while claiming cancel recovery"
            )
        if claim.status in {
            "HEAD_MOVED",
            "DIRECTIVE_MISSING",
            "RISK_STATE_UNAVAILABLE",
            "EVIDENCE_INVALID",
        }:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"recovery claim failed with {claim.status}"
            )

        outcome = recovery_step.outcome
        if claim.status == "WAIT_RISK_RELEASE":
            outcome = f"{outcome}_CLAIM_WAIT_RISK_RELEASE"
        elif claim.status == "BLOCKED_ACTIVE":
            outcome = f"{outcome}_CLAIM_BLOCKED_ACTIVE"
        elif claim.status in {"CLAIMED", "ALREADY_OWNED", "EXPIRED_RECOVERY"}:
            outcome = f"{outcome}_RECOVERY_CLAIMED"
        elif claim.status == "COMPLETED":
            outcome = f"{outcome}_RECOVERY_ALREADY_COMPLETED"

        return RecoveryClaimExecutionStep(
            recovery_step=recovery_step,
            claim=claim,
            outcome=outcome,
            persisted_version=supervisor.persisted_version,
            checkpoint_id=supervisor.runtime.checkpoint().checkpoint_id,
        )
