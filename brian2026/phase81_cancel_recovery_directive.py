from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

from .phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from .phase80_claim_fenced_checkpoint import (
    ClaimFencedExecutionStep,
    PersistedClaimFencedRuntimeSupervisor,
)

PHASE81_SCHEMA_VERSION = "brian.phase81-cancel-recovery-directive.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class CancelRecoveryDirectiveError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CancelRecoveryLeg:
    asset_id: str
    before_weight: float
    current_weight: float
    target_weight: float
    reduce_weight: float
    current_direction: int
    order_direction: int
    reduce_only: bool = True

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("recovery leg asset_id is required")
        if not all(
            math.isfinite(value)
            for value in (
                self.before_weight,
                self.current_weight,
                self.target_weight,
                self.reduce_weight,
            )
        ):
            raise ValueError("recovery leg weights must be finite")
        if self.reduce_weight <= 0:
            raise ValueError("recovery leg reduce_weight must be positive")
        if self.current_direction not in (-1, 1):
            raise ValueError("recovery leg current_direction must be directional")
        if self.order_direction != -self.current_direction:
            raise ValueError("recovery order must oppose current exposure")
        if not self.reduce_only:
            raise ValueError("Phase81 recovery legs must be reduce-only")
        if not math.isclose(
            self.target_weight,
            self.before_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("recovery target must equal authoritative pre-cycle weight")
        if abs(self.target_weight) > abs(self.current_weight) + 1e-12:
            raise ValueError("recovery target cannot increase exposure")
        if abs(self.target_weight) > 1e-12:
            target_direction = 1 if self.target_weight > 0 else -1
            if target_direction != self.current_direction:
                raise ValueError("reduce-only recovery cannot flip position direction")
        expected_reduce = abs(self.current_weight) - abs(self.target_weight)
        if not math.isclose(
            self.reduce_weight,
            expected_reduce,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("recovery reduce_weight does not match target path")

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "before_weight": self.before_weight,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "reduce_weight": self.reduce_weight,
            "current_direction": self.current_direction,
            "order_direction": self.order_direction,
            "reduce_only": self.reduce_only,
        }


@dataclass(frozen=True, slots=True)
class CancelRecoveryDirectiveReceipt:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    runtime_version: int
    fencing_token: int
    status: str
    prepared: bool
    duplicate: bool
    cancel_risk_version: int | None = None
    cancel_risk_receipt_id: str | None = None
    cancel_reason: str | None = None
    pre_state_id: str | None = None
    current_state_id: str | None = None
    current_risk_version: int | None = None
    current_risk_receipt_id: str | None = None
    current_risk_state: str | None = None
    recovery_status: str | None = None
    recovery_legs: tuple[CancelRecoveryLeg, ...] = ()
    unsafe_assets: tuple[Mapping[str, object], ...] = ()
    journal_stage: str | None = None
    reason: str | None = None
    foreign_cycle_id: str | None = None
    foreign_cycle_stage: str | None = None
    schema_version: str = PHASE81_SCHEMA_VERSION
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
            raise ValueError("runtime_version must be non-negative")
        if self.fencing_token <= 0:
            raise ValueError("fencing_token must be positive")
        if self.prepared and self.status not in {"PREPARED", "DUPLICATE"}:
            raise ValueError("prepared recovery receipt has unsupported status")
        if self.duplicate and self.status != "DUPLICATE":
            raise ValueError("duplicate recovery receipt must use DUPLICATE status")
        if self.status == "DUPLICATE" and not (self.prepared and self.duplicate):
            raise ValueError("DUPLICATE flags are inconsistent")
        if self.status == "PREPARED" and (not self.prepared or self.duplicate):
            raise ValueError("PREPARED flags are inconsistent")

        if self.prepared:
            if self.cancel_risk_version is None or self.cancel_risk_version <= 0:
                raise ValueError("prepared recovery requires cancel risk version")
            if (
                self.cancel_risk_receipt_id is None
                or len(self.cancel_risk_receipt_id) != 64
            ):
                raise ValueError("prepared recovery requires cancel risk receipt")
            if self.cancel_reason not in {
                "HALTED",
                "REDUCING_NEW_RISK",
                "ASSET_COOLDOWN",
            }:
                raise ValueError("prepared recovery has invalid cancel reason")
            for label, value in (
                ("pre_state_id", self.pre_state_id),
                ("current_state_id", self.current_state_id),
                ("current_risk_receipt_id", self.current_risk_receipt_id),
            ):
                if value is None or len(value) != 64:
                    raise ValueError(f"prepared recovery requires {label}")
            if self.current_risk_version is None or self.current_risk_version <= 0:
                raise ValueError("prepared recovery requires current risk version")
            if self.current_risk_state not in {"ACTIVE", "REDUCING", "HALTED"}:
                raise ValueError("prepared recovery has invalid current risk state")
            if self.recovery_status not in {
                "READY_REDUCE_ONLY",
                "WAIT_RISK_RELEASE",
                "NO_RECOVERY_REQUIRED",
                "MANUAL_REVIEW",
            }:
                raise ValueError("prepared recovery has invalid recovery_status")

            assets = [leg.asset_id for leg in self.recovery_legs]
            if len(assets) != len(set(assets)):
                raise ValueError("recovery directive cannot contain duplicate assets")
            if (
                self.recovery_status == "READY_REDUCE_ONLY"
                and (not self.recovery_legs or self.current_risk_state == "HALTED")
            ):
                raise ValueError("READY_REDUCE_ONLY requires reducible legs and non-HALTED risk")
            if (
                self.recovery_status == "WAIT_RISK_RELEASE"
                and (not self.recovery_legs or self.current_risk_state != "HALTED")
            ):
                raise ValueError("WAIT_RISK_RELEASE requires HALTED risk and recovery legs")
            if self.recovery_status == "NO_RECOVERY_REQUIRED" and self.recovery_legs:
                raise ValueError("NO_RECOVERY_REQUIRED cannot contain recovery legs")
            if self.recovery_status == "MANUAL_REVIEW" and not self.unsafe_assets:
                raise ValueError("MANUAL_REVIEW requires unsafe asset evidence")

        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase81 recovery receipt must remain shadow-only")


@dataclass(frozen=True, slots=True)
class RecoveryObligationExecutionStep:
    execution: ClaimFencedExecutionStep
    recovery: CancelRecoveryDirectiveReceipt | None
    outcome: str
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE81_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CancelRecoveryDirectiveError(f"{label} returned non-object payload")
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
        raise CancelRecoveryDirectiveError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CancelRecoveryDirectiveError(f"{label} must be integer") from exc


def _optional_hash(value: object, label: str) -> str | None:
    if value is None:
        return None
    result = str(value)
    if len(result) != 64:
        raise CancelRecoveryDirectiveError(f"{label} must be a content hash")
    return result


def _leg(value: object) -> CancelRecoveryLeg:
    row = _mapping(value, "recovery leg")
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


class CancelRecoveryDirectiveStore:
    """Prepare the immutable post-cancel rollback obligation for a committed cycle."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def prepare(
        self,
        lease,
        *,
        cycle_id: str,
        expected_runtime_version: int,
    ) -> CancelRecoveryDirectiveReceipt:
        if not lease.acquired:
            raise CancelRecoveryDirectiveError("runtime lease is not acquired")
        if len(cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")
        if expected_runtime_version <= 0:
            raise ValueError("expected_runtime_version must be positive")

        row = _mapping(
            self._rpc(
                "brian_prepare_shadow_cancel_recovery",
                {
                    "p_runtime_id": lease.runtime_id,
                    "p_owner_token": lease.owner_token,
                    "p_fencing_token": lease.fencing_token,
                    "p_cycle_id": cycle_id,
                    "p_expected_runtime_version": int(expected_runtime_version),
                },
            ),
            "brian_prepare_shadow_cancel_recovery",
        )

        if str(row.get("runtime_id", "")) != lease.runtime_id:
            raise CancelRecoveryDirectiveError("recovery runtime_id mismatch")
        if str(row.get("cycle_id", "")) != cycle_id:
            raise CancelRecoveryDirectiveError("recovery cycle_id mismatch")

        prepared_raw = row.get("prepared")
        if not isinstance(prepared_raw, bool):
            raise CancelRecoveryDirectiveError("prepared must be boolean")
        prepared = prepared_raw
        duplicate = bool(row.get("duplicate", False))
        status = str(row.get("status", ""))

        if status in {"PREPARED", "DUPLICATE"} and not prepared:
            raise CancelRecoveryDirectiveError(f"{status} requires prepared=true")
        if status in {
            "NO_CANCEL_REQUEST",
            "WAIT_ORIGINAL_COMMIT",
            "FOREIGN_CYCLE_ACTIVE",
            "HEAD_MOVED",
            "LEASE_LOST",
            "RUNTIME_VERSION_CONFLICT",
            "RISK_STATE_UNAVAILABLE",
            "EVIDENCE_INVALID",
        } and prepared:
            raise CancelRecoveryDirectiveError(f"{status} cannot be prepared")

        raw_legs = row.get("recovery_legs", ())
        if raw_legs is None:
            raw_legs = ()
        if not isinstance(raw_legs, (list, tuple)):
            raise CancelRecoveryDirectiveError("recovery_legs must be an array")

        raw_unsafe = row.get("unsafe_assets", ())
        if raw_unsafe is None:
            raw_unsafe = ()
        if not isinstance(raw_unsafe, (list, tuple)):
            raise CancelRecoveryDirectiveError("unsafe_assets must be an array")
        unsafe = tuple(
            _mapping(value, "unsafe asset")
            for value in raw_unsafe
        )

        receipt = CancelRecoveryDirectiveReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=cycle_id,
            dispatch_id=str(row.get("dispatch_id", "")),
            runtime_version=_integer(
                row.get("runtime_version"),
                "runtime_version",
                default=expected_runtime_version,
            ),
            fencing_token=_integer(
                row.get("fencing_token"),
                "fencing_token",
                default=lease.fencing_token,
            ),
            status=status,
            prepared=prepared,
            duplicate=duplicate,
            cancel_risk_version=(
                None
                if row.get("cancel_risk_version") is None
                else _integer(row.get("cancel_risk_version"), "cancel_risk_version")
            ),
            cancel_risk_receipt_id=_optional_hash(
                row.get("cancel_risk_receipt_id"),
                "cancel_risk_receipt_id",
            ),
            cancel_reason=(
                None if row.get("cancel_reason") is None else str(row.get("cancel_reason"))
            ),
            pre_state_id=_optional_hash(row.get("pre_state_id"), "pre_state_id"),
            current_state_id=_optional_hash(
                row.get("current_state_id"),
                "current_state_id",
            ),
            current_risk_version=(
                None
                if row.get("current_risk_version") is None
                else _integer(row.get("current_risk_version"), "current_risk_version")
            ),
            current_risk_receipt_id=_optional_hash(
                row.get("current_risk_receipt_id"),
                "current_risk_receipt_id",
            ),
            current_risk_state=(
                None
                if row.get("current_risk_state") is None
                else str(row.get("current_risk_state"))
            ),
            recovery_status=(
                None
                if row.get("recovery_status") is None
                else str(row.get("recovery_status"))
            ),
            recovery_legs=tuple(_leg(value) for value in raw_legs),
            unsafe_assets=unsafe,
            journal_stage=(
                None if row.get("journal_stage") is None else str(row.get("journal_stage"))
            ),
            reason=None if row.get("reason") is None else str(row.get("reason")),
            foreign_cycle_id=_optional_hash(
                row.get("foreign_cycle_id"),
                "foreign_cycle_id",
            ),
            foreign_cycle_stage=(
                None
                if row.get("foreign_cycle_stage") is None
                else str(row.get("foreign_cycle_stage"))
            ),
        )

        if receipt.fencing_token != lease.fencing_token:
            raise CancelRecoveryDirectiveError(
                "database recovery fence does not match runtime lease"
            )
        if receipt.prepared and receipt.runtime_version != expected_runtime_version:
            raise CancelRecoveryDirectiveError(
                "prepared recovery runtime version does not match authoritative step"
            )
        return receipt


class PersistedRecoveryObligationSupervisor:
    """Phase80 execution followed by a durable Phase81 recovery obligation check."""

    def __init__(
        self,
        *,
        execution_supervisor: PersistedClaimFencedRuntimeSupervisor,
        recovery_store: CancelRecoveryDirectiveStore,
    ) -> None:
        self.execution_supervisor = execution_supervisor
        self.recovery_store = recovery_store

    def _runtime_supervisor(self):
        return (
            self.execution_supervisor
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
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> RecoveryObligationExecutionStep:
        execution = self.execution_supervisor.process_governed_cycle(
            governed,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )
        supervisor = self._runtime_supervisor()

        should_probe = (
            execution.start is not None
            and execution.start.started
            and execution.checkpoint_commit is not None
            and execution.checkpoint_commit.committed
        )
        if not should_probe:
            return RecoveryObligationExecutionStep(
                execution=execution,
                recovery=None,
                outcome=execution.outcome,
                persisted_version=execution.persisted_version,
                checkpoint_id=execution.checkpoint_id,
            )

        recovery = self.recovery_store.prepare(
            supervisor.lease,
            cycle_id=execution.claim.cycle_id,
            expected_runtime_version=supervisor.persisted_version,
        )

        quarantined_foreign_cycle = False
        if recovery.status == "FOREIGN_CYCLE_ACTIVE":
            if (
                recovery.foreign_cycle_id is not None
                and recovery.foreign_cycle_stage == "CYCLE_CREATED"
            ):
                phase75 = (
                    self.execution_supervisor
                    .claimed_supervisor
                    .dispatched_supervisor
                    .governed_supervisor
                )
                phase75.abort_authorized_cycle(
                    cycle_id=recovery.foreign_cycle_id,
                    reason="phase86:recovery_admission_interlock",
                )
                quarantined_foreign_cycle = True
                recovery = self.recovery_store.prepare(
                    supervisor.lease,
                    cycle_id=execution.claim.cycle_id,
                    expected_runtime_version=supervisor.persisted_version,
                )

        if recovery.status == "LEASE_LOST":
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "runtime lease lost while preparing cancel recovery"
            )
        if recovery.status in {
            "RUNTIME_VERSION_CONFLICT",
            "HEAD_MOVED",
            "RISK_STATE_UNAVAILABLE",
            "EVIDENCE_INVALID",
        }:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                f"cancel recovery preparation failed with {recovery.status}"
            )

        outcome = execution.outcome
        if recovery.prepared:
            if recovery.recovery_status == "READY_REDUCE_ONLY":
                outcome = f"{outcome}_RECOVERY_READY"
            elif recovery.recovery_status == "WAIT_RISK_RELEASE":
                outcome = f"{outcome}_RECOVERY_WAIT_RISK_RELEASE"
            elif recovery.recovery_status == "MANUAL_REVIEW":
                outcome = f"{outcome}_RECOVERY_MANUAL_REVIEW"
            elif recovery.recovery_status == "NO_RECOVERY_REQUIRED":
                outcome = f"{outcome}_NO_RECOVERY_REQUIRED"
        elif recovery.status == "WAIT_ORIGINAL_COMMIT":
            outcome = f"{outcome}_RECOVERY_WAIT_ORIGINAL_COMMIT"
        elif recovery.status == "FOREIGN_CYCLE_ACTIVE":
            outcome = f"{outcome}_RECOVERY_WAIT_FOREIGN_CYCLE"

        if quarantined_foreign_cycle:
            outcome = f"{outcome}_FOREIGN_CYCLE_ABORTED"

        return RecoveryObligationExecutionStep(
            execution=execution,
            recovery=recovery,
            outcome=outcome,
            persisted_version=supervisor.persisted_version,
            checkpoint_id=supervisor.runtime.checkpoint().checkpoint_id,
        )
