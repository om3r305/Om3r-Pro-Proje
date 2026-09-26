from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .evidence_ledger import content_hash
from .phase70_durable_runtime_store import RuntimeLease
from .phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from .phase75_atomic_governed_writeahead import (
    AtomicGovernedWriteAheadReceipt,
    PersistedGovernedRuntimeSupervisor,
)

PHASE76_SCHEMA_VERSION = "brian.phase76-shadow-execution-outbox.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class ShadowExecutionOutboxError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ShadowExecutionDispatchReceipt:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    governed_result_id: str
    policy_fingerprint: str
    authorization_checkpoint_id: str
    authorization_runtime_version: int
    current_runtime_version: int
    authorization_risk_version: int
    current_risk_version: int
    risk_ledger_hash: str
    risk_receipt_id: str
    fencing_token: int
    status: str
    submitted: bool
    duplicate: bool
    schema_version: str = PHASE76_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        for label, value in (
            ("cycle_id", self.cycle_id),
            ("dispatch_id", self.dispatch_id),
            ("governed_result_id", self.governed_result_id),
            ("policy_fingerprint", self.policy_fingerprint),
            ("authorization_checkpoint_id", self.authorization_checkpoint_id),
            ("risk_ledger_hash", self.risk_ledger_hash),
            ("risk_receipt_id", self.risk_receipt_id),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be a content hash")
        if self.authorization_runtime_version <= 0:
            raise ValueError("authorization_runtime_version must be positive")
        if self.current_runtime_version < 0:
            raise ValueError("current_runtime_version must be non-negative")
        if self.authorization_risk_version <= 0:
            raise ValueError("authorization_risk_version must be positive")
        if self.current_risk_version < 0:
            raise ValueError("current_risk_version must be non-negative")
        if self.fencing_token <= 0:
            raise ValueError("fencing_token must be positive")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase76 dispatch must remain shadow-only")


@dataclass(frozen=True, slots=True)
class StoredShadowExecutionDispatch:
    runtime_id: str
    cycle_id: str
    dispatch_id: str
    governed_result_id: str
    policy_fingerprint: str
    authorization_checkpoint_id: str
    authorization_runtime_version: int
    risk_version: int
    risk_ledger_hash: str
    risk_receipt_id: str
    fencing_token: int
    submitted_at: object | None
    schema_version: str = PHASE76_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        for label, value in (
            ("cycle_id", self.cycle_id),
            ("dispatch_id", self.dispatch_id),
            ("governed_result_id", self.governed_result_id),
            ("policy_fingerprint", self.policy_fingerprint),
            ("authorization_checkpoint_id", self.authorization_checkpoint_id),
            ("risk_ledger_hash", self.risk_ledger_hash),
            ("risk_receipt_id", self.risk_receipt_id),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be a content hash")
        if self.authorization_runtime_version <= 0 or self.risk_version <= 0:
            raise ValueError("stored dispatch versions must be positive")
        if self.fencing_token <= 0:
            raise ValueError("fencing_token must be positive")
        if not self.shadow_only or self.live_execution:
            raise ValueError("stored dispatch crossed live boundary")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ShadowExecutionOutboxError(f"{label} returned non-object payload")
    return {str(key): item for key, item in value.items()}


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ShadowExecutionOutboxError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ShadowExecutionOutboxError(f"{label} must be integer") from exc


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ShadowExecutionOutboxError(f"{label} must be boolean")
    return value


def dispatch_id_for_authorization(
    authorization: AtomicGovernedWriteAheadReceipt,
) -> str:
    return content_hash({
        "schema_version": PHASE76_SCHEMA_VERSION,
        "runtime_id": authorization.runtime_id,
        "cycle_id": authorization.cycle_id,
        "governed_result_id": authorization.governed_result_id,
        "policy_fingerprint": authorization.policy_fingerprint,
        "authorization_checkpoint_id": authorization.checkpoint_id,
        "authorization_runtime_version": authorization.runtime_version_after,
        "risk_version": authorization.risk_version,
        "risk_ledger_hash": authorization.risk_ledger_hash,
        "risk_receipt_id": authorization.risk_receipt_id,
    })


class ShadowExecutionOutboxStore:
    """Submit an already-authorized cycle to the durable paper execution outbox."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def submit(
        self,
        lease: RuntimeLease,
        authorization: AtomicGovernedWriteAheadReceipt,
    ) -> ShadowExecutionDispatchReceipt:
        if not lease.acquired:
            raise ShadowExecutionOutboxError("runtime lease is not acquired")
        if lease.runtime_id != authorization.runtime_id:
            raise ShadowExecutionOutboxError(
                "authorization runtime does not match lease"
            )
        if lease.fencing_token != authorization.fencing_token:
            raise ShadowExecutionOutboxError(
                "authorization fencing token does not match lease"
            )

        dispatch_id = dispatch_id_for_authorization(authorization)
        result = self._rpc(
            "brian_submit_shadow_execution_dispatch",
            {
                "p_runtime_id": lease.runtime_id,
                "p_owner_token": lease.owner_token,
                "p_fencing_token": lease.fencing_token,
                "p_cycle_id": authorization.cycle_id,
                "p_dispatch_id": dispatch_id,
            },
        )
        row = _mapping(result, "brian_submit_shadow_execution_dispatch")
        status = str(row.get("status", ""))
        submitted = _boolean(row.get("submitted"), "submitted")
        duplicate = bool(row.get("duplicate", False))

        if status in {"SUBMITTED", "DUPLICATE_CURRENT", "DUPLICATE_HISTORICAL"}:
            if not submitted:
                raise ShadowExecutionOutboxError(
                    f"{status} returned submitted=false"
                )
        if status == "SUBMITTED" and duplicate:
            raise ShadowExecutionOutboxError("SUBMITTED cannot be duplicate")
        if status.startswith("DUPLICATE_") and not duplicate:
            raise ShadowExecutionOutboxError(
                f"{status} must be marked duplicate"
            )
        if status in {
            "AUTHORIZATION_MISSING",
            "LEASE_LOST",
            "RUNTIME_VERSION_CONFLICT",
            "RISK_VERSION_CONFLICT",
            "RECOVERY_BARRIER",
        } and submitted:
            raise ShadowExecutionOutboxError(
                f"{status} cannot return submitted=true"
            )

        if submitted:
            expected = {
                "cycle_id": authorization.cycle_id,
                "dispatch_id": dispatch_id,
                "governed_result_id": authorization.governed_result_id,
                "policy_fingerprint": authorization.policy_fingerprint,
                "authorization_checkpoint_id": authorization.checkpoint_id,
                "risk_ledger_hash": authorization.risk_ledger_hash,
                "risk_receipt_id": authorization.risk_receipt_id,
            }
            for key, value in expected.items():
                if str(row.get(key, "")) != value:
                    raise ShadowExecutionOutboxError(
                        f"database dispatch {key} does not match authorization"
                    )

        return ShadowExecutionDispatchReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=authorization.cycle_id,
            dispatch_id=dispatch_id,
            governed_result_id=authorization.governed_result_id,
            policy_fingerprint=authorization.policy_fingerprint,
            authorization_checkpoint_id=authorization.checkpoint_id,
            authorization_runtime_version=authorization.runtime_version_after,
            current_runtime_version=_integer(
                row.get(
                    "current_runtime_version",
                    row.get("runtime_version", authorization.runtime_version_after),
                ),
                "current_runtime_version",
            ),
            authorization_risk_version=authorization.risk_version,
            current_risk_version=_integer(
                row.get("risk_version", authorization.risk_version),
                "risk_version",
            ),
            risk_ledger_hash=authorization.risk_ledger_hash,
            risk_receipt_id=authorization.risk_receipt_id,
            fencing_token=_integer(
                row.get("fencing_token", lease.fencing_token),
                "fencing_token",
            ),
            status=status,
            submitted=submitted,
            duplicate=duplicate,
        )

    def load(
        self,
        *,
        runtime_id: str,
        cycle_id: str,
    ) -> StoredShadowExecutionDispatch | None:
        if not runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")
        result = self._rpc(
            "brian_read_shadow_execution_dispatch",
            {
                "p_runtime_id": runtime_id,
                "p_cycle_id": cycle_id,
            },
        )
        if result is None:
            return None
        row = _mapping(result, "brian_read_shadow_execution_dispatch")
        if str(row.get("runtime_id", "")) != runtime_id:
            raise ShadowExecutionOutboxError("stored dispatch runtime_id mismatch")
        if str(row.get("cycle_id", "")) != cycle_id:
            raise ShadowExecutionOutboxError("stored dispatch cycle_id mismatch")
        if row.get("shadow_only") is not True or row.get("live_execution") is not False:
            raise ShadowExecutionOutboxError("stored dispatch crossed live boundary")
        return StoredShadowExecutionDispatch(
            runtime_id=runtime_id,
            cycle_id=cycle_id,
            dispatch_id=str(row.get("dispatch_id", "")),
            governed_result_id=str(row.get("governed_result_id", "")),
            policy_fingerprint=str(row.get("policy_fingerprint", "")),
            authorization_checkpoint_id=str(
                row.get("authorization_checkpoint_id", "")
            ),
            authorization_runtime_version=_integer(
                row.get("authorization_runtime_version"),
                "authorization_runtime_version",
            ),
            risk_version=_integer(row.get("risk_version"), "risk_version"),
            risk_ledger_hash=str(row.get("risk_ledger_hash", "")),
            risk_receipt_id=str(row.get("risk_receipt_id", "")),
            fencing_token=_integer(row.get("fencing_token"), "fencing_token"),
            submitted_at=row.get("submitted_at"),
            shadow_only=True,
            live_execution=False,
        )


@dataclass(frozen=True, slots=True)
class PersistedDispatchedCycleStep:
    authorization: AtomicGovernedWriteAheadReceipt
    dispatch: ShadowExecutionDispatchReceipt
    outcome: str
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE76_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


class PersistedDispatchedRuntimeSupervisor:
    """Require a durable SUBMITTED outbox record before paper execution advances."""

    def __init__(
        self,
        *,
        governed_supervisor: PersistedGovernedRuntimeSupervisor,
        outbox: ShadowExecutionOutboxStore,
    ) -> None:
        self.governed_supervisor = governed_supervisor
        self.outbox = outbox

    def authorize_and_submit(self, governed):
        phase75 = self.governed_supervisor
        supervisor = phase75.runtime_supervisor

        authorization = phase75.authorize_write_ahead(governed)
        dispatch = self.outbox.submit(supervisor.lease, authorization)

        if not dispatch.submitted:
            if dispatch.status == "RISK_VERSION_CONFLICT":
                phase75.abort_authorized_cycle(
                    cycle_id=authorization.cycle_id,
                    reason="phase76:risk_changed_before_dispatch",
                )
                return authorization, dispatch, "ABORTED_RISK_STALE"

            if dispatch.status == "RECOVERY_BARRIER":
                phase75.abort_authorized_cycle(
                    cycle_id=authorization.cycle_id,
                    reason="phase86:recovery_admission_interlock",
                )
                return authorization, dispatch, "ABORTED_RECOVERY_BARRIER"

            supervisor._valid = False
            if dispatch.status == "LEASE_LOST":
                raise PersistedRuntimeLeaseError(
                    "runtime lease was lost before execution dispatch"
                )
            raise PersistedRuntimeStaleError(
                f"execution dispatch rejected with {dispatch.status}; reload required"
            )

        if dispatch.status == "DUPLICATE_HISTORICAL":
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "execution dispatch is historical relative to runtime head"
            )
        if dispatch.current_runtime_version != supervisor.persisted_version:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "dispatch runtime version disagrees with supervisor"
            )
        if dispatch.fencing_token != supervisor.lease.fencing_token:
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "dispatch fencing token disagrees with supervisor lease"
            )
        return authorization, dispatch, None

    def advance_submitted(
        self,
        *,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ):
        return self.governed_supervisor.advance_authorized(
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )

    def process_governed_cycle(
        self,
        governed,
        *,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> PersistedDispatchedCycleStep:
        authorization, dispatch, terminal = self.authorize_and_submit(governed)
        supervisor = self.governed_supervisor.runtime_supervisor

        if terminal is not None:
            checkpoint = supervisor.runtime.checkpoint()
            return PersistedDispatchedCycleStep(
                authorization=authorization,
                dispatch=dispatch,
                outcome=terminal,
                persisted_version=supervisor.persisted_version,
                checkpoint_id=checkpoint.checkpoint_id,
            )

        advanced = self.advance_submitted(
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )
        checkpoint = supervisor.runtime.checkpoint()
        return PersistedDispatchedCycleStep(
            authorization=authorization,
            dispatch=dispatch,
            outcome=(
                "NONE"
                if advanced.durable_receipt is None
                else advanced.durable_receipt.status
            ),
            persisted_version=supervisor.persisted_version,
            checkpoint_id=checkpoint.checkpoint_id,
        )
