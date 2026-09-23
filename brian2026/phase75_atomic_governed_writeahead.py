from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

from .phase69_governed_shadow_execution import GovernedShadowExecution
from .phase70_durable_runtime_store import RuntimeLease
from .phase71_persisted_runtime_supervisor import (
    PersistedDurableRuntimeSupervisor,
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from .phase73_operational_risk_store import (
    OperationalRiskStore,
    StoredOperationalRiskLedger,
)
from .phase74_governed_cycle_binding import GovernedCycleBindingError

PHASE75_SCHEMA_VERSION = "brian.phase75-atomic-governed-writeahead.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class AtomicGovernedWriteAheadError(RuntimeError):
    pass


class AtomicGovernedWriteAheadStaleError(AtomicGovernedWriteAheadError):
    pass


@dataclass(frozen=True, slots=True)
class AtomicGovernedWriteAheadReceipt:
    runtime_id: str
    cycle_id: str
    checkpoint_id: str
    runtime_version_before: int
    runtime_version_after: int
    current_runtime_version: int
    risk_version: int
    risk_ledger_hash: str
    risk_receipt_id: str
    governed_result_id: str
    policy_fingerprint: str
    fencing_token: int
    status: str
    authorized: bool
    duplicate: bool
    schema_version: str = PHASE75_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if self.runtime_version_before <= 0:
            raise ValueError("runtime_version_before must be positive")
        if self.runtime_version_after <= self.runtime_version_before:
            raise ValueError("runtime_version_after must advance")
        if self.current_runtime_version < self.runtime_version_after:
            raise ValueError("current_runtime_version cannot precede authorization")
        if self.risk_version <= 0 or self.fencing_token <= 0:
            raise ValueError("risk_version/fencing_token must be positive")
        for label, value in (
            ("cycle_id", self.cycle_id),
            ("checkpoint_id", self.checkpoint_id),
            ("risk_ledger_hash", self.risk_ledger_hash),
            ("risk_receipt_id", self.risk_receipt_id),
            ("governed_result_id", self.governed_result_id),
            ("policy_fingerprint", self.policy_fingerprint),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be a content hash")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase75 receipt must remain shadow-only")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AtomicGovernedWriteAheadError(f"{label} returned non-object payload")
    return {str(key): item for key, item in value.items()}


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise AtomicGovernedWriteAheadError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AtomicGovernedWriteAheadError(f"{label} must be integer") from exc


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise AtomicGovernedWriteAheadError(f"{label} must be boolean")
    return value


def _risk_head_receipt_id(risk: StoredOperationalRiskLedger) -> str:
    entries = risk.ledger.entries
    if not entries:
        raise AtomicGovernedWriteAheadError(
            "persisted operational-risk ledger has no head receipt"
        )
    return entries[-1].receipt.receipt_id


class AtomicGovernedWriteAheadStore:
    """Client for Phase75's atomic risk authorization + Phase70 checkpoint commit."""

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def authorize_and_persist(
        self,
        lease: RuntimeLease,
        *,
        expected_runtime_version: int,
        risk: StoredOperationalRiskLedger,
        governed: GovernedShadowExecution,
        checkpoint,
    ) -> AtomicGovernedWriteAheadReceipt:
        if not lease.acquired:
            raise AtomicGovernedWriteAheadError("runtime lease is not acquired")
        if expected_runtime_version <= 0:
            raise ValueError("expected_runtime_version must be positive")
        if risk.runtime_id != lease.runtime_id:
            raise AtomicGovernedWriteAheadError(
                "persisted risk ledger runtime does not match lease"
            )
        if not governed.shadow_only or governed.live_execution:
            raise AtomicGovernedWriteAheadError("governed execution crossed live boundary")

        receipt_id = _risk_head_receipt_id(risk)
        if receipt_id != governed.operational_risk_receipt_id:
            raise AtomicGovernedWriteAheadError(
                "governed cycle is not based on persisted risk head receipt"
            )

        checkpoint_dict = checkpoint.to_dict()
        cycle_id = governed.cycle.cycle_id
        cycles = checkpoint_dict.get("journal_manifest", {}).get("cycles", {})
        entries = checkpoint_dict.get("journal_manifest", {}).get("entries", [])
        if not isinstance(cycles, Mapping) or cycle_id not in cycles:
            raise AtomicGovernedWriteAheadError(
                "write-ahead checkpoint does not contain governed cycle"
            )
        if dict(cycles[cycle_id]) != governed.cycle.to_dict():
            raise AtomicGovernedWriteAheadError(
                "write-ahead cycle body differs from governed cycle"
            )
        cycle_entries = [
            row for row in entries
            if isinstance(row, Mapping) and row.get("cycle_id") == cycle_id
        ]
        if not cycle_entries or cycle_entries[-1].get("stage") != "CYCLE_CREATED":
            raise AtomicGovernedWriteAheadError(
                "write-ahead checkpoint must stop at CYCLE_CREATED"
            )

        result = self._rpc(
            "brian_authorize_and_persist_governed_cycle",
            {
                "p_runtime_id": lease.runtime_id,
                "p_owner_token": lease.owner_token,
                "p_fencing_token": lease.fencing_token,
                "p_expected_runtime_version": int(expected_runtime_version),
                "p_risk_version": int(risk.version),
                "p_risk_ledger_hash": risk.ledger_hash,
                "p_risk_receipt_id": receipt_id,
                "p_cycle_id": cycle_id,
                "p_governed_result_id": governed.result_id,
                "p_policy_fingerprint": governed.policy_fingerprint,
                "p_checkpoint": checkpoint_dict,
            },
        )
        row = _mapping(result, "brian_authorize_and_persist_governed_cycle")

        status = str(row.get("status", ""))
        authorized = _boolean(row.get("authorized"), "authorized")
        duplicate = bool(row.get("duplicate", False))

        if status in {"AUTHORIZED_AND_PERSISTED", "DUPLICATE_CURRENT", "DUPLICATE_HISTORICAL"}:
            if not authorized:
                raise AtomicGovernedWriteAheadError(
                    f"{status} returned authorized=false"
                )
        if status == "AUTHORIZED_AND_PERSISTED" and duplicate:
            raise AtomicGovernedWriteAheadError(
                "AUTHORIZED_AND_PERSISTED cannot be duplicate"
            )
        if status.startswith("DUPLICATE_") and not duplicate:
            raise AtomicGovernedWriteAheadError(
                f"{status} must be marked duplicate"
            )
        if status in {
            "LEASE_LOST",
            "RUNTIME_VERSION_CONFLICT",
            "RISK_VERSION_CONFLICT",
        } and authorized:
            raise AtomicGovernedWriteAheadError(
                f"{status} cannot be authorized"
            )

        if not authorized:
            raise AtomicGovernedWriteAheadStaleError(
                f"governed write-ahead rejected with {status}"
            )

        expected_anchors = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "risk_ledger_hash": risk.ledger_hash,
            "risk_receipt_id": receipt_id,
            "governed_result_id": governed.result_id,
            "policy_fingerprint": governed.policy_fingerprint,
            "cycle_id": cycle_id,
        }
        for key, expected in expected_anchors.items():
            if str(row.get(key, "")) != expected:
                raise AtomicGovernedWriteAheadError(
                    f"database authorization {key} does not match submitted evidence"
                )

        return AtomicGovernedWriteAheadReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=cycle_id,
            checkpoint_id=checkpoint.checkpoint_id,
            runtime_version_before=_integer(
                row.get("runtime_version_before"),
                "runtime_version_before",
            ),
            runtime_version_after=_integer(
                row.get("runtime_version_after"),
                "runtime_version_after",
            ),
            current_runtime_version=_integer(
                row.get("current_runtime_version"),
                "current_runtime_version",
            ),
            risk_version=_integer(row.get("risk_version"), "risk_version"),
            risk_ledger_hash=risk.ledger_hash,
            risk_receipt_id=receipt_id,
            governed_result_id=governed.result_id,
            policy_fingerprint=governed.policy_fingerprint,
            fencing_token=_integer(
                row.get("fencing_token", lease.fencing_token),
                "fencing_token",
            ),
            status=status,
            authorized=True,
            duplicate=duplicate,
        )


@dataclass(frozen=True, slots=True)
class PersistedGovernedCycleStep:
    authorization: AtomicGovernedWriteAheadReceipt
    advanced_status: str
    persisted_version: int
    checkpoint_id: str
    schema_version: str = PHASE75_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


class PersistedGovernedRuntimeSupervisor:
    """Phase71 supervisor hardened by Phase73 risk truth + Phase75 atomic write-ahead."""

    def __init__(
        self,
        *,
        runtime_supervisor: PersistedDurableRuntimeSupervisor,
        risk_store: OperationalRiskStore,
        atomic_store: AtomicGovernedWriteAheadStore,
    ) -> None:
        self.runtime_supervisor = runtime_supervisor
        self.risk_store = risk_store
        self.atomic_store = atomic_store

    def authorize_write_ahead(
        self,
        governed: GovernedShadowExecution,
    ) -> AtomicGovernedWriteAheadReceipt:
        supervisor = self.runtime_supervisor
        if not supervisor.valid:
            raise PersistedRuntimeStaleError(
                "runtime supervisor is stale before governed cycle"
            )

        risk = self.risk_store.load(runtime_id=supervisor.runtime_id)
        if risk is None:
            raise AtomicGovernedWriteAheadError(
                "no persisted operational-risk ledger exists for runtime"
            )

        supervisor.runtime.journal_cycle(governed.cycle)
        checkpoint = supervisor.runtime.checkpoint()

        try:
            authorization = self.atomic_store.authorize_and_persist(
                supervisor.lease,
                expected_runtime_version=supervisor.persisted_version,
                risk=risk,
                governed=governed,
                checkpoint=checkpoint,
            )
        except Exception:
            supervisor._valid = False
            raise

        if authorization.status == "DUPLICATE_HISTORICAL":
            supervisor._valid = False
            raise AtomicGovernedWriteAheadStaleError(
                "write-ahead authorization is historical; reload runtime head"
            )
        if authorization.runtime_version_before != supervisor.persisted_version:
            supervisor._valid = False
            raise PersistedRuntimeStaleError(
                "authorization runtime_version_before disagrees with supervisor"
            )
        if authorization.fencing_token != supervisor.lease.fencing_token:
            supervisor._valid = False
            raise PersistedRuntimeLeaseError(
                "authorization fencing token differs from runtime lease"
            )

        supervisor.accept_external_checkpoint_commit(
            checkpoint_id=authorization.checkpoint_id,
            version=authorization.runtime_version_after,
        )
        return authorization

    def advance_authorized(
        self,
        *,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ):
        return self.runtime_supervisor.advance_pending(
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )

    def abort_authorized_cycle(
        self,
        *,
        cycle_id: str,
        reason: str,
    ):
        supervisor = self.runtime_supervisor
        if not supervisor.valid:
            raise PersistedRuntimeStaleError(
                "runtime supervisor is stale before governed abort"
            )
        if supervisor.runtime.journal.latest_stage(cycle_id) != "CYCLE_CREATED":
            raise AtomicGovernedWriteAheadError(
                "only a pre-paper CYCLE_CREATED authorization may be aborted"
            )
        supervisor.runtime.journal.mark_aborted(
            cycle_id,
            reason=reason,
        )
        commit = supervisor.persist_current_checkpoint()
        return commit

    def process_governed_cycle(
        self,
        governed: GovernedShadowExecution,
        *,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> PersistedGovernedCycleStep:
        authorization = self.authorize_write_ahead(governed)
        advanced = self.advance_authorized(
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )
        return PersistedGovernedCycleStep(
            authorization=authorization,
            advanced_status=(
                "NONE"
                if advanced.durable_receipt is None
                else advanced.durable_receipt.status
            ),
            persisted_version=advanced.persisted_version,
            checkpoint_id=advanced.checkpoint_id,
        )
