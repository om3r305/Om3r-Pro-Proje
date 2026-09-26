from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .phase69_governed_shadow_execution import GovernedShadowExecution
from .phase70_durable_runtime_store import RuntimeLease
from .phase73_operational_risk_store import StoredOperationalRiskLedger

PHASE74_SCHEMA_VERSION = "brian.phase74-governed-cycle-binding.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class GovernedCycleBindingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GovernedCycleBinding:
    runtime_id: str
    cycle_id: str
    governed_result_id: str
    policy_fingerprint: str
    risk_version: int
    risk_ledger_hash: str
    risk_receipt_id: str
    runtime_version_before: int
    fencing_token: int
    bound_at: object | None = None
    schema_version: str = PHASE74_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        for label, value in (
            ("runtime_id", self.runtime_id),
            ("cycle_id", self.cycle_id),
            ("governed_result_id", self.governed_result_id),
            ("policy_fingerprint", self.policy_fingerprint),
            ("risk_ledger_hash", self.risk_ledger_hash),
            ("risk_receipt_id", self.risk_receipt_id),
        ):
            if not str(value).strip():
                raise ValueError(f"{label} is required")
        for label, value in (
            ("cycle_id", self.cycle_id),
            ("governed_result_id", self.governed_result_id),
            ("policy_fingerprint", self.policy_fingerprint),
            ("risk_ledger_hash", self.risk_ledger_hash),
            ("risk_receipt_id", self.risk_receipt_id),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be a content hash")
        if self.risk_version <= 0:
            raise ValueError("risk_version must be positive")
        if self.runtime_version_before <= 0:
            raise ValueError("runtime_version_before must be positive")
        if self.fencing_token <= 0:
            raise ValueError("fencing_token must be positive")
        if not self.shadow_only or self.live_execution:
            raise ValueError("governed-cycle binding must remain shadow-only")


@dataclass(frozen=True, slots=True)
class GovernedCycleBindReceipt:
    runtime_id: str
    cycle_id: str
    runtime_version: int
    risk_version: int
    fencing_token: int
    status: str
    bound: bool
    duplicate: bool
    governed_result_id: str
    policy_fingerprint: str
    risk_ledger_hash: str
    risk_receipt_id: str
    schema_version: str = PHASE74_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if self.runtime_version < 0 or self.risk_version < 0:
            raise ValueError("versions must be non-negative")
        if self.fencing_token <= 0:
            raise ValueError("fencing_token must be positive")
        for label, value in (
            ("cycle_id", self.cycle_id),
            ("governed_result_id", self.governed_result_id),
            ("policy_fingerprint", self.policy_fingerprint),
            ("risk_ledger_hash", self.risk_ledger_hash),
            ("risk_receipt_id", self.risk_receipt_id),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be a content hash")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise GovernedCycleBindingError(f"{label} returned non-object payload")
    return {str(key): item for key, item in value.items()}


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise GovernedCycleBindingError(f"{label} must be integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise GovernedCycleBindingError(f"{label} must be integer") from exc


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise GovernedCycleBindingError(f"{label} must be boolean")
    return value


def _risk_head_receipt_id(stored: StoredOperationalRiskLedger) -> str:
    entries = stored.ledger.entries
    if not entries:
        raise GovernedCycleBindingError(
            "operational-risk ledger has no persisted receipt to authorize a cycle"
        )
    receipt_id = entries[-1].receipt.receipt_id
    if len(receipt_id) != 64:
        raise GovernedCycleBindingError("risk head receipt_id is invalid")
    return receipt_id


class GovernedCycleBindingStore:
    """Phase74 client binding a Phase69 cycle to persisted Phase72/73 risk truth.

    The binding must happen before Phase71 write-ahead persistence. This prevents
    a worker from executing a locally generated governed cycle whose risk receipt
    was never committed, or whose runtime/risk versions are already stale.
    """

    def __init__(self, rpc: RpcCall) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        self._rpc = rpc

    def _call(
        self,
        name: str,
        params: Mapping[str, object],
        *,
        allow_none: bool = False,
    ) -> dict[str, object] | None:
        result = self._rpc(name, dict(params))
        if result is None and allow_none:
            return None
        return _mapping(result, name)

    def bind(
        self,
        lease: RuntimeLease,
        *,
        expected_runtime_version: int,
        risk: StoredOperationalRiskLedger,
        governed: GovernedShadowExecution,
    ) -> GovernedCycleBindReceipt:
        if not lease.acquired:
            raise GovernedCycleBindingError("runtime lease is not acquired")
        if expected_runtime_version <= 0:
            raise ValueError("expected_runtime_version must be positive")
        if risk.runtime_id != lease.runtime_id:
            raise GovernedCycleBindingError(
                "risk ledger runtime_id does not match runtime lease"
            )
        if not governed.shadow_only or governed.live_execution:
            raise GovernedCycleBindingError(
                "governed execution crossed the shadow-only boundary"
            )

        head_receipt_id = _risk_head_receipt_id(risk)
        if governed.operational_risk_receipt_id != head_receipt_id:
            raise GovernedCycleBindingError(
                "governed cycle was not produced from the persisted risk head receipt"
            )
        cycle_id = governed.cycle.cycle_id
        if len(cycle_id) != 64:
            raise GovernedCycleBindingError(
                "governed cycle_id must be a content-addressed Phase57 id"
            )

        row = self._call(
            "brian_bind_governed_shadow_cycle",
            {
                "p_runtime_id": lease.runtime_id,
                "p_owner_token": lease.owner_token,
                "p_fencing_token": lease.fencing_token,
                "p_expected_runtime_version": int(expected_runtime_version),
                "p_risk_version": int(risk.version),
                "p_risk_ledger_hash": risk.ledger_hash,
                "p_risk_receipt_id": head_receipt_id,
                "p_cycle_id": cycle_id,
                "p_governed_result_id": governed.result_id,
                "p_policy_fingerprint": governed.policy_fingerprint,
            },
        )
        assert row is not None

        status = str(row.get("status", ""))
        bound = _boolean(row.get("bound"), "bound")
        duplicate = bool(row.get("duplicate", False))
        runtime_version = _integer(row.get("runtime_version", 0), "runtime_version")
        risk_version = _integer(row.get("risk_version", 0), "risk_version")
        fencing_token = _integer(
            row.get("fencing_token", lease.fencing_token),
            "fencing_token",
        )

        if status in {"BOUND", "DUPLICATE"} and not bound:
            raise GovernedCycleBindingError(
                f"{status} binding receipt returned bound=false"
            )
        if status == "BOUND" and duplicate:
            raise GovernedCycleBindingError("BOUND cannot be marked duplicate")
        if status == "DUPLICATE" and not duplicate:
            raise GovernedCycleBindingError("DUPLICATE must be marked duplicate")
        if status in {
            "LEASE_LOST",
            "RUNTIME_VERSION_CONFLICT",
            "RISK_VERSION_CONFLICT",
        } and bound:
            raise GovernedCycleBindingError(
                f"{status} cannot return bound=true"
            )

        # Successful DB responses must echo every immutable evidence anchor.
        if bound:
            for key, expected in (
                ("cycle_id", cycle_id),
                ("governed_result_id", governed.result_id),
                ("policy_fingerprint", governed.policy_fingerprint),
                ("risk_ledger_hash", risk.ledger_hash),
                ("risk_receipt_id", head_receipt_id),
            ):
                if str(row.get(key, "")) != expected:
                    raise GovernedCycleBindingError(
                        f"database binding receipt {key} does not match submitted evidence"
                    )

        return GovernedCycleBindReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=cycle_id,
            runtime_version=runtime_version,
            risk_version=risk_version,
            fencing_token=fencing_token,
            status=status,
            bound=bound,
            duplicate=duplicate,
            governed_result_id=governed.result_id,
            policy_fingerprint=governed.policy_fingerprint,
            risk_ledger_hash=risk.ledger_hash,
            risk_receipt_id=head_receipt_id,
        )

    def load(
        self,
        *,
        runtime_id: str,
        cycle_id: str,
    ) -> GovernedCycleBinding | None:
        if not runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(cycle_id) != 64:
            raise ValueError("cycle_id must be a content hash")
        row = self._call(
            "brian_read_governed_cycle_binding",
            {
                "p_runtime_id": runtime_id,
                "p_cycle_id": cycle_id,
            },
            allow_none=True,
        )
        if row is None:
            return None
        if str(row.get("runtime_id", "")) != runtime_id:
            raise GovernedCycleBindingError("binding runtime_id mismatch")
        if str(row.get("cycle_id", "")) != cycle_id:
            raise GovernedCycleBindingError("binding cycle_id mismatch")
        if row.get("shadow_only") is not True or row.get("live_execution") is not False:
            raise GovernedCycleBindingError("stored binding crossed live boundary")
        return GovernedCycleBinding(
            runtime_id=runtime_id,
            cycle_id=cycle_id,
            governed_result_id=str(row.get("governed_result_id", "")),
            policy_fingerprint=str(row.get("policy_fingerprint", "")),
            risk_version=_integer(row.get("risk_version"), "risk_version"),
            risk_ledger_hash=str(row.get("risk_ledger_hash", "")),
            risk_receipt_id=str(row.get("risk_receipt_id", "")),
            runtime_version_before=_integer(
                row.get("runtime_version_before"),
                "runtime_version_before",
            ),
            fencing_token=_integer(row.get("fencing_token"), "fencing_token"),
            bound_at=row.get("bound_at"),
            shadow_only=True,
            live_execution=False,
        )
