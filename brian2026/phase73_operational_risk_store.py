from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .phase70_durable_runtime_store import RuntimeLease
from .phase72_operational_risk_ledger import (
    OperationalRiskLedger,
    OperationalRiskLedgerError,
    restore_operational_risk_ledger,
)

PHASE73_SCHEMA_VERSION = "brian.phase73-operational-risk-store.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


class OperationalRiskStoreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OperationalRiskCommitReceipt:
    runtime_id: str
    ledger_hash: str
    fencing_token: int
    version: int
    current_version: int
    status: str
    committed: bool
    duplicate: bool
    schema_version: str = PHASE73_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(self.ledger_hash) != 64:
            raise ValueError("ledger_hash must be a content hash")
        if self.fencing_token <= 0:
            raise ValueError("fencing_token must be positive")
        if self.version < 0 or self.current_version < 0:
            raise ValueError("risk versions must be non-negative")


@dataclass(frozen=True, slots=True)
class StoredOperationalRiskLedger:
    runtime_id: str
    version: int
    ledger: OperationalRiskLedger
    ledger_hash: str
    policy_hash: str
    head_entry_id: str | None
    current_state: str
    halt_latched: bool
    schema_version: str = PHASE73_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if self.version <= 0:
            raise ValueError("stored risk version must be positive")
        if len(self.ledger_hash) != 64 or len(self.policy_hash) != 64:
            raise ValueError("stored risk hashes must be content hashes")
        if self.head_entry_id is not None and len(self.head_entry_id) != 64:
            raise ValueError("head_entry_id must be a content hash when set")
        if self.current_state not in ("ACTIVE", "REDUCING", "HALTED"):
            raise ValueError("invalid current_state")
        if self.halt_latched != (self.current_state == "HALTED"):
            raise ValueError("stored HALT latch/state mismatch")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise OperationalRiskStoreError(f"{label} returned a non-object payload")
    return {str(key): item for key, item in value.items()}


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise OperationalRiskStoreError(f"{label} must be integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise OperationalRiskStoreError(f"{label} must be integer") from exc
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise OperationalRiskStoreError(f"{label} must be boolean")
    return value


class OperationalRiskStore:
    """Transport-agnostic Phase73 client over the Phase70 runtime fence."""

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

    def commit(
        self,
        lease: RuntimeLease,
        *,
        expected_version: int,
        ledger: OperationalRiskLedger,
    ) -> OperationalRiskCommitReceipt:
        if expected_version < 0:
            raise ValueError("expected_version must be non-negative")

        manifest = ledger.manifest()
        # Restore from our own serialized representation before it crosses the DB
        # boundary. This catches mutated internal manifests and seals the same
        # parser used on load.
        try:
            canonical = restore_operational_risk_ledger(manifest)
        except (OperationalRiskLedgerError, ValueError, TypeError, KeyError) as exc:
            raise OperationalRiskStoreError(
                f"operational-risk ledger failed validation before commit: {exc}"
            ) from exc
        canonical_manifest = canonical.manifest()
        ledger_hash = str(canonical_manifest["ledger_hash"])

        row = self._call(
            "brian_commit_operational_risk_ledger",
            {
                "p_runtime_id": lease.runtime_id,
                "p_owner_token": lease.owner_token,
                "p_fencing_token": lease.fencing_token,
                "p_expected_version": int(expected_version),
                "p_manifest": canonical_manifest,
            },
        )
        assert row is not None

        returned_hash = str(row.get("ledger_hash", ""))
        if returned_hash != ledger_hash:
            raise OperationalRiskStoreError(
                "database risk commit receipt ledger_hash does not match submitted ledger"
            )

        status = str(row.get("status", ""))
        committed = _boolean(row.get("committed"), "committed")
        duplicate = bool(row.get("duplicate", False))
        version = _integer(row.get("version", 0), "version")
        current_version = _integer(
            row.get("current_version", version),
            "current_version",
        )
        fence = _integer(
            row.get("fencing_token", lease.fencing_token),
            "fencing_token",
        )

        if status == "COMMITTED" and (not committed or duplicate):
            raise OperationalRiskStoreError("inconsistent COMMITTED risk receipt")
        if status.startswith("DUPLICATE_") and not (committed and duplicate):
            raise OperationalRiskStoreError("inconsistent duplicate risk receipt")
        if status in {"CAS_CONFLICT", "LEASE_LOST"} and committed:
            raise OperationalRiskStoreError(f"{status} cannot be committed")

        return OperationalRiskCommitReceipt(
            runtime_id=lease.runtime_id,
            ledger_hash=ledger_hash,
            fencing_token=fence,
            version=version,
            current_version=current_version,
            status=status,
            committed=committed,
            duplicate=duplicate,
        )

    def load(
        self,
        *,
        runtime_id: str,
    ) -> StoredOperationalRiskLedger | None:
        if not runtime_id.strip():
            raise ValueError("runtime_id is required")
        row = self._call(
            "brian_read_operational_risk_ledger",
            {"p_runtime_id": runtime_id},
            allow_none=True,
        )
        if row is None:
            return None
        if str(row.get("runtime_id", "")) != runtime_id:
            raise OperationalRiskStoreError("database returned a different runtime_id")

        version = _integer(row.get("version", 0), "version")
        manifest_raw = row.get("manifest")
        ledger_hash = row.get("ledger_hash")
        if version == 0:
            if manifest_raw is not None or ledger_hash is not None:
                raise OperationalRiskStoreError(
                    "version-0 risk head unexpectedly contains a manifest"
                )
            return None
        if not isinstance(manifest_raw, Mapping):
            raise OperationalRiskStoreError("persisted risk manifest is missing")

        try:
            ledger = restore_operational_risk_ledger(manifest_raw)
        except (OperationalRiskLedgerError, ValueError, TypeError, KeyError) as exc:
            raise OperationalRiskStoreError(
                f"persisted operational-risk ledger failed validation: {exc}"
            ) from exc
        manifest = ledger.manifest()

        actual_hash = str(manifest["ledger_hash"])
        if str(ledger_hash) != actual_hash:
            raise OperationalRiskStoreError(
                "database risk ledger_hash does not match validated manifest"
            )
        policy_hash = str(row.get("policy_hash", ""))
        if policy_hash != str(manifest["policy_hash"]):
            raise OperationalRiskStoreError(
                "database risk policy_hash does not match validated manifest"
            )

        head_entry = (
            None if row.get("head_entry_id") is None else str(row["head_entry_id"])
        )
        if head_entry != manifest["head_entry_id"]:
            raise OperationalRiskStoreError(
                "database risk head_entry_id does not match validated manifest"
            )
        current_state = str(row.get("current_state", ""))
        if current_state != manifest["current_state"]:
            raise OperationalRiskStoreError(
                "database risk current_state does not match validated manifest"
            )
        halt_latched = _boolean(row.get("halt_latched"), "halt_latched")
        if halt_latched != manifest["halt_latched"]:
            raise OperationalRiskStoreError(
                "database risk halt_latched does not match validated manifest"
            )

        return StoredOperationalRiskLedger(
            runtime_id=runtime_id,
            version=version,
            ledger=ledger,
            ledger_hash=actual_hash,
            policy_hash=policy_hash,
            head_entry_id=head_entry,
            current_state=current_state,
            halt_latched=halt_latched,
        )
