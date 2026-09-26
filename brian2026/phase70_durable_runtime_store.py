from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping
import math

from .phase67_durable_runtime_orchestrator import (
    DurableRuntimeCheckpoint,
    DurableRuntimeError,
)

PHASE70_SCHEMA_VERSION = "brian.phase70-durable-runtime-store.v1"

RpcCall = Callable[[str, Mapping[str, object]], object]


class DurableRuntimeStoreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeLease:
    runtime_id: str
    owner_token: str
    fencing_token: int
    version: int
    status: str
    acquired: bool
    lease_until: object | None
    schema_version: str = PHASE70_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.runtime_id.strip() or not self.owner_token.strip():
            raise ValueError("runtime_id and owner_token are required")
        if self.fencing_token <= 0:
            raise ValueError("fencing_token must be positive")
        if self.version < 0:
            raise ValueError("version must be non-negative")


@dataclass(frozen=True, slots=True)
class RuntimeCommitReceipt:
    runtime_id: str
    checkpoint_id: str
    fencing_token: int
    version: int
    status: str
    committed: bool
    duplicate: bool
    current_version: int
    schema_version: str = PHASE70_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(self.checkpoint_id) != 64:
            raise ValueError("checkpoint_id must be a content hash")
        if self.fencing_token <= 0:
            raise ValueError("fencing_token must be positive")
        if self.version < 0 or self.current_version < 0:
            raise ValueError("runtime versions must be non-negative")


@dataclass(frozen=True, slots=True)
class StoredRuntimeCheckpoint:
    runtime_id: str
    version: int
    checkpoint: DurableRuntimeCheckpoint
    journal_hash: str
    head_state_id: str
    pending_cycle_id: str | None
    fencing_token: int
    lease_until: object | None
    schema_version: str = PHASE70_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if self.version <= 0:
            raise ValueError("stored checkpoint version must be positive")
        if len(self.journal_hash) != 64 or len(self.head_state_id) != 64:
            raise ValueError("stored journal/head ids must be content hashes")
        if self.fencing_token <= 0:
            raise ValueError("fencing_token must be positive")


def _as_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DurableRuntimeStoreError(f"{label} RPC returned non-object payload")
    return {str(key): item for key, item in value.items()}


def _bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    raise DurableRuntimeStoreError(f"{label} must be boolean")


def _int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise DurableRuntimeStoreError(f"{label} must be integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise DurableRuntimeStoreError(f"{label} must be integer") from exc
    if isinstance(value, float) and not math.isclose(value, number):
        raise DurableRuntimeStoreError(f"{label} must be integer")
    return number


class DurableRuntimeStore:
    """Fail-closed Phase70 client for the transactional Postgres RPC boundary.

    The caller injects a tiny RPC transport adapter. For Supabase Python this can
    be a wrapper around client.rpc(name, params).execute().data; tests and other
    runtimes can provide the same contract without making supabase-py a core
    Brian dependency.

    This client never trusts DB JSON blindly: loaded durable checkpoints are
    reconstructed through Phase63/67 content-hash validation before being
    returned to the runtime.
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
        return _as_mapping(result, label=name)

    def acquire(
        self,
        *,
        runtime_id: str,
        owner_token: str,
        lease_seconds: int,
    ) -> RuntimeLease:
        if not runtime_id.strip() or not owner_token.strip():
            raise ValueError("runtime_id and owner_token are required")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        row = self._call(
            "brian_acquire_shadow_runtime_lease",
            {
                "p_runtime_id": runtime_id,
                "p_owner_token": owner_token,
                "p_lease_seconds": int(lease_seconds),
            },
        )
        assert row is not None
        return RuntimeLease(
            runtime_id=runtime_id,
            owner_token=owner_token,
            fencing_token=_int(row.get("fencing_token"), label="fencing_token"),
            version=_int(row.get("version"), label="version"),
            status=str(row.get("status", "")),
            acquired=_bool(row.get("acquired"), label="acquired"),
            lease_until=row.get("lease_until"),
        )

    def renew(
        self,
        lease: RuntimeLease,
        *,
        lease_seconds: int,
    ) -> RuntimeLease:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        row = self._call(
            "brian_renew_shadow_runtime_lease",
            {
                "p_runtime_id": lease.runtime_id,
                "p_owner_token": lease.owner_token,
                "p_fencing_token": lease.fencing_token,
                "p_lease_seconds": int(lease_seconds),
            },
        )
        assert row is not None
        renewed = _bool(row.get("renewed"), label="renewed")
        current_fence = _int(row.get("fencing_token"), label="fencing_token")
        current_version = _int(row.get("version"), label="version")
        return RuntimeLease(
            runtime_id=lease.runtime_id,
            owner_token=lease.owner_token,
            fencing_token=current_fence,
            version=current_version,
            status=str(row.get("status", "")),
            acquired=renewed,
            lease_until=row.get("lease_until"),
        )

    def release(self, lease: RuntimeLease) -> bool:
        row = self._call(
            "brian_release_shadow_runtime_lease",
            {
                "p_runtime_id": lease.runtime_id,
                "p_owner_token": lease.owner_token,
                "p_fencing_token": lease.fencing_token,
            },
        )
        assert row is not None
        return _bool(row.get("released"), label="released")

    def commit(
        self,
        lease: RuntimeLease,
        *,
        expected_version: int,
        checkpoint: DurableRuntimeCheckpoint,
    ) -> RuntimeCommitReceipt:
        if expected_version < 0:
            raise ValueError("expected_version must be non-negative")

        # Round-trip through the parser before transport so a mutated in-memory
        # object/manifest cannot be persisted merely because it still has fields
        # with plausible names.
        try:
            canonical = DurableRuntimeCheckpoint.from_dict(checkpoint.to_dict())
        except (DurableRuntimeError, ValueError, TypeError, KeyError) as exc:
            raise DurableRuntimeStoreError(
                f"checkpoint failed Phase67 validation before commit: {exc}"
            ) from exc

        row = self._call(
            "brian_commit_shadow_runtime_checkpoint",
            {
                "p_runtime_id": lease.runtime_id,
                "p_owner_token": lease.owner_token,
                "p_fencing_token": lease.fencing_token,
                "p_expected_version": int(expected_version),
                "p_checkpoint": canonical.to_dict(),
            },
        )
        assert row is not None

        status = str(row.get("status", ""))
        committed = _bool(row.get("committed"), label="committed")
        version = _int(row.get("version"), label="version")
        current_version = _int(
            row.get("current_version", version),
            label="current_version",
        )
        returned_checkpoint_id = str(row.get("checkpoint_id", ""))
        if returned_checkpoint_id != canonical.checkpoint_id:
            raise DurableRuntimeStoreError(
                "database commit receipt checkpoint_id does not match submitted checkpoint"
            )
        returned_fence = _int(
            row.get("fencing_token", lease.fencing_token),
            label="fencing_token",
        )

        duplicate = bool(row.get("duplicate", False))
        if status == "COMMITTED" and not committed:
            raise DurableRuntimeStoreError("COMMITTED status returned committed=false")
        if status in {"CAS_CONFLICT", "LEASE_LOST"} and committed:
            raise DurableRuntimeStoreError(f"{status} cannot return committed=true")
        if status.startswith("DUPLICATE_") and not (committed and duplicate):
            raise DurableRuntimeStoreError(
                "duplicate checkpoint receipt has inconsistent flags"
            )

        return RuntimeCommitReceipt(
            runtime_id=lease.runtime_id,
            checkpoint_id=canonical.checkpoint_id,
            fencing_token=returned_fence,
            version=version,
            status=status,
            committed=committed,
            duplicate=duplicate,
            current_version=current_version,
        )

    def load(self, *, runtime_id: str) -> StoredRuntimeCheckpoint | None:
        if not runtime_id.strip():
            raise ValueError("runtime_id is required")
        row = self._call(
            "brian_read_shadow_runtime_checkpoint",
            {"p_runtime_id": runtime_id},
            allow_none=True,
        )
        if row is None:
            return None
        if str(row.get("runtime_id", "")) != runtime_id:
            raise DurableRuntimeStoreError("database returned a different runtime_id")

        version = _int(row.get("version"), label="version")
        payload = row.get("checkpoint_payload")
        checkpoint_id = row.get("checkpoint_id")
        if version == 0:
            if payload is not None or checkpoint_id is not None:
                raise DurableRuntimeStoreError(
                    "version-0 runtime unexpectedly contains checkpoint payload"
                )
            return None
        if not isinstance(payload, Mapping):
            raise DurableRuntimeStoreError(
                "persisted runtime checkpoint payload is missing"
            )

        try:
            checkpoint = DurableRuntimeCheckpoint.from_dict(payload)
        except (DurableRuntimeError, ValueError, TypeError, KeyError) as exc:
            raise DurableRuntimeStoreError(
                f"persisted checkpoint failed Phase67 validation: {exc}"
            ) from exc

        if str(checkpoint_id) != checkpoint.checkpoint_id:
            raise DurableRuntimeStoreError(
                "head checkpoint_id does not match validated checkpoint payload"
            )
        journal_hash = str(row.get("journal_hash", ""))
        actual_journal_hash = str(
            checkpoint.journal_manifest.get("journal_hash", "")
        )
        if journal_hash != actual_journal_hash:
            raise DurableRuntimeStoreError(
                "head journal_hash does not match validated checkpoint"
            )

        head_state_id = str(row.get("head_state_id", ""))
        actual_head_state_id = str(
            checkpoint.runtime_checkpoint.shadow_ledger_manifest.get(
                "head_state_id",
                "",
            )
        )
        if head_state_id != actual_head_state_id:
            raise DurableRuntimeStoreError(
                "head_state_id does not match validated checkpoint"
            )

        pending = (
            None
            if row.get("pending_cycle_id") is None
            else str(row.get("pending_cycle_id"))
        )
        if pending != checkpoint.runtime_checkpoint.pending_cycle_id:
            raise DurableRuntimeStoreError(
                "pending_cycle_id does not match validated checkpoint"
            )

        return StoredRuntimeCheckpoint(
            runtime_id=runtime_id,
            version=version,
            checkpoint=checkpoint,
            journal_hash=journal_hash,
            head_state_id=head_state_id,
            pending_cycle_id=pending,
            fencing_token=_int(row.get("fencing_token"), label="fencing_token"),
            lease_until=row.get("lease_until"),
        )
