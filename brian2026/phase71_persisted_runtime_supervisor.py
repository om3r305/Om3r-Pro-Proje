from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .phase57_shadow_execution_cycle import ShadowExecutionCycle
from .phase67_durable_runtime_orchestrator import (
    DurableRuntimeReceipt,
    DurableShadowPaperRuntime,
)
from .phase70_durable_runtime_store import (
    DurableRuntimeStore,
    RuntimeCommitReceipt,
    RuntimeLease,
)

PHASE71_SCHEMA_VERSION = "brian.phase71-persisted-runtime-supervisor.v1"


class PersistedRuntimeError(RuntimeError):
    pass


class PersistedRuntimeLeaseError(PersistedRuntimeError):
    pass


class PersistedRuntimeStaleError(PersistedRuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PersistedRuntimeStep:
    runtime_id: str
    durable_receipt: DurableRuntimeReceipt | None
    persisted_version: int
    checkpoint_id: str
    commit_status: str
    schema_version: str = PHASE71_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False


class PersistedDurableRuntimeSupervisor:
    """Lease-owned Phase67 runtime with transactional Phase70 checkpoints.

    The supervisor persists the full write-ahead journal checkpoint before
    calling Phase67.advance_pending. Therefore a process crash during paper
    simulation/reconciliation can only lose replayable in-memory work; it cannot
    lose the original cycle body.

    Any CAS conflict or lost fencing lease invalidates this supervisor instance.
    It must be reconstructed from the authoritative Phase70 database head rather
    than continuing from stale local state.
    """

    def __init__(
        self,
        *,
        store: DurableRuntimeStore,
        lease: RuntimeLease,
        runtime: DurableShadowPaperRuntime,
        persisted_version: int,
    ) -> None:
        if not lease.acquired:
            raise PersistedRuntimeLeaseError("cannot supervise runtime without acquired lease")
        if persisted_version < 0:
            raise ValueError("persisted_version must be non-negative")
        self.store = store
        self.lease = lease
        self.runtime = runtime
        self.persisted_version = int(persisted_version)
        self._valid = True

    @property
    def runtime_id(self) -> str:
        return self.lease.runtime_id

    @property
    def valid(self) -> bool:
        return self._valid

    def _assert_valid(self) -> None:
        if not self._valid:
            raise PersistedRuntimeStaleError(
                "persisted runtime supervisor is stale; reload from database head"
            )

    @classmethod
    def acquire(
        cls,
        *,
        store: DurableRuntimeStore,
        runtime_id: str,
        owner_token: str,
        lease_seconds: int,
        initial_runtime: DurableShadowPaperRuntime | None = None,
    ) -> "PersistedDurableRuntimeSupervisor":
        lease = store.acquire(
            runtime_id=runtime_id,
            owner_token=owner_token,
            lease_seconds=lease_seconds,
        )
        if not lease.acquired:
            raise PersistedRuntimeLeaseError(
                f"runtime {runtime_id} lease is owned by another worker"
            )

        stored = store.load(runtime_id=runtime_id)
        if stored is None:
            if lease.version != 0:
                raise PersistedRuntimeStaleError(
                    "database lease reports nonzero version but no checkpoint can be loaded"
                )
            if initial_runtime is None:
                raise PersistedRuntimeError(
                    "initial_runtime is required when durable runtime has no checkpoint"
                )
            supervisor = cls(
                store=store,
                lease=lease,
                runtime=initial_runtime,
                persisted_version=0,
            )
            supervisor._persist_current_checkpoint()
            return supervisor

        if initial_runtime is not None:
            # A durable head already exists. Never overwrite it with caller memory.
            initial_runtime = None
        if stored.version != lease.version:
            raise PersistedRuntimeStaleError(
                "lease version and loaded durable checkpoint version disagree"
            )
        runtime = DurableShadowPaperRuntime.restore(stored.checkpoint)
        return cls(
            store=store,
            lease=lease,
            runtime=runtime,
            persisted_version=stored.version,
        )

    def _persist_current_checkpoint(self) -> RuntimeCommitReceipt:
        self._assert_valid()
        checkpoint = self.runtime.checkpoint()
        receipt = self.store.commit(
            self.lease,
            expected_version=self.persisted_version,
            checkpoint=checkpoint,
        )

        if receipt.status == "COMMITTED":
            if not receipt.committed or receipt.duplicate:
                self._valid = False
                raise PersistedRuntimeStaleError(
                    "database returned inconsistent COMMITTED receipt"
                )
            self.persisted_version = receipt.version
            return receipt

        if receipt.status == "DUPLICATE_CURRENT":
            if not receipt.committed or not receipt.duplicate:
                self._valid = False
                raise PersistedRuntimeStaleError(
                    "database returned inconsistent duplicate receipt"
                )
            if receipt.current_version != receipt.version:
                self._valid = False
                raise PersistedRuntimeStaleError(
                    "duplicate-current receipt version mismatch"
                )
            self.persisted_version = receipt.current_version
            return receipt

        # DUPLICATE_HISTORICAL means this local checkpoint is older than the DB
        # head. CAS_CONFLICT and LEASE_LOST are also definitive stale/ownership
        # signals. Continuing from this in-memory runtime would risk a fork.
        self._valid = False
        if receipt.status == "LEASE_LOST":
            raise PersistedRuntimeLeaseError(
                "runtime lease/fencing ownership was lost during checkpoint commit"
            )
        raise PersistedRuntimeStaleError(
            f"runtime checkpoint commit rejected with {receipt.status}; reload required"
        )

    def renew(self, *, lease_seconds: int) -> RuntimeLease:
        self._assert_valid()
        renewed = self.store.renew(self.lease, lease_seconds=lease_seconds)
        if not renewed.acquired:
            self._valid = False
            raise PersistedRuntimeLeaseError("runtime lease renewal failed")
        if renewed.fencing_token != self.lease.fencing_token:
            self._valid = False
            raise PersistedRuntimeLeaseError("runtime fencing token changed during renewal")
        if renewed.version != self.persisted_version:
            self._valid = False
            raise PersistedRuntimeStaleError(
                "database runtime version advanced outside this supervisor"
            )
        self.lease = renewed
        return renewed

    def release(self) -> bool:
        if not self._valid:
            # A stale supervisor may no longer own the lease. Still issue the
            # owner+fence gated release; the DB will reject it if ownership moved.
            return self.store.release(self.lease)
        released = self.store.release(self.lease)
        if released:
            self._valid = False
        return released

    def persist_write_ahead_cycle(
        self,
        cycle: ShadowExecutionCycle,
    ) -> PersistedRuntimeStep:
        self._assert_valid()
        self.runtime.journal_cycle(cycle)
        commit = self._persist_current_checkpoint()
        checkpoint = self.runtime.checkpoint()
        return PersistedRuntimeStep(
            runtime_id=self.runtime_id,
            durable_receipt=None,
            persisted_version=self.persisted_version,
            checkpoint_id=checkpoint.checkpoint_id,
            commit_status=commit.status,
        )

    def advance_pending(
        self,
        *,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> PersistedRuntimeStep:
        self._assert_valid()
        durable = self.runtime.advance_pending(
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )
        commit = self._persist_current_checkpoint()
        checkpoint = self.runtime.checkpoint()
        return PersistedRuntimeStep(
            runtime_id=self.runtime_id,
            durable_receipt=durable,
            persisted_version=self.persisted_version,
            checkpoint_id=checkpoint.checkpoint_id,
            commit_status=commit.status,
        )

    def process_cycle(
        self,
        cycle: ShadowExecutionCycle,
        *,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> tuple[PersistedRuntimeStep, PersistedRuntimeStep]:
        """Persist cycle body first, then advance and persist resulting state."""
        write_ahead = self.persist_write_ahead_cycle(cycle)
        advanced = self.advance_pending(
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )
        return write_ahead, advanced
