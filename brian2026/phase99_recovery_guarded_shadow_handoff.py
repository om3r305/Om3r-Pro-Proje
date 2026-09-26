from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .phase69_governed_shadow_execution import GovernedShadowExecution
from .phase71_persisted_runtime_supervisor import PersistedRuntimeStaleError
from .phase73_operational_risk_store import OperationalRiskStore
from .phase75_atomic_governed_writeahead import (
    AtomicGovernedWriteAheadStore,
    PersistedGovernedRuntimeSupervisor,
)
from .phase76_shadow_execution_outbox import (
    PersistedDispatchedRuntimeSupervisor,
    ShadowExecutionOutboxStore,
)
from .phase77_execution_claim_lifecycle import (
    ExecutionClaimStore,
    PersistedClaimedRuntimeSupervisor,
)
from .phase78_execution_kill_switch import ExecutionKillSwitchStore
from .phase79_atomic_execution_start import AtomicExecutionStartStore
from .phase80_claim_fenced_checkpoint import (
    ClaimFencedCheckpointStore,
    ClaimFencedExecutionStep,
    PersistedClaimFencedRuntimeSupervisor,
)
from .phase86_recovery_admission_interlock import RecoveryAdmissionState
from .phase97_bounded_auto_recovery_drain import BoundedAutoRecoveryDrainReceipt

PHASE99_SCHEMA_VERSION = "brian.phase99-recovery-guarded-shadow-handoff.v1"


class RecoveryGuardedShadowHandoffError(RuntimeError):
    pass


class RecoveryGuardedShadowBlockedError(RecoveryGuardedShadowHandoffError):
    pass


@dataclass(frozen=True, slots=True)
class NormalShadowExecutionStack:
    runtime_supervisor: object
    risk_store: OperationalRiskStore
    atomic_writeahead: AtomicGovernedWriteAheadStore
    governed: PersistedGovernedRuntimeSupervisor
    outbox: ShadowExecutionOutboxStore
    dispatched: PersistedDispatchedRuntimeSupervisor
    claims: ExecutionClaimStore
    claimed: PersistedClaimedRuntimeSupervisor
    kill_switch: ExecutionKillSwitchStore
    starts: AtomicExecutionStartStore
    checkpoints: ClaimFencedCheckpointStore
    execution: PersistedClaimFencedRuntimeSupervisor
    schema_version: str = PHASE99_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not getattr(self.runtime_supervisor, "valid", False):
            raise PersistedRuntimeStaleError(
                "Phase99 cannot assemble normal shadow stack on stale runtime"
            )
        if self.governed.runtime_supervisor is not self.runtime_supervisor:
            raise ValueError("Phase99 governed runtime authority wiring drift")
        if self.governed.risk_store is not self.risk_store:
            raise ValueError("Phase99 operational-risk store wiring drift")
        if self.governed.atomic_store is not self.atomic_writeahead:
            raise ValueError("Phase99 atomic write-ahead store wiring drift")
        if self.dispatched.governed_supervisor is not self.governed:
            raise ValueError("Phase99 dispatch supervisor wiring drift")
        if self.dispatched.outbox is not self.outbox:
            raise ValueError("Phase99 outbox store wiring drift")
        if self.claimed.dispatched_supervisor is not self.dispatched:
            raise ValueError("Phase99 claim supervisor wiring drift")
        if self.claimed.claims is not self.claims:
            raise ValueError("Phase99 claim store wiring drift")
        if self.execution.claimed_supervisor is not self.claimed:
            raise ValueError("Phase99 execution claim authority wiring drift")
        if self.execution.starts is not self.starts:
            raise ValueError("Phase99 STARTED store wiring drift")
        if self.execution.checkpoints is not self.checkpoints:
            raise ValueError("Phase99 checkpoint store wiring drift")
        if self.execution.kill_switch is not self.kill_switch:
            raise ValueError("Phase99 kill-switch store wiring drift")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase99 normal stack must remain shadow-only")


@dataclass(frozen=True, slots=True)
class RecoveryGuardedShadowExecutionReceipt:
    runtime_id: str
    admission: RecoveryAdmissionState
    execution: ClaimFencedExecutionStep
    status: str
    schema_version: str = PHASE99_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if self.admission.runtime_id != self.runtime_id:
            raise ValueError("Phase99 admission runtime mismatch")
        if self.admission.blocked or self.admission.status != "OPEN":
            raise ValueError("Phase99 execution receipt requires OPEN admission")
        if not getattr(self.execution, "shadow_only", False):
            raise ValueError("Phase99 execution result is not shadow-only")
        if getattr(self.execution, "live_execution", True):
            raise ValueError("Phase99 execution result crossed live boundary")
        if self.status != "NORMAL_SHADOW_PROCESSED":
            raise ValueError("invalid Phase99 execution status")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase99 receipt must remain shadow-only")


def build_normal_shadow_execution_stack(
    *,
    rpc,
    runtime_supervisor,
) -> NormalShadowExecutionStack:
    """Assemble Phase73/75–80 from one RPC transport and runtime authority."""
    if not callable(rpc):
        raise TypeError("rpc transport must be callable")
    if not getattr(runtime_supervisor, "valid", False):
        raise PersistedRuntimeStaleError(
            "cannot assemble normal shadow stack on stale runtime supervisor"
        )

    risk_store = OperationalRiskStore(rpc)
    atomic_writeahead = AtomicGovernedWriteAheadStore(rpc)
    governed = PersistedGovernedRuntimeSupervisor(
        runtime_supervisor=runtime_supervisor,
        risk_store=risk_store,
        atomic_store=atomic_writeahead,
    )
    outbox = ShadowExecutionOutboxStore(rpc)
    dispatched = PersistedDispatchedRuntimeSupervisor(
        governed_supervisor=governed,
        outbox=outbox,
    )
    claims = ExecutionClaimStore(rpc)
    claimed = PersistedClaimedRuntimeSupervisor(
        dispatched_supervisor=dispatched,
        claims=claims,
    )
    kill_switch = ExecutionKillSwitchStore(rpc)
    starts = AtomicExecutionStartStore(rpc)
    checkpoints = ClaimFencedCheckpointStore(rpc)
    execution = PersistedClaimFencedRuntimeSupervisor(
        claimed_supervisor=claimed,
        starts=starts,
        checkpoints=checkpoints,
        kill_switch=kill_switch,
    )
    return NormalShadowExecutionStack(
        runtime_supervisor=runtime_supervisor,
        risk_store=risk_store,
        atomic_writeahead=atomic_writeahead,
        governed=governed,
        outbox=outbox,
        dispatched=dispatched,
        claims=claims,
        claimed=claimed,
        kill_switch=kill_switch,
        starts=starts,
        checkpoints=checkpoints,
        execution=execution,
    )


class RecoveryGuardedShadowHandoff:
    """Hand normal shadow work the same Phase92 runtime only after recovery.

    Phase97's READY receipt is required at construction. Every normal shadow
    cycle re-reads the exact Phase86 admission authority immediately before
    Phase75/76/77/79/80 processing. If a barrier appears after this pre-read,
    Phase86's database wrappers around Phase75 authorization and Phase76
    dispatch remain the transactional final interlock.
    """

    def __init__(
        self,
        *,
        session,
        recovery: BoundedAutoRecoveryDrainReceipt,
        normal: NormalShadowExecutionStack,
    ) -> None:
        if getattr(session, "closed", False):
            raise RecoveryGuardedShadowHandoffError(
                "cannot hand off from a closed recovery worker session"
            )
        runtime_id = str(getattr(session, "runtime_id", "")).strip()
        if not runtime_id:
            raise RecoveryGuardedShadowHandoffError(
                "session runtime_id is required"
            )
        supervisor = getattr(session, "runtime_supervisor", None)
        if supervisor is None or not getattr(supervisor, "valid", False):
            raise PersistedRuntimeStaleError(
                "Phase99 requires a valid Phase71 runtime supervisor"
            )
        if recovery.runtime_id != runtime_id:
            raise RecoveryGuardedShadowHandoffError(
                "Phase97 recovery runtime differs from Phase92 session"
            )
        if (
            not recovery.ready_for_normal_work
            or recovery.status != "READY_FOR_NORMAL_WORK"
            or recovery.final_admission.blocked
            or recovery.final_admission.status != "OPEN"
        ):
            raise RecoveryGuardedShadowBlockedError(
                "Phase97 did not authorize normal shadow handoff"
            )
        if normal.runtime_supervisor is not supervisor:
            raise RecoveryGuardedShadowHandoffError(
                "normal shadow stack uses different runtime authority"
            )
        admission = getattr(getattr(session, "stack", None), "admission", None)
        if admission is None or not hasattr(admission, "read"):
            raise RecoveryGuardedShadowHandoffError(
                "session stack must expose Phase86 admission reader"
            )

        self.session = session
        self.recovery = recovery
        self.normal = normal
        self.admission = admission

    @classmethod
    def from_recovery(
        cls,
        *,
        session,
        recovery: BoundedAutoRecoveryDrainReceipt,
    ) -> "RecoveryGuardedShadowHandoff":
        rpc = getattr(session, "rpc", None)
        supervisor = getattr(session, "runtime_supervisor", None)
        normal = build_normal_shadow_execution_stack(
            rpc=rpc,
            runtime_supervisor=supervisor,
        )
        return cls(
            session=session,
            recovery=recovery,
            normal=normal,
        )

    @property
    def runtime_id(self) -> str:
        return self.recovery.runtime_id

    def process_governed_cycle(
        self,
        governed: GovernedShadowExecution,
        *,
        worker_token: str,
        claim_seconds: int,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> RecoveryGuardedShadowExecutionReceipt:
        if getattr(self.session, "closed", False):
            raise RecoveryGuardedShadowHandoffError(
                "recovery worker session closed before normal shadow cycle"
            )
        supervisor = self.normal.runtime_supervisor
        if not getattr(supervisor, "valid", False):
            raise PersistedRuntimeStaleError(
                "runtime supervisor became stale before normal shadow cycle"
            )
        if not worker_token.strip():
            raise ValueError("worker_token is required")
        if claim_seconds < 10 or claim_seconds > 300:
            raise ValueError("claim_seconds must be in [10,300]")
        if not math.isfinite(float(observed_at)):
            raise ValueError("observed_at must be finite")
        if not source_ref.strip():
            raise ValueError("source_ref is required")
        if not getattr(governed, "shadow_only", False):
            raise RecoveryGuardedShadowHandoffError(
                "governed cycle is not shadow-only"
            )
        if getattr(governed, "live_execution", True):
            raise RecoveryGuardedShadowHandoffError(
                "governed cycle crossed live-execution boundary"
            )

        admission = self.admission.read(runtime_id=self.runtime_id)
        if admission.blocked or admission.status != "OPEN":
            raise RecoveryGuardedShadowBlockedError(
                "Phase86 recovery barrier blocks normal shadow cycle"
            )

        execution = self.normal.execution.process_governed_cycle(
            governed,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
            marks=marks,
            observed_at=float(observed_at),
            source_ref=source_ref,
        )
        return RecoveryGuardedShadowExecutionReceipt(
            runtime_id=self.runtime_id,
            admission=admission,
            execution=execution,
            status="NORMAL_SHADOW_PROCESSED",
        )
