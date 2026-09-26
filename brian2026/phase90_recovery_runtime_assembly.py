from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .phase71_persisted_runtime_supervisor import PersistedRuntimeStaleError
from .phase81_cancel_recovery_directive import CancelRecoveryDirectiveStore
from .phase82_recovery_claim_fencing import RecoveryClaimStore
from .phase83_atomic_recovery_start import AtomicRecoveryStartStore
from .phase84_recovery_execution_checkpoint import (
    PersistedRecoveryExecutionSupervisor,
    RecoveryCheckpointStore,
)
from .phase85_recovery_completion_audit import RecoveryCompletionAuditStore
from .phase86_recovery_admission_interlock import RecoveryAdmissionInterlockStore
from .phase87_recovery_restart_resume import RecoveryRestartWorkStore
from .phase88_recovery_restart_orchestrator import RecoveryRestartOrchestrator
from .phase89_recovery_startup_gate import RecoveryStartupGate

PHASE90_SCHEMA_VERSION = "brian.phase90-recovery-runtime-assembly.v1"
RpcCall = Callable[[str, Mapping[str, object]], object]


@dataclass(frozen=True, slots=True)
class RecoveryRuntimeStack:
    runtime_supervisor: object
    directives: CancelRecoveryDirectiveStore
    claims: RecoveryClaimStore
    starts: AtomicRecoveryStartStore
    checkpoints: RecoveryCheckpointStore
    execution: PersistedRecoveryExecutionSupervisor
    audits: RecoveryCompletionAuditStore
    admission: RecoveryAdmissionInterlockStore
    backlog: RecoveryRestartWorkStore
    restart: RecoveryRestartOrchestrator
    startup_gate: RecoveryStartupGate
    schema_version: str = PHASE90_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not getattr(self.runtime_supervisor, "valid", False):
            raise PersistedRuntimeStaleError(
                "Phase90 cannot assemble recovery stack on stale runtime"
            )
        if self.execution._runtime_supervisor() is not self.runtime_supervisor:
            raise ValueError("Phase84 runtime supervisor wiring drift")
        if self.execution._claims() is not self.claims:
            raise ValueError("Phase84 recovery claim store wiring drift")
        if self.restart.runtime_supervisor is not self.runtime_supervisor:
            raise ValueError("Phase88 runtime supervisor wiring drift")
        if self.restart.directives is not self.directives:
            raise ValueError("Phase88 directive store wiring drift")
        if self.restart.claims is not self.claims:
            raise ValueError("Phase88 claim store wiring drift")
        if self.restart.starts is not self.starts:
            raise ValueError("Phase88 start store wiring drift")
        if self.restart.recovery_execution is not self.execution:
            raise ValueError("Phase88 execution supervisor wiring drift")
        if self.restart.audits is not self.audits:
            raise ValueError("Phase88 audit store wiring drift")
        if self.restart.work is not self.backlog:
            raise ValueError("Phase88 backlog store wiring drift")
        if self.startup_gate.recovery is not self.restart:
            raise ValueError("Phase89 restart orchestrator wiring drift")
        if self.startup_gate.admission is not self.admission:
            raise ValueError("Phase89 admission store wiring drift")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase90 recovery stack must remain shadow-only")


def build_recovery_runtime_stack(
    *,
    rpc: RpcCall,
    runtime_supervisor,
    foreign_cycle_aborter=None,
) -> RecoveryRuntimeStack:
    """Build one Phase81-89 recovery stack from one RPC transport/runtime.

    Keeping assembly in one place prevents production workers from accidentally
    mixing stores backed by different Supabase clients, runtime supervisors or
    recovery-claim authorities.
    """
    if not callable(rpc):
        raise TypeError("rpc transport must be callable")
    if not getattr(runtime_supervisor, "valid", False):
        raise PersistedRuntimeStaleError(
            "cannot assemble recovery stack on stale runtime supervisor"
        )
    runtime_id = getattr(runtime_supervisor, "runtime_id", "")
    if not isinstance(runtime_id, str) or not runtime_id.strip():
        raise ValueError("runtime_supervisor must expose runtime_id")

    directives = CancelRecoveryDirectiveStore(rpc)
    claims = RecoveryClaimStore(rpc)
    starts = AtomicRecoveryStartStore(rpc)
    checkpoints = RecoveryCheckpointStore(rpc)
    execution = PersistedRecoveryExecutionSupervisor(
        checkpoints=checkpoints,
        runtime_supervisor=runtime_supervisor,
        claims=claims,
    )
    audits = RecoveryCompletionAuditStore(rpc)
    admission = RecoveryAdmissionInterlockStore(rpc)
    backlog = RecoveryRestartWorkStore(rpc)

    restart = RecoveryRestartOrchestrator(
        runtime_supervisor=runtime_supervisor,
        work=backlog,
        directives=directives,
        claims=claims,
        starts=starts,
        recovery_execution=execution,
        audits=audits,
        foreign_cycle_aborter=foreign_cycle_aborter,
    )
    startup_gate = RecoveryStartupGate(
        recovery=restart,
        admission=admission,
    )

    return RecoveryRuntimeStack(
        runtime_supervisor=runtime_supervisor,
        directives=directives,
        claims=claims,
        starts=starts,
        checkpoints=checkpoints,
        execution=execution,
        audits=audits,
        admission=admission,
        backlog=backlog,
        restart=restart,
        startup_gate=startup_gate,
    )
