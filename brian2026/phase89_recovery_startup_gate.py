from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .phase56_pretrade_risk_engine import InstrumentRiskLimits
from .phase57_shadow_execution_cycle import ExecutionMarketInput
from .phase86_recovery_admission_interlock import (
    RecoveryAdmissionInterlockStore,
    RecoveryAdmissionState,
)
from .phase88_recovery_restart_orchestrator import (
    RecoveryRestartOrchestrator,
    RecoveryRestartStep,
)

PHASE89_SCHEMA_VERSION = "brian.phase89-recovery-startup-gate.v1"

_SAFE_CONTINUE_OUTCOMES = {
    "RECOVERY_COMPLETED",
    "NO_RECOVERY_REQUIRED",
}
_SAFE_READY_OUTCOMES = _SAFE_CONTINUE_OUTCOMES | {"IDLE"}


@dataclass(frozen=True, slots=True)
class RecoveryStartupGateReceipt:
    runtime_id: str
    steps: tuple[RecoveryRestartStep, ...]
    admission: RecoveryAdmissionState
    status: str
    ready_for_normal_work: bool
    processed_items: int
    max_items: int
    schema_version: str = PHASE89_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if self.admission.runtime_id != self.runtime_id:
            raise ValueError("admission runtime does not match startup gate")
        if self.max_items <= 0:
            raise ValueError("max_items must be positive")
        if self.processed_items < 0 or self.processed_items > self.max_items:
            raise ValueError("processed_items outside bounded worker budget")
        if self.ready_for_normal_work:
            if self.status != "READY_FOR_NORMAL_WORK":
                raise ValueError("ready gate must use READY_FOR_NORMAL_WORK")
            if self.admission.blocked:
                raise ValueError("normal work cannot be ready behind recovery barrier")
            if self.steps and self.steps[-1].outcome not in _SAFE_READY_OUTCOMES:
                raise ValueError("non-terminal recovery outcome cannot release startup gate")
        elif self.status == "READY_FOR_NORMAL_WORK":
            raise ValueError("READY_FOR_NORMAL_WORK requires ready=true")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase89 startup gate must remain shadow-only")


class RecoveryStartupGate:
    """Bounded restart-recovery drain before normal shadow work is admitted.

    The Phase86 DB admission read is the final authority. Even if Phase88 saw
    IDLE a moment earlier, a newly-created recovery barrier keeps normal work
    closed. Conversely an OPEN admission never overrides a non-terminal Phase88
    result from the same invocation.
    """

    def __init__(
        self,
        *,
        recovery: RecoveryRestartOrchestrator,
        admission: RecoveryAdmissionInterlockStore,
    ) -> None:
        self.recovery = recovery
        self.admission = admission

    def run(
        self,
        *,
        max_items: int,
        recovery_worker_token: str,
        recovery_claim_seconds: int,
        recovery_markets: Mapping[str, ExecutionMarketInput],
        recovery_risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
        recovery_ttl_seconds: int,
        marks: Mapping[str, float],
        observed_at: float,
        source_ref: str,
    ) -> RecoveryStartupGateReceipt:
        if max_items <= 0:
            raise ValueError("max_items must be positive")

        runtime_id = self.recovery.runtime_supervisor.runtime_id
        steps: list[RecoveryRestartStep] = []
        processed = 0

        for _ in range(max_items):
            step = self.recovery.resume_next(
                recovery_worker_token=recovery_worker_token,
                recovery_claim_seconds=recovery_claim_seconds,
                recovery_markets=recovery_markets,
                recovery_risk_limits_by_asset=recovery_risk_limits_by_asset,
                recovery_ttl_seconds=recovery_ttl_seconds,
                marks=marks,
                observed_at=observed_at,
                source_ref=source_ref,
            )
            steps.append(step)

            if step.outcome == "IDLE":
                break

            processed += 1
            if step.outcome in _SAFE_CONTINUE_OUTCOMES:
                continue

            # WAIT, MANUAL_REVIEW, MARKS_REQUIRED, RECONCILIATION_BLOCKED,
            # foreign-cycle wait and any future non-terminal outcome stop this
            # bounded invocation. Never spin on external dependencies.
            break

        authoritative = self.admission.read(runtime_id=runtime_id)
        last_outcome = "IDLE" if not steps else steps[-1].outcome

        ready = (
            not authoritative.blocked
            and last_outcome in _SAFE_READY_OUTCOMES
        )

        if ready:
            status = "READY_FOR_NORMAL_WORK"
        elif authoritative.blocked and last_outcome == "IDLE":
            # Race-safe case: a new AFTER_START barrier appeared between the
            # Phase87 IDLE read and this final Phase86 admission read.
            status = "RECOVERY_BARRIER_APPEARED"
        elif authoritative.blocked and (
            last_outcome in _SAFE_CONTINUE_OUTCOMES
            and processed >= max_items
        ):
            status = "RECOVERY_BUDGET_EXHAUSTED"
        elif authoritative.blocked and last_outcome in _SAFE_CONTINUE_OUTCOMES:
            status = "RECOVERY_BACKLOG_REMAINS"
        elif authoritative.blocked:
            status = "RECOVERY_BLOCKED"
        else:
            # Admission may have opened concurrently, but the worker itself saw
            # a non-terminal outcome. Fail conservatively for this invocation.
            status = "RECOVERY_OUTCOME_NOT_TERMINAL"

        return RecoveryStartupGateReceipt(
            runtime_id=runtime_id,
            steps=tuple(steps),
            admission=authoritative,
            status=status,
            ready_for_normal_work=ready,
            processed_items=processed,
            max_items=max_items,
        )
