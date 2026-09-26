from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from .phase71_persisted_runtime_supervisor import PersistedRuntimeStaleError
from .phase86_recovery_admission_interlock import RecoveryAdmissionState
from .phase95_auto_binance_recovery_worker import (
    AutoRecoveryStartupReceipt,
    ProviderFactory,
    run_one_auto_binance_recovery,
)

PHASE97_SCHEMA_VERSION = "brian.phase97-bounded-auto-recovery-drain.v1"

_RETRYABLE_GATE_STATUSES = frozenset({
    "RECOVERY_BUDGET_EXHAUSTED",
    "RECOVERY_BACKLOG_REMAINS",
    "RECOVERY_BARRIER_APPEARED",
})


class BoundedAutoRecoveryDrainError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BoundedAutoRecoveryDrainReceipt:
    runtime_id: str
    attempts: tuple[AutoRecoveryStartupReceipt, ...]
    final_admission: RecoveryAdmissionState
    status: str
    ready_for_normal_work: bool
    processed_items: int
    max_items: int
    schema_version: str = PHASE97_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if self.max_items <= 0 or self.max_items > 32:
            raise ValueError("max_items must be in [1,32]")
        if not self.attempts or len(self.attempts) > self.max_items:
            raise ValueError("attempt count must be in [1,max_items]")
        if self.processed_items < 0 or self.processed_items > self.max_items:
            raise ValueError("processed_items outside bounded drain budget")
        if self.final_admission.runtime_id != self.runtime_id:
            raise ValueError("final admission runtime mismatch")
        for attempt in self.attempts:
            if attempt.gate.runtime_id != self.runtime_id:
                raise ValueError("attempt runtime mismatch")
            if not attempt.shadow_only or attempt.live_execution:
                raise ValueError("attempt crossed shadow-only boundary")
        if self.ready_for_normal_work:
            if self.status != "READY_FOR_NORMAL_WORK":
                raise ValueError("ready drain must use READY_FOR_NORMAL_WORK")
            if self.final_admission.blocked:
                raise ValueError("ready drain cannot end behind recovery barrier")
            if not self.attempts[-1].gate.ready_for_normal_work:
                raise ValueError("ready drain requires a ready final startup gate")
        elif self.status == "READY_FOR_NORMAL_WORK":
            raise ValueError("READY_FOR_NORMAL_WORK requires ready=true")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase97 drain must remain shadow-only")


ItemRunner = Callable[..., AutoRecoveryStartupReceipt]


def _positive_int(value: object, *, label: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be integer") from exc
    if parsed <= 0 or parsed > maximum:
        raise ValueError(f"{label} must be in [1,{maximum}]")
    return parsed


def _receipt(
    *,
    runtime_id: str,
    attempts: list[AutoRecoveryStartupReceipt],
    final_admission: RecoveryAdmissionState,
    status: str,
    ready: bool,
    max_items: int,
) -> BoundedAutoRecoveryDrainReceipt:
    return BoundedAutoRecoveryDrainReceipt(
        runtime_id=runtime_id,
        attempts=tuple(attempts),
        final_admission=final_admission,
        status=status,
        ready_for_normal_work=ready,
        processed_items=sum(
            attempt.gate.processed_items
            for attempt in attempts
        ),
        max_items=max_items,
    )


def run_bounded_auto_binance_recovery(
    session,
    *,
    max_items: int,
    recovery_worker_token: str,
    recovery_claim_seconds: int,
    recovery_ttl_seconds: int,
    source_ref: str,
    provider_factory: ProviderFactory,
    clock,
    item_runner: ItemRunner = run_one_auto_binance_recovery,
) -> BoundedAutoRecoveryDrainReceipt:
    """Drain multiple recovery identities without reusing market evidence.

    Every item is delegated to Phase95 separately, so each recovery identity gets
    its own decision timestamp and fresh Phase94 evidence. The same Phase92
    session and Phase70 lease stay owned across the bounded drain.

    A Phase89 READY result is rechecked through Phase86 immediately before the
    drain returns. If a new barrier appears in that handoff window, the worker
    loops again while budget remains. Phase86's authorization/dispatch wrappers
    remain the database-level final interlock for any later normal cycle.
    """
    if getattr(session, "closed", False):
        raise BoundedAutoRecoveryDrainError("recovery worker session is closed")
    runtime_id = str(getattr(session, "runtime_id", "")).strip()
    if not runtime_id:
        raise BoundedAutoRecoveryDrainError("session runtime_id is required")
    max_items = _positive_int(max_items, label="max_items", maximum=32)
    if not str(recovery_worker_token).strip():
        raise ValueError("recovery_worker_token is required")
    if recovery_claim_seconds <= 0 or recovery_ttl_seconds <= 0:
        raise ValueError("recovery claim/intent TTL must be positive")
    if not str(source_ref).strip():
        raise ValueError("source_ref is required")
    if not callable(provider_factory):
        raise TypeError("provider_factory must be callable")
    if not callable(clock):
        raise TypeError("clock must be callable")
    if not callable(item_runner):
        raise TypeError("item_runner must be callable")

    attempts: list[AutoRecoveryStartupReceipt] = []
    final_admission: RecoveryAdmissionState | None = None

    for _ in range(max_items):
        supervisor = getattr(session, "runtime_supervisor", None)
        if supervisor is not None and not getattr(supervisor, "valid", False):
            raise PersistedRuntimeStaleError(
                "runtime supervisor became stale during Phase97 recovery drain"
            )

        attempt = item_runner(
            session,
            recovery_worker_token=recovery_worker_token,
            recovery_claim_seconds=recovery_claim_seconds,
            recovery_ttl_seconds=recovery_ttl_seconds,
            source_ref=source_ref,
            provider_factory=provider_factory,
            clock=clock,
        )
        if attempt.gate.runtime_id != runtime_id:
            raise BoundedAutoRecoveryDrainError(
                "Phase95 attempt runtime differs from Phase92 session"
            )
        attempts.append(attempt)
        gate = attempt.gate
        final_admission = gate.admission

        if gate.ready_for_normal_work:
            # Close the Phase87-IDLE/Phase86-OPEN handoff window once more while
            # keeping the same runtime lease. A newly-arrived AFTER_START barrier
            # is processed as another fresh-evidence Phase95 item when possible.
            admission_store = getattr(getattr(session, "stack", None), "admission", None)
            if admission_store is None or not hasattr(admission_store, "read"):
                raise BoundedAutoRecoveryDrainError(
                    "session stack must expose Phase86 admission reader"
                )
            final_admission = admission_store.read(runtime_id=runtime_id)
            if not final_admission.blocked:
                return _receipt(
                    runtime_id=runtime_id,
                    attempts=attempts,
                    final_admission=final_admission,
                    status="READY_FOR_NORMAL_WORK",
                    ready=True,
                    max_items=max_items,
                )
            # The gate was ready, but Phase86 observed a new barrier immediately
            # afterwards. Do not hand off normal work; spend another bounded item
            # attempt so Phase95 re-reads identity and fetches fresh evidence.
            continue

        if gate.status not in _RETRYABLE_GATE_STATUSES:
            return _receipt(
                runtime_id=runtime_id,
                attempts=attempts,
                final_admission=final_admission,
                status=gate.status,
                ready=False,
                max_items=max_items,
            )

    assert final_admission is not None
    return _receipt(
        runtime_id=runtime_id,
        attempts=attempts,
        final_admission=final_admission,
        status="RECOVERY_DRAIN_BUDGET_EXHAUSTED",
        ready=False,
        max_items=max_items,
    )
