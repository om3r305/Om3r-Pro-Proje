from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping

from .phase69_governed_shadow_execution import GovernedShadowExecution
from .phase71_persisted_runtime_supervisor import PersistedRuntimeStaleError
from .phase92_recovery_worker_session import RecoveryWorkerSession
from .phase94_binance_spot_recovery_evidence import (
    BinanceSpotRecoveryEvidenceProvider,
)
from .phase97_bounded_auto_recovery_drain import (
    BoundedAutoRecoveryDrainReceipt,
    ProviderFactory,
    run_bounded_auto_binance_recovery,
)
from .phase99_recovery_guarded_shadow_handoff import (
    RecoveryGuardedShadowExecutionReceipt,
    RecoveryGuardedShadowHandoff,
)

PHASE100_SCHEMA_VERSION = "brian.phase100-recovery-first-shadow-worker.v1"


class RecoveryFirstShadowWorkerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryFirstShadowStartupReceipt:
    runtime_id: str
    recovery: BoundedAutoRecoveryDrainReceipt
    status: str
    ready_for_normal_shadow: bool
    schema_version: str = PHASE100_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if self.recovery.runtime_id != self.runtime_id:
            raise ValueError("Phase100 recovery runtime mismatch")
        if self.ready_for_normal_shadow:
            if self.status != "READY_FOR_NORMAL_SHADOW":
                raise ValueError(
                    "ready Phase100 startup must use READY_FOR_NORMAL_SHADOW"
                )
            if not self.recovery.ready_for_normal_work:
                raise ValueError(
                    "Phase100 cannot be ready before Phase97 recovery is ready"
                )
            if self.recovery.final_admission.blocked:
                raise ValueError(
                    "Phase100 cannot be ready behind Phase86 barrier"
                )
        elif self.status == "READY_FOR_NORMAL_SHADOW":
            raise ValueError(
                "READY_FOR_NORMAL_SHADOW requires ready_for_normal_shadow"
            )
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase100 startup must remain shadow-only")


RecoveryRunner = Callable[..., BoundedAutoRecoveryDrainReceipt]
HandoffFactory = Callable[..., RecoveryGuardedShadowHandoff]


class RecoveryFirstShadowWorkerSession:
    """Keep recovery and normal shadow execution under one Phase92 session.

    Phase100 is deliberately not a scheduler. It is the lifecycle object a
    backend worker can keep open: acquire/restore once, drain recovery first,
    construct the Phase99 normal-shadow handoff only after Phase97 READY, then
    process governed shadow cycles without changing runtime authority.
    """

    def __init__(
        self,
        *,
        session: RecoveryWorkerSession,
        owns_session: bool = False,
        recovery_runner: RecoveryRunner = run_bounded_auto_binance_recovery,
        handoff_factory: HandoffFactory = RecoveryGuardedShadowHandoff.from_recovery,
    ) -> None:
        if getattr(session, "closed", False):
            raise RecoveryFirstShadowWorkerError(
                "cannot wrap a closed Phase92 recovery worker session"
            )
        if not str(getattr(session, "runtime_id", "")).strip():
            raise RecoveryFirstShadowWorkerError(
                "Phase92 session runtime_id is required"
            )
        supervisor = getattr(session, "runtime_supervisor", None)
        if supervisor is None or not getattr(supervisor, "valid", False):
            raise PersistedRuntimeStaleError(
                "Phase100 requires a valid Phase71 runtime supervisor"
            )
        if not callable(recovery_runner):
            raise TypeError("recovery_runner must be callable")
        if not callable(handoff_factory):
            raise TypeError("handoff_factory must be callable")

        self.session = session
        self._owns_session = bool(owns_session)
        self._recovery_runner = recovery_runner
        self._handoff_factory = handoff_factory
        self._startup: RecoveryFirstShadowStartupReceipt | None = None
        self._handoff: RecoveryGuardedShadowHandoff | None = None
        self._closed = False

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        initial_runtime=None,
        foreign_cycle_aborter=None,
        client=None,
        session_factory=RecoveryWorkerSession.from_env,
        recovery_runner: RecoveryRunner = run_bounded_auto_binance_recovery,
        handoff_factory: HandoffFactory = RecoveryGuardedShadowHandoff.from_recovery,
    ) -> "RecoveryFirstShadowWorkerSession":
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        session = session_factory(
            env=env,
            initial_runtime=initial_runtime,
            foreign_cycle_aborter=foreign_cycle_aborter,
            client=client,
        )
        try:
            return cls(
                session=session,
                owns_session=True,
                recovery_runner=recovery_runner,
                handoff_factory=handoff_factory,
            )
        except Exception:
            try:
                session.close()
            except Exception:
                pass
            raise

    @property
    def runtime_id(self) -> str:
        return self.session.runtime_id

    @property
    def startup(self) -> RecoveryFirstShadowStartupReceipt | None:
        return self._startup

    @property
    def ready_for_normal_shadow(self) -> bool:
        return bool(
            self._startup is not None
            and self._startup.ready_for_normal_shadow
            and self._handoff is not None
            and not self._closed
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def run_recovery_gate(
        self,
        *,
        max_items: int,
        recovery_worker_token: str,
        recovery_claim_seconds: int,
        recovery_ttl_seconds: int,
        source_ref: str,
        provider_factory: ProviderFactory = BinanceSpotRecoveryEvidenceProvider,
        clock,
    ) -> RecoveryFirstShadowStartupReceipt:
        if self._closed:
            raise RecoveryFirstShadowWorkerError(
                "Phase100 worker session is closed"
            )
        if self.ready_for_normal_shadow:
            raise RecoveryFirstShadowWorkerError(
                "recovery gate already released normal shadow work"
            )
        supervisor = self.session.runtime_supervisor
        if not getattr(supervisor, "valid", False):
            raise PersistedRuntimeStaleError(
                "runtime supervisor became stale before Phase100 recovery"
            )

        recovery = self._recovery_runner(
            self.session,
            max_items=max_items,
            recovery_worker_token=recovery_worker_token,
            recovery_claim_seconds=recovery_claim_seconds,
            recovery_ttl_seconds=recovery_ttl_seconds,
            source_ref=source_ref,
            provider_factory=provider_factory,
            clock=clock,
        )
        if recovery.runtime_id != self.runtime_id:
            raise RecoveryFirstShadowWorkerError(
                "Phase97 recovery runtime differs from Phase100 session"
            )

        if recovery.ready_for_normal_work:
            handoff = self._handoff_factory(
                session=self.session,
                recovery=recovery,
            )
            startup = RecoveryFirstShadowStartupReceipt(
                runtime_id=self.runtime_id,
                recovery=recovery,
                status="READY_FOR_NORMAL_SHADOW",
                ready_for_normal_shadow=True,
            )
            self._handoff = handoff
        else:
            startup = RecoveryFirstShadowStartupReceipt(
                runtime_id=self.runtime_id,
                recovery=recovery,
                status=recovery.status,
                ready_for_normal_shadow=False,
            )
            self._handoff = None

        self._startup = startup
        return startup

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
        if self._closed:
            raise RecoveryFirstShadowWorkerError(
                "Phase100 worker session is closed"
            )
        if not self.ready_for_normal_shadow or self._handoff is None:
            raise RecoveryFirstShadowWorkerError(
                "normal shadow work is not released by recovery gate"
            )
        return self._handoff.process_governed_cycle(
            governed,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )

    def close(self) -> bool:
        if self._closed:
            return True
        released = True
        try:
            if self._owns_session:
                released = bool(self.session.close())
        finally:
            self._closed = True
            self._handoff = None
        return released

    def __enter__(self) -> "RecoveryFirstShadowWorkerSession":
        if self._closed:
            raise RecoveryFirstShadowWorkerError(
                "Phase100 worker session is closed"
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
