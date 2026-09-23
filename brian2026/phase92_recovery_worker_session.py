from __future__ import annotations

import os
import uuid
from collections.abc import Mapping

from .phase56_pretrade_risk_engine import InstrumentRiskLimits
from .phase57_shadow_execution_cycle import ExecutionMarketInput
from .phase67_durable_runtime_orchestrator import DurableShadowPaperRuntime
from .phase70_durable_runtime_store import DurableRuntimeStore
from .phase71_persisted_runtime_supervisor import PersistedDurableRuntimeSupervisor
from .phase89_recovery_startup_gate import RecoveryStartupGateReceipt
from .phase90_recovery_runtime_assembly import (
    RecoveryRuntimeStack,
    build_recovery_runtime_stack,
)
from .phase91_supabase_rpc_transport import SupabaseRecoveryRpcTransport

PHASE92_SCHEMA_VERSION = "brian.phase92-recovery-worker-session.v1"


class RecoveryWorkerSessionError(RuntimeError):
    pass


def _positive_int(
    value: object,
    *,
    label: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise RecoveryWorkerSessionError(f"{label} must be integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RecoveryWorkerSessionError(f"{label} must be integer") from exc
    if parsed < minimum:
        raise RecoveryWorkerSessionError(
            f"{label} must be at least {minimum}"
        )
    if maximum is not None and parsed > maximum:
        raise RecoveryWorkerSessionError(
            f"{label} must be at most {maximum}"
        )
    return parsed


class RecoveryWorkerSession:
    """Lease-owned Phase70-91 recovery worker session.

    A session intentionally keeps the runtime lease while the caller evaluates
    the Phase89 startup gate and, if admitted, can continue using the same
    authoritative runtime supervisor. Closing the session releases the
    owner+fence-gated lease and then closes the owned Supabase HTTP transport.
    """

    def __init__(
        self,
        *,
        rpc,
        runtime_store: DurableRuntimeStore,
        runtime_supervisor: PersistedDurableRuntimeSupervisor,
        stack: RecoveryRuntimeStack,
        owns_rpc: bool,
    ) -> None:
        if not callable(rpc):
            raise TypeError("rpc transport must be callable")
        if stack.runtime_supervisor is not runtime_supervisor:
            raise ValueError("Phase92 stack/runtime supervisor wiring drift")
        if runtime_supervisor.store is not runtime_store:
            raise ValueError("Phase92 runtime store wiring drift")
        self.rpc = rpc
        self.runtime_store = runtime_store
        self.runtime_supervisor = runtime_supervisor
        self.stack = stack
        self._owns_rpc = bool(owns_rpc)
        self._closed = False

    @property
    def runtime_id(self) -> str:
        return self.runtime_supervisor.runtime_id

    @property
    def closed(self) -> bool:
        return self._closed

    @classmethod
    def open(
        cls,
        *,
        rpc,
        runtime_id: str,
        owner_token: str,
        lease_seconds: int,
        initial_runtime: DurableShadowPaperRuntime | None = None,
        foreign_cycle_aborter=None,
        owns_rpc: bool = False,
    ) -> "RecoveryWorkerSession":
        if not runtime_id.strip():
            raise ValueError("runtime_id is required")
        if not owner_token.strip():
            raise ValueError("owner_token is required")
        lease_seconds = _positive_int(
            lease_seconds,
            label="lease_seconds",
            minimum=10,
            maximum=300,
        )

        store = DurableRuntimeStore(rpc)
        supervisor: PersistedDurableRuntimeSupervisor | None = None
        try:
            supervisor = PersistedDurableRuntimeSupervisor.acquire(
                store=store,
                runtime_id=runtime_id,
                owner_token=owner_token,
                lease_seconds=lease_seconds,
                initial_runtime=initial_runtime,
            )
            stack = build_recovery_runtime_stack(
                rpc=rpc,
                runtime_supervisor=supervisor,
                foreign_cycle_aborter=foreign_cycle_aborter,
            )
            return cls(
                rpc=rpc,
                runtime_store=store,
                runtime_supervisor=supervisor,
                stack=stack,
                owns_rpc=owns_rpc,
            )
        except Exception:
            # Phase71 now releases any lease acquired before supervisor
            # construction fails. If stack construction failed after a
            # supervisor existed, release its still-owned lease here.
            if supervisor is not None:
                try:
                    supervisor.release()
                except Exception:
                    pass
            if owns_rpc and hasattr(rpc, "close"):
                try:
                    rpc.close()
                except Exception:
                    pass
            raise

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        initial_runtime: DurableShadowPaperRuntime | None = None,
        foreign_cycle_aborter=None,
        client=None,
    ) -> "RecoveryWorkerSession":
        source = os.environ if env is None else env
        runtime_id = source.get("BRIAN_RUNTIME_ID", "").strip()
        if not runtime_id:
            raise RecoveryWorkerSessionError("BRIAN_RUNTIME_ID is required")

        owner_token = source.get("BRIAN_RECOVERY_OWNER_TOKEN", "").strip()
        if not owner_token:
            # Never silently reuse one owner identity across independent
            # processes. A per-process token makes fencing ownership explicit.
            owner_token = f"phase92-{uuid.uuid4().hex}"

        lease_seconds = _positive_int(
            source.get("BRIAN_RUNTIME_LEASE_SECONDS", "60"),
            label="BRIAN_RUNTIME_LEASE_SECONDS",
            minimum=10,
            maximum=300,
        )

        rpc = SupabaseRecoveryRpcTransport.from_env(
            env=source,
            client=client,
        )
        return cls.open(
            rpc=rpc,
            runtime_id=runtime_id,
            owner_token=owner_token,
            lease_seconds=lease_seconds,
            initial_runtime=initial_runtime,
            foreign_cycle_aborter=foreign_cycle_aborter,
            owns_rpc=True,
        )

    def run_startup_gate(
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
        if self._closed:
            raise RecoveryWorkerSessionError("recovery worker session is closed")
        return self.stack.startup_gate.run(
            max_items=max_items,
            recovery_worker_token=recovery_worker_token,
            recovery_claim_seconds=recovery_claim_seconds,
            recovery_markets=recovery_markets,
            recovery_risk_limits_by_asset=recovery_risk_limits_by_asset,
            recovery_ttl_seconds=recovery_ttl_seconds,
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )

    def close(self) -> bool:
        if self._closed:
            return True
        released = False
        release_error: Exception | None = None
        try:
            released = bool(self.runtime_supervisor.release())
        except Exception as exc:
            release_error = exc
        finally:
            if self._owns_rpc and hasattr(self.rpc, "close"):
                try:
                    self.rpc.close()
                except Exception:
                    if release_error is None:
                        raise
            self._closed = True
        if release_error is not None:
            raise release_error
        return released

    def __enter__(self) -> "RecoveryWorkerSession":
        if self._closed:
            raise RecoveryWorkerSessionError("recovery worker session is closed")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def run_recovery_startup_once_from_env(
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
    env: Mapping[str, str] | None = None,
    initial_runtime: DurableShadowPaperRuntime | None = None,
    foreign_cycle_aborter=None,
    client=None,
) -> RecoveryStartupGateReceipt:
    """One-shot backend worker helper that never leaks its runtime lease."""
    with RecoveryWorkerSession.from_env(
        env=env,
        initial_runtime=initial_runtime,
        foreign_cycle_aborter=foreign_cycle_aborter,
        client=client,
    ) as session:
        return session.run_startup_gate(
            max_items=max_items,
            recovery_worker_token=recovery_worker_token,
            recovery_claim_seconds=recovery_claim_seconds,
            recovery_markets=recovery_markets,
            recovery_risk_limits_by_asset=recovery_risk_limits_by_asset,
            recovery_ttl_seconds=recovery_ttl_seconds,
            marks=marks,
            observed_at=observed_at,
            source_ref=source_ref,
        )
