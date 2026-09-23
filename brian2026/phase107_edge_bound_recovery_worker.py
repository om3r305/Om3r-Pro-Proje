from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .phase54_integrated_shadow_decision import (
    AssetDecisionInput,
    IntegratedShadowConfig,
)
from .phase56_pretrade_risk_engine import InstrumentRiskLimits
from .phase57_shadow_execution_cycle import ExecutionMarketInput
from .phase94_binance_spot_recovery_evidence import (
    BinanceSpotRecoveryEvidenceProvider,
)
from .phase97_bounded_auto_recovery_drain import ProviderFactory
from .phase100_recovery_first_shadow_worker import (
    RecoveryFirstShadowStartupReceipt,
    RecoveryFirstShadowWorkerSession,
)
from .phase101_integrated_decision_shadow_runtime import (
    IntegratedDecisionShadowRuntime,
)
from .phase102_grounded_decision_worker_cycle import (
    GroundedDecisionWorkerCycle,
    GroundedDecisionWorkerCycleReceipt,
)
from .phase106_decision_bound_lagged_edge import (
    AssetLaggedEdgeContext,
    DecisionBoundLaggedEdgeRuntime,
)

PHASE107_SCHEMA_VERSION = "brian.phase107-edge-bound-recovery-worker.v1"


class EdgeBoundRecoveryWorkerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PrefetchedLaggedEdgeGroundedCycle:
    bundle_ref: str
    asset_inputs: Mapping[str, AssetDecisionInput]
    timestamp: float
    model_weights: Mapping[str, float]
    returns_by_asset: Mapping[str, Sequence[float]]
    config: IntegratedShadowConfig
    edge_contexts_by_asset: Mapping[str, AssetLaggedEdgeContext]
    max_slippage_bps: float
    ttl_seconds: int
    markets: Mapping[str, ExecutionMarketInput]
    risk_limits_by_asset: Mapping[str, InstrumentRiskLimits]
    marks: Mapping[str, float]
    observed_at: float
    source_ref: str
    schema_version: str = PHASE107_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.bundle_ref.strip():
            raise ValueError("bundle_ref is required")
        if not self.asset_inputs:
            raise ValueError("prefetched lagged-edge cycle requires asset_inputs")
        if not self.source_ref.strip():
            raise ValueError("source_ref is required")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase107 prefetch bundle must remain shadow-only")


@dataclass(frozen=True, slots=True)
class EdgeBoundRecoveryWorkerReceipt:
    runtime_id: str
    startup: RecoveryFirstShadowStartupReceipt
    status: str
    prefetched: bool
    bundle_ref: str | None
    cycle: GroundedDecisionWorkerCycleReceipt | None
    schema_version: str = PHASE107_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if self.startup.runtime_id != self.runtime_id:
            raise ValueError("Phase107 startup runtime mismatch")
        if self.prefetched:
            if not self.bundle_ref or self.cycle is None:
                raise ValueError("prefetched Phase107 receipt requires bundle/cycle")
            if self.status != self.cycle.status:
                raise ValueError("Phase107 status must mirror Phase102 cycle")
        else:
            if self.bundle_ref is not None or self.cycle is not None:
                raise ValueError("blocked Phase107 receipt cannot carry cycle data")
            if self.startup.ready_for_normal_shadow:
                raise ValueError("ready Phase107 startup must continue to prefetch")
            if self.status != self.startup.status:
                raise ValueError("blocked Phase107 status must mirror startup")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase107 receipt must remain shadow-only")


PrefetchProvider = Callable[[], PrefetchedLaggedEdgeGroundedCycle]
BaseRuntimeFactory = Callable[..., IntegratedDecisionShadowRuntime]
CycleFactory = Callable[..., GroundedDecisionWorkerCycle]


class EdgeBoundRecoveryWorker:
    """Recovery-first backend worker with Phase105/106 edge authority.

    No caller-supplied expected-edge numbers enter Phase102. Once recovery is
    READY, the fully-prefetched bundle supplies lagged PIT reliability and
    decision-time costs. Phase106 binds those inputs to the *actual* Phase54
    support groups and blocks only new/increasing risk that lacks eligible edge;
    reduce-only risk remains available through Phase55.
    """

    def __init__(
        self,
        *,
        worker: RecoveryFirstShadowWorkerSession,
        owns_worker: bool = False,
        base_runtime_factory: BaseRuntimeFactory = IntegratedDecisionShadowRuntime,
        cycle_factory: CycleFactory = GroundedDecisionWorkerCycle,
    ) -> None:
        if getattr(worker, "closed", False):
            raise EdgeBoundRecoveryWorkerError("Phase100 worker session is closed")
        if not callable(base_runtime_factory):
            raise TypeError("base_runtime_factory must be callable")
        if not callable(cycle_factory):
            raise TypeError("cycle_factory must be callable")
        self.worker = worker
        self._owns_worker = bool(owns_worker)
        self._base_runtime_factory = base_runtime_factory
        self._cycle_factory = cycle_factory
        self._closed = False

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        initial_runtime=None,
        foreign_cycle_aborter=None,
        client=None,
        worker_factory=RecoveryFirstShadowWorkerSession.from_env,
        base_runtime_factory: BaseRuntimeFactory = IntegratedDecisionShadowRuntime,
        cycle_factory: CycleFactory = GroundedDecisionWorkerCycle,
    ) -> "EdgeBoundRecoveryWorker":
        if not callable(worker_factory):
            raise TypeError("worker_factory must be callable")
        worker = worker_factory(
            env=env,
            initial_runtime=initial_runtime,
            foreign_cycle_aborter=foreign_cycle_aborter,
            client=client,
        )
        try:
            return cls(
                worker=worker,
                owns_worker=True,
                base_runtime_factory=base_runtime_factory,
                cycle_factory=cycle_factory,
            )
        except Exception:
            try:
                worker.close()
            except Exception:
                pass
            raise

    @property
    def runtime_id(self) -> str:
        return self.worker.runtime_id

    @property
    def closed(self) -> bool:
        return self._closed

    def run_once(
        self,
        *,
        prefetch_provider: PrefetchProvider,
        recovery_max_items: int,
        recovery_worker_token: str,
        recovery_claim_seconds: int,
        recovery_ttl_seconds: int,
        recovery_source_ref: str,
        normal_worker_token: str,
        normal_claim_seconds: int,
        recovery_provider_factory: ProviderFactory = BinanceSpotRecoveryEvidenceProvider,
        clock=time.time,
    ) -> EdgeBoundRecoveryWorkerReceipt:
        if self._closed:
            raise EdgeBoundRecoveryWorkerError("Phase107 worker is closed")
        if not callable(prefetch_provider):
            raise TypeError("prefetch_provider must be callable")
        if not normal_worker_token.strip():
            raise ValueError("normal_worker_token is required")
        if normal_claim_seconds < 10 or normal_claim_seconds > 300:
            raise ValueError("normal_claim_seconds must be in [10,300]")

        if self.worker.ready_for_normal_shadow:
            startup = self.worker.startup
            if startup is None:
                raise EdgeBoundRecoveryWorkerError(
                    "Phase100 ready state is missing startup receipt"
                )
        else:
            startup = self.worker.run_recovery_gate(
                max_items=recovery_max_items,
                recovery_worker_token=recovery_worker_token,
                recovery_claim_seconds=recovery_claim_seconds,
                recovery_ttl_seconds=recovery_ttl_seconds,
                source_ref=recovery_source_ref,
                provider_factory=recovery_provider_factory,
                clock=clock,
            )

        if not startup.ready_for_normal_shadow:
            return EdgeBoundRecoveryWorkerReceipt(
                runtime_id=self.runtime_id,
                startup=startup,
                status=startup.status,
                prefetched=False,
                bundle_ref=None,
                cycle=None,
            )

        bundle = prefetch_provider()
        if not isinstance(bundle, PrefetchedLaggedEdgeGroundedCycle):
            raise EdgeBoundRecoveryWorkerError(
                "prefetch_provider returned invalid Phase107 bundle type"
            )
        if not bundle.shadow_only or bundle.live_execution:
            raise EdgeBoundRecoveryWorkerError(
                "prefetch bundle crossed shadow-only boundary"
            )

        base_runtime = self._base_runtime_factory(worker=self.worker)
        edge_runtime = DecisionBoundLaggedEdgeRuntime(
            base_runtime=base_runtime,
            contexts_by_asset=bundle.edge_contexts_by_asset,
        )
        cycle_runner = self._cycle_factory(
            worker=self.worker,
            integrated_runtime=edge_runtime,
        )
        cycle = cycle_runner.process(
            bundle.asset_inputs,
            timestamp=bundle.timestamp,
            model_weights=bundle.model_weights,
            returns_by_asset=bundle.returns_by_asset,
            config=bundle.config,
            expected_edge_bps_by_asset={},
            max_slippage_bps=bundle.max_slippage_bps,
            ttl_seconds=bundle.ttl_seconds,
            markets=bundle.markets,
            risk_limits_by_asset=bundle.risk_limits_by_asset,
            marks=bundle.marks,
            worker_token=normal_worker_token,
            claim_seconds=normal_claim_seconds,
            observed_at=bundle.observed_at,
            source_ref=bundle.source_ref,
        )
        return EdgeBoundRecoveryWorkerReceipt(
            runtime_id=self.runtime_id,
            startup=startup,
            status=cycle.status,
            prefetched=True,
            bundle_ref=bundle.bundle_ref,
            cycle=cycle,
        )

    def close(self) -> bool:
        if self._closed:
            return True
        released = True
        try:
            if self._owns_worker:
                released = bool(self.worker.close())
        finally:
            self._closed = True
        return released

    def __enter__(self) -> "EdgeBoundRecoveryWorker":
        if self._closed:
            raise EdgeBoundRecoveryWorkerError("Phase107 worker is closed")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
