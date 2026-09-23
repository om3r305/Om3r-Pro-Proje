from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .phase54_integrated_shadow_decision import (
    AssetDecisionInput,
    IntegratedShadowConfig,
    IntegratedShadowDecision,
    run_integrated_shadow_decision,
)
from .phase56_pretrade_risk_engine import InstrumentRiskLimits
from .phase57_shadow_execution_cycle import ExecutionMarketInput
from .phase100_recovery_first_shadow_worker import RecoveryFirstShadowWorkerSession
from .phase101_integrated_decision_shadow_runtime import (
    IntegratedDecisionShadowReceipt,
    IntegratedDecisionShadowRuntime,
)

PHASE102_SCHEMA_VERSION = "brian.phase102-grounded-decision-worker-cycle.v1"


class GroundedDecisionWorkerCycleError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GroundedDecisionWorkerCycleReceipt:
    runtime_id: str
    account_state_id: str
    decision: IntegratedShadowDecision
    execution: IntegratedDecisionShadowReceipt
    status: str
    schema_version: str = PHASE102_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(self.account_state_id) != 64:
            raise ValueError("account_state_id must be a content hash")
        if self.execution.runtime_id != self.runtime_id:
            raise ValueError("Phase102 execution runtime mismatch")
        if self.execution.decision_pipeline_id != self.decision.pipeline_id:
            raise ValueError("Phase102 decision/execution pipeline mismatch")
        if self.status != self.execution.status:
            raise ValueError("Phase102 status must mirror Phase101 execution")
        if not self.decision.shadow_only or self.decision.live_execution:
            raise ValueError("Phase102 decision crossed shadow-only boundary")
        if self.decision.automatic_promotion:
            raise ValueError("Phase102 decision cannot auto-promote")
        if not self.execution.shadow_only or self.execution.live_execution:
            raise ValueError("Phase102 execution crossed shadow-only boundary")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase102 receipt must remain shadow-only")


DecisionRunner = Callable[..., IntegratedShadowDecision]


def _runtime_head(worker: RecoveryFirstShadowWorkerSession):
    session = getattr(worker, "session", None)
    supervisor = getattr(session, "runtime_supervisor", None)
    runtime = getattr(supervisor, "runtime", None)
    ledger = getattr(runtime, "ledger", None)
    head = getattr(ledger, "head_state", None)
    if head is None:
        raise GroundedDecisionWorkerCycleError(
            "Phase100 session does not expose authoritative Phase60 head"
        )
    state_id = str(getattr(head, "state_id", ""))
    if len(state_id) != 64:
        raise GroundedDecisionWorkerCycleError(
            "authoritative Phase60 head state_id is invalid"
        )
    return head


class GroundedDecisionWorkerCycle:
    """Run the real Phase43/44/52/53/54 producer before Phase101 execution.

    Current weights, equity and cash are not caller inputs. They come from the
    exact Phase60 account head owned by the Phase100 session. The head state id
    is rechecked after Phase54 finishes and before Phase101 is allowed to create
    any governed/durable work.
    """

    def __init__(
        self,
        *,
        worker: RecoveryFirstShadowWorkerSession,
        integrated_runtime: IntegratedDecisionShadowRuntime | None = None,
        decision_runner: DecisionRunner = run_integrated_shadow_decision,
    ) -> None:
        if getattr(worker, "closed", False):
            raise GroundedDecisionWorkerCycleError(
                "Phase100 worker session is closed"
            )
        if not getattr(worker, "ready_for_normal_shadow", False):
            raise GroundedDecisionWorkerCycleError(
                "Phase100 recovery gate has not released normal shadow work"
            )
        if not callable(decision_runner):
            raise TypeError("decision_runner must be callable")

        runtime = integrated_runtime or IntegratedDecisionShadowRuntime(worker=worker)
        if runtime.worker is not worker:
            raise GroundedDecisionWorkerCycleError(
                "Phase102 integrated runtime uses a different Phase100 worker"
            )

        self.worker = worker
        self.integrated_runtime = runtime
        self.decision_runner = decision_runner

    @property
    def runtime_id(self) -> str:
        return self.worker.runtime_id

    def process(
        self,
        asset_inputs: Mapping[str, AssetDecisionInput],
        *,
        timestamp: float,
        model_weights: Mapping[str, float],
        returns_by_asset: Mapping[str, Sequence[float]],
        config: IntegratedShadowConfig,
        expected_edge_bps_by_asset: Mapping[str, float],
        max_slippage_bps: float,
        ttl_seconds: int,
        markets: Mapping[str, ExecutionMarketInput],
        risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
        marks: Mapping[str, float],
        worker_token: str,
        claim_seconds: int,
        observed_at: float,
        source_ref: str,
    ) -> GroundedDecisionWorkerCycleReceipt:
        if self.worker.closed:
            raise GroundedDecisionWorkerCycleError(
                "Phase100 worker session closed before Phase102 cycle"
            )
        if not self.worker.ready_for_normal_shadow:
            raise GroundedDecisionWorkerCycleError(
                "normal shadow work is no longer released"
            )
        if not math.isfinite(float(timestamp)):
            raise ValueError("timestamp must be finite")
        if not math.isfinite(float(observed_at)):
            raise ValueError("observed_at must be finite")

        head = _runtime_head(self.worker)
        head_state_id = str(head.state_id)
        if float(timestamp) < float(head.observed_at):
            raise GroundedDecisionWorkerCycleError(
                "decision timestamp predates authoritative Phase60 head"
            )

        current_weights = {
            str(asset): float(weight)
            for asset, weight in head.position_weights
        }
        decision = self.decision_runner(
            asset_inputs,
            timestamp=float(timestamp),
            model_weights=model_weights,
            current_weights=current_weights,
            returns_by_asset=returns_by_asset,
            config=config,
        )
        if not decision.shadow_only or decision.live_execution:
            raise GroundedDecisionWorkerCycleError(
                "Phase54 decision crossed shadow-only boundary"
            )
        if decision.automatic_promotion:
            raise GroundedDecisionWorkerCycleError(
                "Phase54 automatic promotion is forbidden"
            )

        # Phase54 can be CPU-heavy. Preserve the exact account-state identity
        # across that work; never execute a decision built on a head that moved.
        current_head = _runtime_head(self.worker)
        if current_head.state_id != head_state_id:
            raise GroundedDecisionWorkerCycleError(
                "authoritative Phase60 head changed while Phase54 decision ran"
            )

        execution = self.integrated_runtime.process_integrated_decision(
            decision,
            expected_edge_bps_by_asset=expected_edge_bps_by_asset,
            max_slippage_bps=max_slippage_bps,
            ttl_seconds=ttl_seconds,
            equity_usd=float(head.equity_usd),
            available_cash_usd=float(head.available_cash_usd),
            markets=markets,
            risk_limits_by_asset=risk_limits_by_asset,
            marks=marks,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
            observed_at=float(observed_at),
            source_ref=source_ref,
        )
        return GroundedDecisionWorkerCycleReceipt(
            runtime_id=self.runtime_id,
            account_state_id=head_state_id,
            decision=decision,
            execution=execution,
            status=execution.status,
        )
