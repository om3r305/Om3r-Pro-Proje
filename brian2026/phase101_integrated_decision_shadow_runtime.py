from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .phase54_integrated_shadow_decision import IntegratedShadowDecision
from .phase56_pretrade_risk_engine import InstrumentRiskLimits
from .phase57_shadow_execution_cycle import ExecutionMarketInput
from .phase69_governed_shadow_execution import (
    GovernedShadowExecution,
    run_governed_integrated_shadow_execution_cycle,
)
from .phase73_operational_risk_store import (
    OperationalRiskStore,
    StoredOperationalRiskLedger,
)
from .phase99_recovery_guarded_shadow_handoff import (
    RecoveryGuardedShadowExecutionReceipt,
)
from .phase100_recovery_first_shadow_worker import (
    RecoveryFirstShadowWorkerSession,
)

PHASE101_SCHEMA_VERSION = "brian.phase101-integrated-decision-shadow-runtime.v1"

_WAIT_STATUSES = frozenset({
    "WAIT_NO_GROUNDED_SIGNALS",
    "HOLD_CURRENT_BOOK",
})


class IntegratedDecisionShadowRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IntegratedDecisionShadowReceipt:
    runtime_id: str
    decision_pipeline_id: str
    decision_status: str
    status: str
    executed: bool
    risk_version: int | None
    risk_receipt_id: str | None
    governed_result_id: str | None
    execution: RecoveryGuardedShadowExecutionReceipt | None
    schema_version: str = PHASE101_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if len(self.decision_pipeline_id) != 64:
            raise ValueError("decision_pipeline_id must be a content hash")
        if self.executed:
            if self.status != "SHADOW_EXECUTED":
                raise ValueError("executed Phase101 receipt must use SHADOW_EXECUTED")
            if self.risk_version is None or self.risk_version <= 0:
                raise ValueError("executed Phase101 receipt requires risk_version")
            if self.risk_receipt_id is None or len(self.risk_receipt_id) != 64:
                raise ValueError("executed Phase101 receipt requires risk receipt id")
            if self.governed_result_id is None or len(self.governed_result_id) != 64:
                raise ValueError("executed Phase101 receipt requires governed result id")
            if self.execution is None:
                raise ValueError("executed Phase101 receipt requires execution receipt")
        else:
            if self.status == "SHADOW_EXECUTED":
                raise ValueError("SHADOW_EXECUTED requires executed=true")
            if self.execution is not None:
                raise ValueError("non-executed Phase101 receipt cannot carry execution")
        if self.execution is not None:
            if not self.execution.shadow_only or self.execution.live_execution:
                raise ValueError("Phase101 execution crossed shadow-only boundary")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase101 receipt must remain shadow-only")


GovernedRunner = Callable[..., GovernedShadowExecution]


def _finite_mapping(
    values: Mapping[str, float],
    *,
    label: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_asset, raw_value in values.items():
        asset = str(raw_asset).strip()
        if not asset:
            raise ValueError(f"{label} contains blank asset")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{label}[{asset}] must be finite")
        result[asset] = value
    return result


def _grounded_execution_metadata(
    decision: IntegratedShadowDecision,
) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
    """Derive execution confidence/evidence from Phase54's grounded outputs.

    Phase101 deliberately does not let a caller inject confidence or evidence
    lineage independently from the Phase54 decision. Expected edge remains an
    explicit external input because Phase54 has no bps-return estimate.
    """
    confidence: dict[str, float] = {}
    if decision.portfolio_book is not None:
        for asset, conviction in decision.portfolio_book.blend.convictions.items():
            value = abs(float(conviction))
            if not math.isfinite(value) or value < 0 or value > 1:
                raise IntegratedDecisionShadowRuntimeError(
                    f"Phase54 conviction for {asset} is outside [0,1]"
                )
            confidence[str(asset)] = value

    evidence: dict[str, tuple[str, ...]] = {}
    for asset, result in decision.asset_results.items():
        ids = {
            str(evidence_id)
            for claim in result.analyst_claims
            for evidence_id in claim.support_evidence_ids
            if str(evidence_id)
        }
        evidence[str(asset)] = tuple(sorted(ids))

    return confidence, evidence


def _persisted_risk_head(
    stored: StoredOperationalRiskLedger | None,
):
    if stored is None:
        raise IntegratedDecisionShadowRuntimeError(
            "persisted operational-risk head is required before execution"
        )
    entries = stored.ledger.entries
    if not entries or stored.head_entry_id is None:
        raise IntegratedDecisionShadowRuntimeError(
            "persisted operational-risk ledger has no head receipt"
        )
    entry = entries[-1]
    receipt = entry.receipt
    if entry.entry_id != stored.head_entry_id:
        raise IntegratedDecisionShadowRuntimeError(
            "persisted operational-risk head entry identity drift"
        )
    if not receipt.verify_identity():
        raise IntegratedDecisionShadowRuntimeError(
            "persisted operational-risk receipt identity mismatch"
        )
    if receipt.trading_state != stored.current_state:
        raise IntegratedDecisionShadowRuntimeError(
            "persisted operational-risk state differs from head receipt"
        )
    return receipt


class IntegratedDecisionShadowRuntime:
    """Phase54 -> persisted Phase68/73 -> Phase69 -> Phase100 execution bridge.

    The Phase54 decision remains the source of portfolio weights, confidence and
    evidence lineage. The current Phase73 head supplies operational risk. Phase75
    re-loads that same persisted risk head during authorization, so a risk change
    between Phase101 compilation and durable write-ahead fails closed instead of
    authorizing a stale governed cycle.
    """

    def __init__(
        self,
        *,
        worker: RecoveryFirstShadowWorkerSession,
        risk_store: OperationalRiskStore | None = None,
        governed_runner: GovernedRunner = run_governed_integrated_shadow_execution_cycle,
    ) -> None:
        if getattr(worker, "closed", False):
            raise IntegratedDecisionShadowRuntimeError(
                "Phase100 worker session is closed"
            )
        if not getattr(worker, "ready_for_normal_shadow", False):
            raise IntegratedDecisionShadowRuntimeError(
                "Phase100 recovery gate has not released normal shadow work"
            )
        rpc = getattr(getattr(worker, "session", None), "rpc", None)
        if risk_store is None:
            if not callable(rpc):
                raise IntegratedDecisionShadowRuntimeError(
                    "Phase100 session must expose callable RPC transport"
                )
            risk_store = OperationalRiskStore(rpc)
        if not callable(governed_runner):
            raise TypeError("governed_runner must be callable")

        self.worker = worker
        self.risk_store = risk_store
        self.governed_runner = governed_runner

    @property
    def runtime_id(self) -> str:
        return self.worker.runtime_id

    def process_integrated_decision(
        self,
        decision: IntegratedShadowDecision,
        *,
        expected_edge_bps_by_asset: Mapping[str, float],
        max_slippage_bps: float,
        ttl_seconds: int,
        equity_usd: float,
        available_cash_usd: float,
        markets: Mapping[str, ExecutionMarketInput],
        risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
        worker_token: str,
        claim_seconds: int,
        observed_at: float,
        source_ref: str,
    ) -> IntegratedDecisionShadowReceipt:
        if self.worker.closed:
            raise IntegratedDecisionShadowRuntimeError(
                "Phase100 worker session closed before integrated decision"
            )
        if not self.worker.ready_for_normal_shadow:
            raise IntegratedDecisionShadowRuntimeError(
                "normal shadow work is no longer released"
            )
        if not decision.shadow_only or decision.live_execution:
            raise IntegratedDecisionShadowRuntimeError(
                "Phase54 decision crossed shadow-only boundary"
            )
        if decision.automatic_promotion:
            raise IntegratedDecisionShadowRuntimeError(
                "Phase54 automatic promotion is forbidden in Phase101"
            )
        if len(decision.pipeline_id) != 64:
            raise IntegratedDecisionShadowRuntimeError(
                "Phase54 pipeline_id must be a content hash"
            )
        if not math.isfinite(float(decision.timestamp)):
            raise ValueError("decision timestamp must be finite")
        if not math.isfinite(float(observed_at)):
            raise ValueError("observed_at must be finite")
        if float(observed_at) < float(decision.timestamp):
            raise ValueError("execution observation cannot precede Phase54 decision")
        if not worker_token.strip():
            raise ValueError("worker_token is required")
        if claim_seconds < 10 or claim_seconds > 300:
            raise ValueError("claim_seconds must be in [10,300]")
        if not source_ref.strip():
            raise ValueError("source_ref is required")
        if not math.isfinite(float(max_slippage_bps)) or max_slippage_bps < 0:
            raise ValueError("max_slippage_bps must be finite and non-negative")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        if decision.status in _WAIT_STATUSES:
            return IntegratedDecisionShadowReceipt(
                runtime_id=self.runtime_id,
                decision_pipeline_id=decision.pipeline_id,
                decision_status=decision.status,
                status=decision.status,
                executed=False,
                risk_version=None,
                risk_receipt_id=None,
                governed_result_id=None,
                execution=None,
            )
        if decision.status != "REBALANCE_PLANNED":
            raise IntegratedDecisionShadowRuntimeError(
                f"unsupported Phase54 decision status {decision.status}"
            )

        clean_edges = _finite_mapping(
            expected_edge_bps_by_asset,
            label="expected_edge_bps_by_asset",
        )
        confidence, evidence = _grounded_execution_metadata(decision)

        stored = self.risk_store.load(runtime_id=self.runtime_id)
        risk = _persisted_risk_head(stored)
        assert stored is not None

        governed = self.governed_runner(
            decision,
            operational_risk=risk,
            expected_edge_bps_by_asset=clean_edges,
            confidence_by_asset=confidence,
            evidence_ids_by_asset=evidence,
            max_slippage_bps=float(max_slippage_bps),
            ttl_seconds=int(ttl_seconds),
            equity_usd=float(equity_usd),
            available_cash_usd=float(available_cash_usd),
            markets=markets,
            risk_limits_by_asset=risk_limits_by_asset,
        )
        if not governed.shadow_only or governed.live_execution:
            raise IntegratedDecisionShadowRuntimeError(
                "Phase69 governed result crossed shadow-only boundary"
            )
        if governed.operational_risk_receipt_id != risk.receipt_id:
            raise IntegratedDecisionShadowRuntimeError(
                "Phase69 governed result is not bound to persisted risk head"
            )

        # A Phase54 rebalance can collapse to zero executable legs after exact
        # delta compilation. Do not create durable execution state for a no-op.
        if not governed.cycle.items:
            return IntegratedDecisionShadowReceipt(
                runtime_id=self.runtime_id,
                decision_pipeline_id=decision.pipeline_id,
                decision_status=decision.status,
                status="NO_EXECUTABLE_INSTRUCTIONS",
                executed=False,
                risk_version=stored.version,
                risk_receipt_id=risk.receipt_id,
                governed_result_id=governed.result_id,
                execution=None,
            )

        execution = self.worker.process_governed_cycle(
            governed,
            worker_token=worker_token,
            claim_seconds=claim_seconds,
            marks={
                str(asset): float(market.reference_price)
                for asset, market in markets.items()
            },
            observed_at=float(observed_at),
            source_ref=source_ref,
        )
        return IntegratedDecisionShadowReceipt(
            runtime_id=self.runtime_id,
            decision_pipeline_id=decision.pipeline_id,
            decision_status=decision.status,
            status="SHADOW_EXECUTED",
            executed=True,
            risk_version=stored.version,
            risk_receipt_id=risk.receipt_id,
            governed_result_id=governed.result_id,
            execution=execution,
        )
