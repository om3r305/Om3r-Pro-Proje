from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .evidence_ledger import content_hash
from .phase45_execution_contract import TripleBarrierPolicy
from .phase46_execution_simulator import ProbabilisticFillModel, StaticLatencyModel
from .phase54_integrated_shadow_decision import IntegratedShadowDecision
from .phase55_rebalance_execution_intents import (
    RebalanceExecutionPlan,
    compile_from_integrated_decision,
)
from .phase56_pretrade_risk_engine import InstrumentRiskLimits, PreTradeRiskPolicy
from .phase57_shadow_execution_cycle import (
    ExecutionMarketInput,
    ShadowExecutionCycle,
    run_shadow_execution_cycle,
)
from .phase68_operational_risk_governor import OperationalRiskReceipt

PHASE69_SCHEMA_VERSION = "brian.phase69-governed-shadow-execution.v1"


@dataclass(frozen=True, slots=True)
class GovernedShadowExecution:
    operational_risk_receipt_id: str
    trading_state: str
    blocked_new_risk_assets: tuple[str, ...]
    policy_fingerprint: str
    cycle: ShadowExecutionCycle
    result_id: str
    schema_version: str = PHASE69_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operational_risk_receipt_id": self.operational_risk_receipt_id,
            "trading_state": self.trading_state,
            "blocked_new_risk_assets": self.blocked_new_risk_assets,
            "policy_fingerprint": self.policy_fingerprint,
            "cycle": self.cycle.to_dict(),
            "result_id": self.result_id,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }


def _policies_for_plan(
    plan: RebalanceExecutionPlan,
    operational_risk: OperationalRiskReceipt,
    risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
) -> dict[str, PreTradeRiskPolicy]:
    assets = tuple(sorted({instruction.asset_id for instruction in plan.instructions}))
    return {
        asset: operational_risk.pretrade_policy_for_asset(
            asset,
            risk_limits_by_asset.get(asset, InstrumentRiskLimits()),
        )
        for asset in assets
    }


def run_governed_shadow_execution_cycle(
    plan: RebalanceExecutionPlan,
    *,
    operational_risk: OperationalRiskReceipt,
    equity_usd: float,
    available_cash_usd: float,
    current_weights: Mapping[str, float],
    markets: Mapping[str, ExecutionMarketInput],
    risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
    barrier: TripleBarrierPolicy = TripleBarrierPolicy(),
    latency: StaticLatencyModel = StaticLatencyModel(),
    fill_models_by_asset: Mapping[str, ProbabilisticFillModel] | None = None,
    blocked_new_risk_assets: Sequence[str] = (),
) -> GovernedShadowExecution:
    """Run the real Phase 57 execution path under the Phase 68 governor.

    Global ACTIVE/REDUCING/HALTED state and per-asset cooldown locks are
    translated into real Phase 56 policies. Cooldowns block only new/increased
    risk for that asset; valid reduce-only instructions still pass while the
    global state is ACTIVE or REDUCING.
    """
    if not operational_risk.shadow_only or operational_risk.live_execution:
        raise ValueError("operational risk receipt crossed the shadow boundary")

    policies = _policies_for_plan(
        plan,
        operational_risk,
        risk_limits_by_asset,
    )
    policy_payload = {
        asset: asdict(policy)
        for asset, policy in sorted(policies.items())
    }
    policy_fingerprint = content_hash({
        "schema_version": PHASE69_SCHEMA_VERSION,
        "operational_risk_receipt_id": operational_risk.receipt_id,
        "policies": policy_payload,
    })

    cycle = run_shadow_execution_cycle(
        plan,
        equity_usd=equity_usd,
        available_cash_usd=available_cash_usd,
        current_weights=current_weights,
        markets=markets,
        risk_limits_by_asset=risk_limits_by_asset,
        trading_state=operational_risk.trading_state,
        risk_policy_by_asset=policies,
        barrier=barrier,
        latency=latency,
        fill_models_by_asset=fill_models_by_asset,
    )
    blocked = tuple(sorted(
        asset
        for asset, policy in policies.items()
        if policy.block_new_risk
    ))
    result_payload = {
        "schema_version": PHASE69_SCHEMA_VERSION,
        "operational_risk_receipt_id": operational_risk.receipt_id,
        "trading_state": operational_risk.trading_state,
        "blocked_new_risk_assets": blocked,
        "policy_fingerprint": policy_fingerprint,
        "cycle_id": cycle.cycle_id,
    }
    return GovernedShadowExecution(
        operational_risk_receipt_id=operational_risk.receipt_id,
        trading_state=operational_risk.trading_state,
        blocked_new_risk_assets=blocked,
        policy_fingerprint=policy_fingerprint,
        cycle=cycle,
        result_id=content_hash(result_payload),
    )


def run_governed_integrated_shadow_execution_cycle(
    decision: IntegratedShadowDecision,
    *,
    operational_risk: OperationalRiskReceipt,
    expected_edge_bps_by_asset: Mapping[str, float],
    confidence_by_asset: Mapping[str, float],
    evidence_ids_by_asset: Mapping[str, Sequence[str]],
    max_slippage_bps: float,
    ttl_seconds: int,
    equity_usd: float,
    available_cash_usd: float,
    markets: Mapping[str, ExecutionMarketInput],
    risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
    barrier: TripleBarrierPolicy = TripleBarrierPolicy(),
    latency: StaticLatencyModel = StaticLatencyModel(),
    fill_models_by_asset: Mapping[str, ProbabilisticFillModel] | None = None,
) -> GovernedShadowExecution:
    plan = compile_from_integrated_decision(
        decision,
        expected_edge_bps_by_asset=expected_edge_bps_by_asset,
        confidence_by_asset=confidence_by_asset,
        evidence_ids_by_asset=evidence_ids_by_asset,
        max_slippage_bps=max_slippage_bps,
        ttl_seconds=ttl_seconds,
        blocked_new_risk_assets=blocked_new_risk_assets,
    )
    return run_governed_shadow_execution_cycle(
        plan,
        operational_risk=operational_risk,
        equity_usd=equity_usd,
        available_cash_usd=available_cash_usd,
        current_weights=decision.current_weights,
        markets=markets,
        risk_limits_by_asset=risk_limits_by_asset,
        barrier=barrier,
        latency=latency,
        fill_models_by_asset=fill_models_by_asset,
    )
