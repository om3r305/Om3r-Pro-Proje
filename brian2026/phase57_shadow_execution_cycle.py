from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Mapping, Sequence
import json
import math

from .phase45_execution_contract import TripleBarrierPolicy, create_position_executor_action
from .phase46_execution_simulator import (
    OrderBookSnapshot,
    ProbabilisticFillModel,
    SimulatedExecutionReceipt,
    StaticLatencyModel,
    simulate_create_action,
    simulate_order,
)
from .phase54_integrated_shadow_decision import IntegratedShadowDecision
from .phase55_rebalance_execution_intents import (
    PendingReversalOpen,
    RebalanceExecutionInstruction,
    RebalanceExecutionPlan,
    compile_from_integrated_decision,
)
from .phase56_pretrade_risk_engine import (
    InstrumentRiskLimits,
    PreTradeAccountState,
    PreTradeRiskEngine,
    PreTradeRiskPolicy,
    PreTradeRiskReceipt,
    TradingState,
)

PHASE57_SCHEMA_VERSION = "brian.phase57-shadow-execution-cycle.v1"


def _hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExecutionMarketInput:
    reference_price: float
    tick_size: float
    snapshots: tuple[OrderBookSnapshot, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.reference_price) or self.reference_price <= 0:
            raise ValueError("reference_price must be positive")
        if not math.isfinite(self.tick_size) or self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if not self.snapshots:
            raise ValueError("execution market input requires order-book snapshots")


@dataclass(frozen=True, slots=True)
class ShadowExecutionCycleItem:
    instruction_kind: str
    asset_id: str
    risk_receipt: PreTradeRiskReceipt
    execution_receipt: SimulatedExecutionReceipt | None
    pending_reversal: PendingReversalOpen | None
    new_risk_cash_reserved_usd: float
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction_kind": self.instruction_kind,
            "asset_id": self.asset_id,
            "risk_receipt": self.risk_receipt.to_dict(),
            "execution_receipt": (
                None if self.execution_receipt is None else self.execution_receipt.to_dict()
            ),
            "pending_reversal": (
                None if self.pending_reversal is None else asdict(self.pending_reversal)
            ),
            "new_risk_cash_reserved_usd": self.new_risk_cash_reserved_usd,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ShadowExecutionCycle:
    source_plan_id: str
    items: tuple[ShadowExecutionCycleItem, ...]
    initial_available_cash_usd: float
    reserved_new_risk_cash_usd: float
    remaining_unreserved_cash_usd: float
    denied_assets: tuple[str, ...]
    pending_reversal_assets: tuple[str, ...]
    cycle_id: str
    schema_version: str = PHASE57_SCHEMA_VERSION
    account_state_mutated: bool = False
    shadow_only: bool = True
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_plan_id": self.source_plan_id,
            "items": [item.to_dict() for item in self.items],
            "initial_available_cash_usd": self.initial_available_cash_usd,
            "reserved_new_risk_cash_usd": self.reserved_new_risk_cash_usd,
            "remaining_unreserved_cash_usd": self.remaining_unreserved_cash_usd,
            "denied_assets": self.denied_assets,
            "pending_reversal_assets": self.pending_reversal_assets,
            "cycle_id": self.cycle_id,
            "account_state_mutated": self.account_state_mutated,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }


def _market_for(
    asset_id: str,
    markets: Mapping[str, ExecutionMarketInput],
) -> ExecutionMarketInput:
    try:
        return markets[asset_id]
    except KeyError as exc:
        raise KeyError(f"missing execution market input for {asset_id}") from exc


def run_shadow_execution_cycle(
    plan: RebalanceExecutionPlan,
    *,
    equity_usd: float,
    available_cash_usd: float,
    current_weights: Mapping[str, float],
    markets: Mapping[str, ExecutionMarketInput],
    risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
    trading_state: TradingState = "ACTIVE",
    barrier: TripleBarrierPolicy = TripleBarrierPolicy(),
    latency: StaticLatencyModel = StaticLatencyModel(),
    fill_models_by_asset: Mapping[str, ProbabilisticFillModel] | None = None,
) -> ShadowExecutionCycle:
    """Run Phase55 intents through Phase56 risk and Phase45/46 shadow execution.

    Cash for every ALLOWed new-risk leg is reserved immediately for this cycle,
    before fill simulation. A later instruction cannot spend the same dollars.
    Risk reductions never create spendable cash inside this cycle because a
    simulated fill is not authoritative account reconciliation.
    """
    if not math.isfinite(equity_usd) or equity_usd <= 0:
        raise ValueError("equity_usd must be positive")
    if not math.isfinite(available_cash_usd) or available_cash_usd < 0:
        raise ValueError("available_cash_usd must be non-negative")

    weights = {str(asset): float(value) for asset, value in current_weights.items()}
    if any(not math.isfinite(value) for value in weights.values()):
        raise ValueError("current_weights must be finite")

    reserved_cash = 0.0
    items: list[ShadowExecutionCycleItem] = []
    denied: set[str] = set()
    pending_assets: set[str] = set()
    fill_models = fill_models_by_asset or {}

    for instruction in plan.instructions:
        asset = instruction.asset_id
        market = _market_for(asset, markets)
        limits = risk_limits_by_asset.get(asset, InstrumentRiskLimits())
        account = PreTradeAccountState(
            equity_usd=float(equity_usd),
            available_cash_usd=max(0.0, float(available_cash_usd) - reserved_cash),
            open_position_weight=float(weights.get(asset, instruction.current_weight)),
        )
        engine = PreTradeRiskEngine(
            PreTradeRiskPolicy(trading_state=trading_state, limits=limits)
        )

        execution: SimulatedExecutionReceipt | None = None
        cash_reservation = 0.0
        pending = instruction.pending_reversal

        if instruction.reduction_intent is not None:
            risk = engine.review(instruction.reduction_intent, account)
            if risk.allowed:
                reduction = instruction.reduction_intent
                requested_base = risk.requested_notional_usd / market.reference_price
                execution = simulate_order(
                    side="BUY" if reduction.order_direction > 0 else "SELL",
                    order_type="MARKET",
                    requested_base=requested_base,
                    submit_timestamp=reduction.created_at,
                    snapshots=market.snapshots,
                    latency=latency,
                    fill_model=fill_models.get(asset),
                    tick_size=market.tick_size,
                    # Risk reduction is not vetoed because alpha slippage is high;
                    # the simulator records the realized adverse slippage instead.
                    max_slippage_bps=None,
                )
                status = f"REDUCTION_{execution.status}"
            else:
                denied.add(asset)
                status = "RISK_DENIED"
            if pending is not None:
                pending_assets.add(asset)

        elif instruction.trade_intent is not None:
            intent = instruction.trade_intent
            risk = engine.review(intent, account)
            if risk.allowed:
                # Reserve the full requested new-risk notional before simulation.
                # This is conservative if a later fill is partial/no-fill, but prevents
                # two same-cycle orders from both spending the same cash.
                cash_reservation = risk.requested_notional_usd
                reserved_cash += cash_reservation
                action = create_position_executor_action(
                    intent,
                    equity_usd=equity_usd,
                    reference_price=market.reference_price,
                    barrier=barrier,
                )
                execution = simulate_create_action(
                    action,
                    intent,
                    snapshots=market.snapshots,
                    latency=latency,
                    fill_model=fill_models.get(asset),
                    tick_size=market.tick_size,
                )
                status = f"NEW_RISK_{execution.status}"
            else:
                denied.add(asset)
                status = "RISK_DENIED"
        else:
            raise ValueError(f"{asset} instruction has no executable payload")

        items.append(ShadowExecutionCycleItem(
            instruction_kind=instruction.kind,
            asset_id=asset,
            risk_receipt=risk,
            execution_receipt=execution,
            pending_reversal=pending,
            new_risk_cash_reserved_usd=cash_reservation,
            status=status,
        ))

    remaining = max(0.0, float(available_cash_usd) - reserved_cash)
    payload = {
        "schema": PHASE57_SCHEMA_VERSION,
        "source_plan_id": plan.plan_id,
        "items": [item.to_dict() for item in items],
        "initial_available_cash_usd": available_cash_usd,
        "reserved_new_risk_cash_usd": reserved_cash,
        "remaining_unreserved_cash_usd": remaining,
    }
    return ShadowExecutionCycle(
        source_plan_id=plan.plan_id,
        items=tuple(items),
        initial_available_cash_usd=float(available_cash_usd),
        reserved_new_risk_cash_usd=reserved_cash,
        remaining_unreserved_cash_usd=remaining,
        denied_assets=tuple(sorted(denied)),
        pending_reversal_assets=tuple(sorted(pending_assets)),
        cycle_id=_hash(payload),
    )


def run_integrated_shadow_execution_cycle(
    decision: IntegratedShadowDecision,
    *,
    expected_edge_bps_by_asset: Mapping[str, float],
    confidence_by_asset: Mapping[str, float],
    evidence_ids_by_asset: Mapping[str, Sequence[str]],
    max_slippage_bps: float,
    ttl_seconds: int,
    equity_usd: float,
    available_cash_usd: float,
    markets: Mapping[str, ExecutionMarketInput],
    risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
    trading_state: TradingState = "ACTIVE",
    barrier: TripleBarrierPolicy = TripleBarrierPolicy(),
    latency: StaticLatencyModel = StaticLatencyModel(),
    fill_models_by_asset: Mapping[str, ProbabilisticFillModel] | None = None,
) -> ShadowExecutionCycle:
    plan = compile_from_integrated_decision(
        decision,
        expected_edge_bps_by_asset=expected_edge_bps_by_asset,
        confidence_by_asset=confidence_by_asset,
        evidence_ids_by_asset=evidence_ids_by_asset,
        max_slippage_bps=max_slippage_bps,
        ttl_seconds=ttl_seconds,
    )
    return run_shadow_execution_cycle(
        plan,
        equity_usd=equity_usd,
        available_cash_usd=available_cash_usd,
        current_weights=decision.current_weights,
        markets=markets,
        risk_limits_by_asset=risk_limits_by_asset,
        trading_state=trading_state,
        barrier=barrier,
        latency=latency,
        fill_models_by_asset=fill_models_by_asset,
    )
