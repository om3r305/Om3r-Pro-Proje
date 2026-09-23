from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Literal, Mapping, Sequence
import json
import math

from .phase45_execution_contract import TradeIntent
from .phase53_turnover_rebalance import RebalanceLeg, TurnoverPlan
from .phase54_integrated_shadow_decision import IntegratedShadowDecision

PHASE55_SCHEMA_VERSION = "brian.phase55-rebalance-execution-intents.v1"
InstructionKind = Literal[
    "REDUCE",
    "CLOSE",
    "OPEN",
    "INCREASE",
    "REVERSAL_CLOSE",
    "REVERSAL_OPEN_PENDING",
]


def _hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(text.encode("utf-8")).hexdigest()


def _sign(value: float, *, eps: float = 1e-12) -> int:
    return 1 if value > eps else -1 if value < -eps else 0


@dataclass(frozen=True, slots=True)
class RiskReductionIntent:
    intent_id: str
    asset_id: str
    current_direction: int
    order_direction: int
    reduce_weight: float
    current_weight: float
    resulting_weight: float
    reason: str
    created_at: float
    ttl_seconds: int
    reduce_only: bool = True
    shadow_only: bool = True
    live_execution: bool = False
    schema_version: str = PHASE55_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.intent_id.strip() or not self.asset_id.strip() or not self.reason.strip():
            raise ValueError("risk-reduction intent identity/reason is required")
        if self.current_direction not in (-1, 1):
            raise ValueError("current_direction must be -1 or 1")
        if self.order_direction not in (-1, 1):
            raise ValueError("order_direction must be -1 or 1")
        if self.order_direction != -self.current_direction:
            raise ValueError("reduce-only order direction must oppose current position")
        if not math.isfinite(self.reduce_weight) or self.reduce_weight <= 0:
            raise ValueError("reduce_weight must be positive")
        if not all(math.isfinite(value) for value in (self.current_weight, self.resulting_weight, self.created_at)):
            raise ValueError("risk-reduction values must be finite")
        if _sign(self.current_weight) != self.current_direction:
            raise ValueError("current position sign does not match current_direction")
        if self.reduce_weight > abs(self.current_weight) + 1e-12:
            raise ValueError("reduce-only weight cannot exceed current position")
        if abs(self.resulting_weight) > abs(self.current_weight) + 1e-12:
            raise ValueError("risk-reduction intent cannot increase exposure")
        if _sign(self.resulting_weight) not in (0, self.current_direction):
            raise ValueError("reduce-only intent cannot flip the position")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if not self.reduce_only or not self.shadow_only or self.live_execution:
            raise ValueError("Phase 55 risk reductions must remain reduce-only shadow intents")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PendingReversalOpen:
    pending_id: str
    asset_id: str
    required_flat_weight: float
    target_weight: float
    expected_edge_bps: float
    confidence: float
    max_slippage_bps: float
    created_at: float
    ttl_seconds: int
    evidence_ids: tuple[str, ...]
    parent_reduction_intent_id: str
    shadow_only: bool = True
    live_execution: bool = False
    automatic_release: bool = False
    schema_version: str = PHASE55_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.pending_id.strip() or not self.asset_id.strip() or not self.parent_reduction_intent_id.strip():
            raise ValueError("pending reversal identity is required")
        if abs(self.required_flat_weight) > 1e-12:
            raise ValueError("reversal open requires a flat-state barrier")
        if not math.isfinite(self.target_weight) or abs(self.target_weight) <= 1e-12:
            raise ValueError("reversal target_weight must be non-zero")
        if not math.isfinite(self.expected_edge_bps):
            raise ValueError("expected_edge_bps must be finite")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0,1]")
        if not math.isfinite(self.max_slippage_bps) or self.max_slippage_bps < 0:
            raise ValueError("max_slippage_bps must be non-negative")
        if not math.isfinite(self.created_at) or self.ttl_seconds <= 0:
            raise ValueError("pending reversal time contract is invalid")
        if not self.evidence_ids:
            raise ValueError("new-risk reversal leg requires evidence lineage")
        if not self.shadow_only or self.live_execution or self.automatic_release:
            raise ValueError("pending reversal opens cannot auto-release or execute live")


@dataclass(frozen=True, slots=True)
class FlatPositionReceipt:
    asset_id: str
    observed_at: float
    position_weight: float
    reconciliation_complete: bool
    authoritative: bool
    source_ref: str

    def __post_init__(self) -> None:
        if not self.asset_id.strip() or not self.source_ref.strip():
            raise ValueError("flat receipt identity is required")
        if not math.isfinite(self.observed_at) or not math.isfinite(self.position_weight):
            raise ValueError("flat receipt values must be finite")


@dataclass(frozen=True, slots=True)
class RebalanceExecutionInstruction:
    kind: InstructionKind
    asset_id: str
    current_weight: float
    planned_weight: float
    planned_delta: float
    reduction_intent: RiskReductionIntent | None = None
    trade_intent: TradeIntent | None = None
    pending_reversal: PendingReversalOpen | None = None
    reason: str = ""
    schema_version: str = PHASE55_SCHEMA_VERSION

    def __post_init__(self) -> None:
        populated = sum(
            value is not None
            for value in (self.reduction_intent, self.trade_intent, self.pending_reversal)
        )
        if self.kind == "REVERSAL_CLOSE":
            if self.reduction_intent is None or self.pending_reversal is None or self.trade_intent is not None:
                raise ValueError("REVERSAL_CLOSE requires reduction + pending open only")
        elif populated != 1:
            raise ValueError("instruction requires exactly one execution payload")
        if not self.asset_id.strip() or not self.reason.strip():
            raise ValueError("instruction asset/reason is required")


@dataclass(frozen=True, slots=True)
class RebalanceExecutionPlan:
    instructions: tuple[RebalanceExecutionInstruction, ...]
    skipped_assets: tuple[str, ...]
    plan_id: str
    schema_version: str = PHASE55_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False
    automatic_release: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "instructions": [asdict(row) for row in self.instructions],
            "skipped_assets": self.skipped_assets,
            "plan_id": self.plan_id,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
            "automatic_release": self.automatic_release,
        }


def _new_risk_intent(
    *,
    asset_id: str,
    delta_weight: float,
    expected_edge_bps: float,
    confidence: float,
    max_slippage_bps: float,
    created_at: float,
    ttl_seconds: int,
    evidence_ids: Sequence[str],
    label: str,
) -> TradeIntent:
    direction = _sign(delta_weight)
    if direction == 0:
        raise ValueError("new-risk delta must be directional")
    ids = tuple(sorted({str(value) for value in evidence_ids if str(value)}))
    if not ids:
        raise ValueError(f"{asset_id} new-risk intent requires evidence ids")
    intent_id = _hash({
        "schema": PHASE55_SCHEMA_VERSION,
        "asset_id": asset_id,
        "delta_weight": float(delta_weight),
        "expected_edge_bps": float(expected_edge_bps),
        "confidence": float(confidence),
        "created_at": float(created_at),
        "label": label,
        "evidence_ids": ids,
    })
    return TradeIntent(
        intent_id=intent_id,
        asset_id=asset_id,
        direction=direction,
        target_weight=float(delta_weight),
        expected_edge_bps=float(expected_edge_bps),
        confidence=float(confidence),
        max_slippage_bps=float(max_slippage_bps),
        created_at=float(created_at),
        ttl_seconds=int(ttl_seconds),
        evidence_ids=ids,
    )


def _risk_reduction(
    leg: RebalanceLeg,
    *,
    created_at: float,
    ttl_seconds: int,
    resulting_weight: float,
    label: str,
) -> RiskReductionIntent:
    current_direction = _sign(leg.current_weight)
    if current_direction == 0:
        raise ValueError("cannot reduce a flat position")
    reduce_weight = abs(leg.current_weight - resulting_weight)
    intent_id = _hash({
        "schema": PHASE55_SCHEMA_VERSION,
        "asset_id": leg.asset_id,
        "current_weight": leg.current_weight,
        "resulting_weight": resulting_weight,
        "created_at": created_at,
        "label": label,
    })
    return RiskReductionIntent(
        intent_id=intent_id,
        asset_id=leg.asset_id,
        current_direction=current_direction,
        order_direction=-current_direction,
        reduce_weight=reduce_weight,
        current_weight=leg.current_weight,
        resulting_weight=resulting_weight,
        reason=label,
        created_at=float(created_at),
        ttl_seconds=int(ttl_seconds),
    )


def compile_rebalance_execution_plan(
    turnover: TurnoverPlan,
    *,
    expected_edge_bps_by_asset: Mapping[str, float],
    confidence_by_asset: Mapping[str, float],
    evidence_ids_by_asset: Mapping[str, Sequence[str]],
    created_at: float,
    max_slippage_bps: float,
    ttl_seconds: int,
) -> RebalanceExecutionPlan:
    """Compile planned *deltas*, never final target weights, into execution intents.

    Risk reductions are reduce-only and cannot flip a position. New risk requires
    evidence + edge + confidence. A sign reversal is deliberately split into a
    close-to-flat instruction plus a pending opposite-side open which cannot be
    released until an authoritative flat-position receipt is supplied.
    """
    if not math.isfinite(created_at):
        raise ValueError("created_at must be finite")
    if not math.isfinite(max_slippage_bps) or max_slippage_bps < 0:
        raise ValueError("max_slippage_bps must be non-negative")
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")

    instructions: list[RebalanceExecutionInstruction] = []
    skipped: list[str] = []

    for leg in turnover.legs:
        if abs(leg.planned_delta) <= 1e-12:
            skipped.append(leg.asset_id)
            continue

        current_sign = _sign(leg.current_weight)
        planned_sign = _sign(leg.planned_weight)

        # Flat/open or same-side increase = new risk.
        if current_sign == 0 or (
            planned_sign == current_sign and abs(leg.planned_weight) > abs(leg.current_weight) + 1e-12
        ):
            edge = expected_edge_bps_by_asset.get(leg.asset_id)
            confidence = confidence_by_asset.get(leg.asset_id)
            if edge is None or confidence is None:
                raise ValueError(f"{leg.asset_id} new-risk leg missing edge/confidence")
            trade = _new_risk_intent(
                asset_id=leg.asset_id,
                delta_weight=leg.planned_delta,
                expected_edge_bps=float(edge),
                confidence=float(confidence),
                max_slippage_bps=max_slippage_bps,
                created_at=created_at,
                ttl_seconds=ttl_seconds,
                evidence_ids=evidence_ids_by_asset.get(leg.asset_id, ()),
                label="OPEN" if current_sign == 0 else "INCREASE",
            )
            instructions.append(RebalanceExecutionInstruction(
                kind="OPEN" if current_sign == 0 else "INCREASE",
                asset_id=leg.asset_id,
                current_weight=leg.current_weight,
                planned_weight=leg.planned_weight,
                planned_delta=leg.planned_delta,
                trade_intent=trade,
                reason="new or increased exposure requires grounded trade intent",
            ))
            continue

        # Same-side shrink or close to flat = reduce-only. No alpha requirement.
        if current_sign != 0 and planned_sign in (0, current_sign):
            if abs(leg.planned_weight) >= abs(leg.current_weight) - 1e-12:
                raise ValueError(f"{leg.asset_id} classified as reduction but exposure did not shrink")
            reduction = _risk_reduction(
                leg,
                created_at=created_at,
                ttl_seconds=ttl_seconds,
                resulting_weight=leg.planned_weight,
                label="CLOSE" if planned_sign == 0 else "REDUCE",
            )
            instructions.append(RebalanceExecutionInstruction(
                kind="CLOSE" if planned_sign == 0 else "REDUCE",
                asset_id=leg.asset_id,
                current_weight=leg.current_weight,
                planned_weight=leg.planned_weight,
                planned_delta=leg.planned_delta,
                reduction_intent=reduction,
                reason="risk-reducing rebalance leg; alpha evidence is not required to lower exposure",
            ))
            continue

        # Sign reversal: close existing risk first. Opposite-side open is contingent.
        if current_sign != 0 and planned_sign == -current_sign:
            reduction = _risk_reduction(
                leg,
                created_at=created_at,
                ttl_seconds=ttl_seconds,
                resulting_weight=0.0,
                label="REVERSAL_CLOSE",
            )
            edge = expected_edge_bps_by_asset.get(leg.asset_id)
            confidence = confidence_by_asset.get(leg.asset_id)
            ids = tuple(sorted({
                str(value)
                for value in evidence_ids_by_asset.get(leg.asset_id, ())
                if str(value)
            }))
            if edge is None or confidence is None or not ids:
                raise ValueError(f"{leg.asset_id} reversal open leg missing grounded edge/confidence/evidence")
            pending_id = _hash({
                "schema": PHASE55_SCHEMA_VERSION,
                "asset_id": leg.asset_id,
                "target_weight": leg.planned_weight,
                "parent_reduction_intent_id": reduction.intent_id,
                "created_at": created_at,
            })
            pending = PendingReversalOpen(
                pending_id=pending_id,
                asset_id=leg.asset_id,
                required_flat_weight=0.0,
                target_weight=leg.planned_weight,
                expected_edge_bps=float(edge),
                confidence=float(confidence),
                max_slippage_bps=float(max_slippage_bps),
                created_at=float(created_at),
                ttl_seconds=int(ttl_seconds),
                evidence_ids=ids,
                parent_reduction_intent_id=reduction.intent_id,
            )
            instructions.append(RebalanceExecutionInstruction(
                kind="REVERSAL_CLOSE",
                asset_id=leg.asset_id,
                current_weight=leg.current_weight,
                planned_weight=leg.planned_weight,
                planned_delta=leg.planned_delta,
                reduction_intent=reduction,
                pending_reversal=pending,
                reason="direction reversal is split: authoritative close-to-flat must precede opposite-side new risk",
            ))
            continue

        raise ValueError(f"unsupported rebalance path for {leg.asset_id}")

    payload = {
        "schema": PHASE55_SCHEMA_VERSION,
        "instructions": [asdict(row) for row in instructions],
        "skipped_assets": sorted(skipped),
    }
    return RebalanceExecutionPlan(
        instructions=tuple(instructions),
        skipped_assets=tuple(sorted(skipped)),
        plan_id=_hash(payload),
    )


def release_reversal_open(
    pending: PendingReversalOpen,
    receipt: FlatPositionReceipt,
) -> TradeIntent:
    """Release the opposite-side leg only after authoritative flat reconciliation."""
    if receipt.asset_id != pending.asset_id:
        raise ValueError("flat receipt belongs to a different asset")
    if receipt.observed_at < pending.created_at:
        raise ValueError("flat receipt predates reversal request")
    if receipt.observed_at - pending.created_at > pending.ttl_seconds:
        raise ValueError("pending reversal expired before flat confirmation")
    if not receipt.authoritative:
        raise ValueError("reversal release requires authoritative position state")
    if not receipt.reconciliation_complete:
        raise ValueError("reversal release requires completed reconciliation")
    if abs(receipt.position_weight - pending.required_flat_weight) > 1e-12:
        raise ValueError("position is not flat; opposite-side risk remains blocked")

    return _new_risk_intent(
        asset_id=pending.asset_id,
        delta_weight=pending.target_weight,
        expected_edge_bps=pending.expected_edge_bps,
        confidence=pending.confidence,
        max_slippage_bps=pending.max_slippage_bps,
        created_at=receipt.observed_at,
        ttl_seconds=pending.ttl_seconds,
        evidence_ids=pending.evidence_ids,
        label="REVERSAL_OPEN_AFTER_FLAT",
    )


def compile_from_integrated_decision(
    decision: IntegratedShadowDecision,
    *,
    expected_edge_bps_by_asset: Mapping[str, float],
    confidence_by_asset: Mapping[str, float],
    evidence_ids_by_asset: Mapping[str, Sequence[str]],
    max_slippage_bps: float,
    ttl_seconds: int,
) -> RebalanceExecutionPlan:
    if decision.turnover_plan is None:
        return RebalanceExecutionPlan((), tuple(sorted(decision.current_weights)), _hash({
            "schema": PHASE55_SCHEMA_VERSION,
            "pipeline_id": decision.pipeline_id,
            "status": decision.status,
            "instructions": [],
        }))
    return compile_rebalance_execution_plan(
        decision.turnover_plan,
        expected_edge_bps_by_asset=expected_edge_bps_by_asset,
        confidence_by_asset=confidence_by_asset,
        evidence_ids_by_asset=evidence_ids_by_asset,
        created_at=decision.timestamp,
        max_slippage_bps=max_slippage_bps,
        ttl_seconds=ttl_seconds,
    )
