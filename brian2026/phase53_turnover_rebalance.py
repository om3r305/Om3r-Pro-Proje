from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping
import math

from .phase52_covariance_risk import CovarianceRiskOverlay

PHASE53_SCHEMA_VERSION = "brian.phase53-turnover-rebalance.v1"


@dataclass(frozen=True, slots=True)
class TurnoverConfig:
    max_l1_turnover: float = 0.25
    risk_reduction_bypass: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_l1_turnover) or self.max_l1_turnover < 0:
            raise ValueError("max_l1_turnover must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RebalanceLeg:
    asset_id: str
    current_weight: float
    target_weight: float
    mandatory_risk_weight: float
    planned_weight: float
    mandatory_delta: float
    discretionary_delta: float
    planned_delta: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TurnoverPlan:
    legs: tuple[RebalanceLeg, ...]
    desired_turnover: float
    mandatory_risk_turnover: float
    discretionary_turnover: float
    planned_turnover: float
    discretionary_scale: float
    residual_l1_to_target: float
    turnover_limit_exceeded_only_for_risk_reduction: bool
    schema_version: str = PHASE53_SCHEMA_VERSION
    risk_reductions_not_blocked: bool = True
    shadow_only: bool = True
    live_execution: bool = False

    @property
    def planned_weights(self) -> dict[str, float]:
        return {leg.asset_id: leg.planned_weight for leg in self.legs}

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["planned_weights"] = self.planned_weights
        return payload


def _finite_weight(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _mandatory_risk_weight(current: float, target: float) -> float:
    """First move only toward lower absolute exposure, never into new risk.

    - Same sign: cut current down to target magnitude when target is smaller.
    - Target flat: close current.
    - Opposite sign: close current to flat before any opposite-side opening.
    - Larger same-side target: no mandatory move; increase is discretionary.
    """
    if abs(current) <= 1e-15:
        return 0.0
    if abs(target) <= 1e-15:
        return 0.0
    same_sign = math.copysign(1.0, current) == math.copysign(1.0, target)
    if not same_sign:
        return 0.0
    if abs(target) < abs(current):
        return target
    return current


def plan_turnover_constrained_rebalance(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    *,
    config: TurnoverConfig = TurnoverConfig(),
) -> TurnoverPlan:
    """Move toward target weights under a Qlib-style L1 turnover budget.

    Qlib exposes turnover as |w - w0| <= delta inside portfolio optimization.
    Brian's clean-room planner applies the same L1 budget directly to an already
    risk-approved target. A safety adaptation executes exposure-reducing moves
    first; a transaction-cost budget is never allowed to block a hard risk cut.
    Remaining risk additions/rotations are scaled proportionally to fit the
    residual turnover budget.
    """
    assets = tuple(sorted(set(current_weights) | set(target_weights)))
    if not assets:
        raise ValueError("rebalance requires at least one asset")

    rows: list[tuple[str, float, float, float, float, float]] = []
    mandatory_turnover = 0.0
    discretionary_turnover = 0.0
    desired_turnover = 0.0

    for asset in assets:
        current = _finite_weight(current_weights.get(asset, 0.0), label=f"{asset} current_weight")
        target = _finite_weight(target_weights.get(asset, 0.0), label=f"{asset} target_weight")
        mandatory_weight = _mandatory_risk_weight(current, target)
        mandatory_delta = mandatory_weight - current
        discretionary_delta = target - mandatory_weight

        mandatory_turnover += abs(mandatory_delta)
        discretionary_turnover += abs(discretionary_delta)
        desired_turnover += abs(target - current)
        rows.append((
            asset,
            current,
            target,
            mandatory_weight,
            mandatory_delta,
            discretionary_delta,
        ))

    if config.risk_reduction_bypass:
        remaining = max(0.0, config.max_l1_turnover - mandatory_turnover)
        if discretionary_turnover <= 1e-15:
            discretionary_scale = 1.0
        else:
            discretionary_scale = min(1.0, remaining / discretionary_turnover)
        mandatory_scale = 1.0
    else:
        total = mandatory_turnover + discretionary_turnover
        common_scale = 1.0 if total <= 1e-15 else min(1.0, config.max_l1_turnover / total)
        mandatory_scale = common_scale
        discretionary_scale = common_scale

    legs: list[RebalanceLeg] = []
    planned_turnover = 0.0
    residual = 0.0

    for asset, current, target, mandatory_weight, mandatory_delta, discretionary_delta in rows:
        if config.risk_reduction_bypass:
            risk_stage_weight = current + mandatory_delta
            planned = risk_stage_weight + discretionary_scale * discretionary_delta
        else:
            planned = current + mandatory_scale * mandatory_delta + discretionary_scale * discretionary_delta

        # Numerical fail-safe: planned path must lie on the line segments from
        # current -> mandatory risk weight -> target and therefore never
        # overshoot the target.
        planned_delta = planned - current
        desired_delta = target - current
        if abs(planned_delta) > abs(mandatory_delta) + abs(discretionary_delta) + 1e-12:
            raise ValueError("turnover planner overshot requested path")

        planned_turnover += abs(planned_delta)
        residual += abs(target - planned)
        legs.append(RebalanceLeg(
            asset_id=asset,
            current_weight=current,
            target_weight=target,
            mandatory_risk_weight=mandatory_weight,
            planned_weight=float(planned),
            mandatory_delta=mandatory_delta,
            discretionary_delta=discretionary_delta,
            planned_delta=planned_delta,
        ))

    bypass_exceeded = bool(
        config.risk_reduction_bypass
        and mandatory_turnover > config.max_l1_turnover + 1e-12
        and planned_turnover > config.max_l1_turnover + 1e-12
    )
    if not bypass_exceeded and planned_turnover > config.max_l1_turnover + 1e-10:
        raise ValueError("planned turnover exceeds configured limit")

    return TurnoverPlan(
        legs=tuple(legs),
        desired_turnover=desired_turnover,
        mandatory_risk_turnover=mandatory_turnover,
        discretionary_turnover=discretionary_turnover,
        planned_turnover=planned_turnover,
        discretionary_scale=discretionary_scale,
        residual_l1_to_target=residual,
        turnover_limit_exceeded_only_for_risk_reduction=bypass_exceeded,
        risk_reductions_not_blocked=config.risk_reduction_bypass,
    )


def plan_from_covariance_overlay(
    current_weights: Mapping[str, float],
    overlay: CovarianceRiskOverlay,
    *,
    config: TurnoverConfig = TurnoverConfig(),
) -> TurnoverPlan:
    return plan_turnover_constrained_rebalance(
        current_weights,
        overlay.scaled_weights,
        config=config,
    )
