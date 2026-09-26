from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
import math

from .phase43_grounded_analysts import ValidatedAnalystClaim

PHASE44_SCHEMA_VERSION = "brian.phase44-portfolio-brain.v1"


@dataclass(frozen=True, slots=True)
class PortfolioSignal:
    model_name: str
    asset_id: str
    conviction: float
    abstained: bool = False
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_name.strip() or not self.asset_id.strip():
            raise ValueError("model_name and asset_id are required")
        if not math.isfinite(float(self.conviction)) or not -1.0 <= float(self.conviction) <= 1.0:
            raise ValueError("conviction must be finite in [-1,1]")
        if not self.abstained and self.conviction != 0.0 and not self.evidence_ids:
            raise ValueError("directional portfolio signals require evidence ids")


@dataclass(frozen=True, slots=True)
class BlendResult:
    convictions: Mapping[str, float]
    requested_weights: Mapping[str, float]
    gross_target: float
    market_neutral: bool
    schema_version: str = PHASE44_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "convictions": dict(self.convictions),
            "requested_weights": dict(self.requested_weights),
            "gross_target": self.gross_target,
            "market_neutral": self.market_neutral,
        }


def blend_signals(
    signals: Sequence[PortfolioSignal],
    model_weights: Mapping[str, float],
    *,
    gross_target: float,
    market_neutral: bool = False,
) -> BlendResult:
    """Conviction-weighted portfolio request.

    Behavior follows the public ai-hedge-fund portfolio constructor:
    abstentions are excluded from numerator and denominator, a real 0.0 is a
    neutral vote that dilutes, optional market-neutral mode demeans
    cross-sectionally, and requested weights are normalized to gross_target.
    """
    if not math.isfinite(float(gross_target)) or gross_target < 0:
        raise ValueError("gross_target must be finite and non-negative")
    if not signals:
        raise ValueError("portfolio blending requires signals")

    clean_weights: dict[str, float] = {}
    for name, value in model_weights.items():
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("model weights must be finite and non-negative")
        clean_weights[str(name)] = weight

    weighted_sum: dict[str, float] = {}
    weight_total: dict[str, float] = {}
    assets = sorted({signal.asset_id for signal in signals})
    for signal in signals:
        if signal.model_name not in clean_weights:
            raise ValueError(f"missing model weight: {signal.model_name}")
        if signal.abstained:
            continue
        weight = clean_weights[signal.model_name]
        weighted_sum[signal.asset_id] = weighted_sum.get(signal.asset_id, 0.0) + weight * signal.conviction
        weight_total[signal.asset_id] = weight_total.get(signal.asset_id, 0.0) + weight

    convictions = {
        asset: weighted_sum.get(asset, 0.0) / weight_total[asset]
        if weight_total.get(asset, 0.0) > 0 else 0.0
        for asset in assets
    }
    scaled = dict(convictions)
    if market_neutral and assets:
        mean = sum(convictions.values()) / len(assets)
        scaled = {asset: conviction - mean for asset, conviction in convictions.items()}

    gross = sum(abs(value) for value in scaled.values())
    if gross < 1e-9 or gross_target == 0:
        requested = {asset: 0.0 for asset in assets}
    else:
        requested = {asset: value / gross * gross_target for asset, value in scaled.items()}

    return BlendResult(
        convictions=dict(sorted(convictions.items())),
        requested_weights=dict(sorted(requested.items())),
        gross_target=float(gross_target),
        market_neutral=bool(market_neutral),
    )


@dataclass(frozen=True, slots=True)
class PortfolioRiskLimits:
    max_position_pct: float
    max_gross_exposure: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.max_position_pct)) or not 0 < self.max_position_pct <= 1.0:
            raise ValueError("max_position_pct must be in (0,1]")
        if not math.isfinite(float(self.max_gross_exposure)) or self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be positive")


@dataclass(frozen=True, slots=True)
class PortfolioClampEvent:
    limit: str
    asset_id: str | None
    before: float
    after: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioRiskResult:
    weights: Mapping[str, float]
    clamps: tuple[PortfolioClampEvent, ...]
    cash_weight: float
    gross_exposure: float
    net_exposure: float
    schema_version: str = PHASE44_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "weights": dict(self.weights),
            "clamps": [row.to_dict() for row in self.clamps],
            "cash_weight": self.cash_weight,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }


def apply_hard_limits(
    weights: Mapping[str, float],
    limits: PortfolioRiskLimits,
) -> PortfolioRiskResult:
    """Risk only shrinks requested exposure; it never reallocates freed capital.

    Limit order matches ai-hedge-fund exactly:
    1) absolute per-asset position cap;
    2) proportional gross cap.
    A clamp can only reduce |weight|, so capital removed by risk stays cash.
    """
    clamped: dict[str, float] = {}
    events: list[PortfolioClampEvent] = []

    for asset in sorted(weights):
        weight = float(weights[asset])
        if not math.isfinite(weight):
            raise ValueError("portfolio weights must be finite")
        if abs(weight) > limits.max_position_pct:
            new_weight = limits.max_position_pct if weight > 0 else -limits.max_position_pct
            events.append(PortfolioClampEvent("max_position_pct", asset, weight, new_weight))
            clamped[asset] = new_weight
        else:
            clamped[asset] = weight

    gross = sum(abs(value) for value in clamped.values())
    if gross > limits.max_gross_exposure:
        scale = limits.max_gross_exposure / gross
        before = gross
        clamped = {asset: value * scale for asset, value in clamped.items()}
        events.append(PortfolioClampEvent("max_gross_exposure", None, before, limits.max_gross_exposure))

    final_gross = sum(abs(value) for value in clamped.values())
    net = sum(clamped.values())
    # Cash is uncommitted unlevered equity. If a future research configuration
    # allows gross > 1, cash cannot become negative in this shadow plan.
    cash_weight = max(0.0, 1.0 - final_gross)
    return PortfolioRiskResult(
        weights=dict(sorted(clamped.items())),
        clamps=tuple(events),
        cash_weight=cash_weight,
        gross_exposure=final_gross,
        net_exposure=net,
    )


@dataclass(frozen=True, slots=True)
class PortfolioBookPlan:
    blend: BlendResult
    risk: PortfolioRiskResult
    requested_gross_exposure: float
    released_to_cash: float
    source_evidence_ids: tuple[str, ...]
    schema_version: str = PHASE44_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False
    automatic_promotion: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "blend": self.blend.to_dict(),
            "risk": self.risk.to_dict(),
            "requested_gross_exposure": self.requested_gross_exposure,
            "released_to_cash": self.released_to_cash,
            "source_evidence_ids": self.source_evidence_ids,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
            "automatic_promotion": self.automatic_promotion,
        }


def construct_portfolio_book(
    signals: Sequence[PortfolioSignal],
    model_weights: Mapping[str, float],
    *,
    gross_target: float,
    limits: PortfolioRiskLimits,
    market_neutral: bool = False,
) -> PortfolioBookPlan:
    blend = blend_signals(
        signals,
        model_weights,
        gross_target=gross_target,
        market_neutral=market_neutral,
    )
    risk = apply_hard_limits(blend.requested_weights, limits)
    requested_gross = sum(abs(value) for value in blend.requested_weights.values())
    released = max(0.0, requested_gross - risk.gross_exposure)
    source_ids = tuple(sorted({
        evidence_id
        for signal in signals
        if not signal.abstained
        for evidence_id in signal.evidence_ids
    }))
    return PortfolioBookPlan(
        blend=blend,
        risk=risk,
        requested_gross_exposure=requested_gross,
        released_to_cash=released,
        source_evidence_ids=source_ids,
    )


def signals_from_grounded_claims(
    asset_id: str,
    claims: Sequence[ValidatedAnalystClaim],
) -> tuple[PortfolioSignal, ...]:
    """Translate Phase 43 grounded claims into portfolio requests.

    Grounded confidence becomes conviction magnitude; direction provides sign.
    Evidence lineage is preserved. No claim means no synthetic neutral model.
    """
    rows: list[PortfolioSignal] = []
    for claim in claims:
        if claim.direction == 0:
            rows.append(PortfolioSignal(
                model_name=claim.analyst,
                asset_id=asset_id,
                conviction=0.0,
                abstained=False,
                evidence_ids=tuple(claim.evidence_ids),
            ))
            continue
        rows.append(PortfolioSignal(
            model_name=claim.analyst,
            asset_id=asset_id,
            conviction=claim.direction * claim.grounded_confidence,
            abstained=False,
            evidence_ids=tuple(claim.support_evidence_ids),
        ))
    return tuple(rows)
