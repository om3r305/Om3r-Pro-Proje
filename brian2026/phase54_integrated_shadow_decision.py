from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Callable, Mapping, Sequence
import json
import math

from .global_sensor_mesh import SensorObservation
from .phase43_grounded_analysts import Phase43GroundedResult, run_grounded_phase43
from .phase44_portfolio_brain import (
    PortfolioBookPlan,
    PortfolioRiskLimits,
    PortfolioSignal,
    construct_portfolio_book,
    signals_from_grounded_claims,
)
from .phase52_covariance_risk import (
    CovarianceRiskConfig,
    CovarianceRiskOverlay,
    apply_covariance_risk_overlay,
)
from .phase53_turnover_rebalance import (
    TurnoverConfig,
    TurnoverPlan,
    plan_from_covariance_overlay,
)

PHASE54_SCHEMA_VERSION = "brian.phase54-integrated-shadow-decision.v1"


def _hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AssetDecisionInput:
    snapshot: Mapping[str, object]
    observations: tuple[SensorObservation, ...]
    source_kind_by_eye: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("asset decision input requires observations")


@dataclass(frozen=True, slots=True)
class IntegratedShadowConfig:
    gross_target: float
    position_limits: PortfolioRiskLimits
    covariance: CovarianceRiskConfig
    turnover: TurnoverConfig
    market_neutral: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.gross_target) or self.gross_target < 0:
            raise ValueError("gross_target must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class IntegratedShadowDecision:
    status: str
    timestamp: float
    asset_results: Mapping[str, Phase43GroundedResult]
    portfolio_book: PortfolioBookPlan | None
    covariance_overlay: CovarianceRiskOverlay | None
    turnover_plan: TurnoverPlan | None
    current_weights: Mapping[str, float]
    final_planned_weights: Mapping[str, float]
    source_evidence_ids: tuple[str, ...]
    pipeline_id: str
    schema_version: str = PHASE54_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False
    automatic_promotion: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "timestamp": self.timestamp,
            "asset_results": {
                asset: result.to_dict()
                for asset, result in sorted(self.asset_results.items())
            },
            "portfolio_book": None if self.portfolio_book is None else self.portfolio_book.to_dict(),
            "covariance_overlay": None if self.covariance_overlay is None else self.covariance_overlay.to_dict(),
            "turnover_plan": None if self.turnover_plan is None else self.turnover_plan.to_dict(),
            "current_weights": dict(sorted(self.current_weights.items())),
            "final_planned_weights": dict(sorted(self.final_planned_weights.items())),
            "source_evidence_ids": self.source_evidence_ids,
            "pipeline_id": self.pipeline_id,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
            "automatic_promotion": self.automatic_promotion,
        }


def _phase43_signals(
    results: Mapping[str, Phase43GroundedResult],
) -> tuple[PortfolioSignal, ...]:
    rows: list[PortfolioSignal] = []
    for asset, result in sorted(results.items()):
        rows.extend(signals_from_grounded_claims(asset, result.analyst_claims))
    return tuple(rows)


GroundedRunner = Callable[..., Phase43GroundedResult]


def run_integrated_shadow_decision(
    asset_inputs: Mapping[str, AssetDecisionInput],
    *,
    timestamp: float,
    model_weights: Mapping[str, float],
    current_weights: Mapping[str, float],
    returns_by_asset: Mapping[str, Sequence[float]],
    config: IntegratedShadowConfig,
    grounded_runner: GroundedRunner = run_grounded_phase43,
) -> IntegratedShadowDecision:
    """Run the evidence -> portfolio -> covariance -> turnover chain.

    This is the first integration layer for Phases 43/44/52/53. Each component
    remains independently testable, but production-style composition now uses
    the actual output contract of the prior stage rather than recreating logic.
    No exchange/executor surface exists in this phase.
    """
    if not asset_inputs:
        raise ValueError("integrated decision requires asset inputs")
    if not math.isfinite(timestamp):
        raise ValueError("timestamp must be finite")

    clean_current = {
        str(asset): float(weight)
        for asset, weight in current_weights.items()
    }
    if any(not math.isfinite(value) for value in clean_current.values()):
        raise ValueError("current weights must be finite")

    if not callable(grounded_runner):
        raise TypeError("grounded_runner must be callable")

    results: dict[str, Phase43GroundedResult] = {}
    for asset, item in sorted(asset_inputs.items()):
        observations = tuple(item.observations)
        observed_assets = {row.asset_id for row in observations}
        if observed_assets != {asset}:
            raise ValueError(
                f"asset input key {asset} does not match observation assets {sorted(observed_assets)}"
            )
        results[asset] = grounded_runner(
            item.snapshot,
            observations,
            timestamp=timestamp,
            source_kind_by_eye=item.source_kind_by_eye,
        )

    signals = _phase43_signals(results)
    evidence_ids = tuple(sorted({
        evidence_id
        for result in results.values()
        for claim in result.analyst_claims
        for evidence_id in claim.support_evidence_ids
    }))

    if not signals:
        payload = {
            "schema": PHASE54_SCHEMA_VERSION,
            "status": "WAIT_NO_GROUNDED_SIGNALS",
            "timestamp": timestamp,
            "current_weights": dict(sorted(clean_current.items())),
            "asset_packet_ids": {
                asset: tuple(result.packet.usable_evidence_ids)
                for asset, result in sorted(results.items())
            },
        }
        return IntegratedShadowDecision(
            status="WAIT_NO_GROUNDED_SIGNALS",
            timestamp=float(timestamp),
            asset_results=results,
            portfolio_book=None,
            covariance_overlay=None,
            turnover_plan=None,
            current_weights=clean_current,
            final_planned_weights=clean_current,
            source_evidence_ids=evidence_ids,
            pipeline_id=_hash(payload),
        )

    missing_weights = sorted({
        signal.model_name
        for signal in signals
        if signal.model_name not in model_weights
    })
    if missing_weights:
        raise ValueError(f"missing model weights for grounded analysts: {missing_weights}")

    book = construct_portfolio_book(
        signals,
        model_weights,
        gross_target=config.gross_target,
        limits=config.position_limits,
        market_neutral=config.market_neutral,
    )
    overlay = apply_covariance_risk_overlay(
        book,
        returns_by_asset,
        config=config.covariance,
    )
    turnover = plan_from_covariance_overlay(
        clean_current,
        overlay,
        config=config.turnover,
    )
    planned = turnover.planned_weights
    changed = any(
        abs(planned.get(asset, 0.0) - clean_current.get(asset, 0.0)) > 1e-12
        for asset in set(planned) | set(clean_current)
    )
    status = "REBALANCE_PLANNED" if changed else "HOLD_CURRENT_BOOK"

    payload = {
        "schema": PHASE54_SCHEMA_VERSION,
        "status": status,
        "timestamp": timestamp,
        "portfolio": book.to_dict(),
        "covariance": overlay.to_dict(),
        "turnover": turnover.to_dict(),
        "evidence_ids": evidence_ids,
    }
    return IntegratedShadowDecision(
        status=status,
        timestamp=float(timestamp),
        asset_results=results,
        portfolio_book=book,
        covariance_overlay=overlay,
        turnover_plan=turnover,
        current_weights=clean_current,
        final_planned_weights=planned,
        source_evidence_ids=evidence_ids,
        pipeline_id=_hash(payload),
    )
