from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

from .intelligence_fabric import WhaleObservation, assess_whale


@dataclass(frozen=True, slots=True)
class SmartMoneyConfig:
    min_entity_confidence: float = 0.55
    min_materiality: float = 0.10
    strong_breadth_entities: int = 3
    max_single_entity_share: float = 0.70
    concentration_penalty: float = 0.65

    def __post_init__(self) -> None:
        if not 0 <= self.min_entity_confidence <= 1 or not 0 <= self.min_materiality <= 1:
            raise ValueError("invalid smart-money thresholds")
        if self.strong_breadth_entities < 2:
            raise ValueError("strong_breadth_entities must be at least 2")
        if not 0 < self.max_single_entity_share <= 1 or not 0 <= self.concentration_penalty <= 1:
            raise ValueError("invalid concentration settings")


@dataclass(frozen=True, slots=True)
class SmartMoneyConsensus:
    asset: str
    observed_at: float
    qualified_observations: int
    unique_entities: int
    bullish_entities: int
    bearish_entities: int
    unresolved_observations: int
    gross_usd: float
    directional_usd: float
    single_entity_share: float
    breadth_score: float
    direction: float
    confidence: float
    accumulation_score: float
    distribution_score: float
    concentrated: bool
    veto_reasons: tuple[str, ...]
    schema_version: str = "brian.smart-money-consensus.v1"


def smart_money_consensus(
    observations: Sequence[WhaleObservation],
    *,
    config: SmartMoneyConfig = SmartMoneyConfig(),
    material_usd: float = 100_000.0,
) -> SmartMoneyConsensus:
    if not observations:
        raise ValueError("smart-money consensus requires observations")
    if material_usd <= 0:
        raise ValueError("material_usd must be positive")
    asset = observations[0].asset.strip().upper()
    if any(row.asset.strip().upper() != asset for row in observations):
        raise ValueError("smart-money consensus requires one asset")

    qualified: list[tuple[WhaleObservation, object]] = []
    unresolved = 0
    for row in observations:
        assessment = assess_whale(row, material_usd=material_usd)
        if assessment.suspicious_or_unresolved:
            unresolved += 1
        if (
            not assessment.suspicious_or_unresolved and
            assessment.confidence >= config.min_entity_confidence and
            assessment.materiality >= config.min_materiality and
            abs(assessment.economic_direction) > 0
        ):
            qualified.append((row, assessment))

    gross = sum(row.usd_value for row, _ in qualified)
    by_entity: dict[str, float] = {}
    direction_by_entity: dict[str, float] = {}
    for row, assessment in qualified:
        by_entity[row.entity_id] = by_entity.get(row.entity_id, 0.0) + row.usd_value
        direction_by_entity[row.entity_id] = direction_by_entity.get(row.entity_id, 0.0) + (
            row.usd_value * assessment.economic_direction
        )

    unique_entities = len(by_entity)
    single_share = max(by_entity.values(), default=0.0) / gross if gross > 0 else 0.0
    concentrated = bool(gross > 0 and single_share > config.max_single_entity_share)
    breadth = min(1.0, unique_entities / config.strong_breadth_entities)

    entity_signs = {
        entity: 1 if value > 0 else -1 if value < 0 else 0
        for entity, value in direction_by_entity.items()
    }
    bullish = sum(sign > 0 for sign in entity_signs.values())
    bearish = sum(sign < 0 for sign in entity_signs.values())

    directional_usd = sum(
        row.usd_value * assessment.economic_direction for row, assessment in qualified
    )
    direction = directional_usd / gross if gross > 0 else 0.0
    direction = max(-1.0, min(1.0, direction))

    mean_confidence = (
        sum(assessment.confidence for _, assessment in qualified) / len(qualified)
        if qualified else 0.0
    )
    unresolved_fraction = unresolved / len(observations)
    concentration_scale = (1.0 - config.concentration_penalty) if concentrated else 1.0
    confidence = mean_confidence * breadth * (1.0 - 0.70 * unresolved_fraction) * concentration_scale
    confidence = max(0.0, min(1.0, confidence))

    accumulation = max(0.0, direction) * confidence
    distribution = max(0.0, -direction) * confidence
    veto: list[str] = []
    if not qualified:
        veto.append("no verified directional smart-money observations")
    if unique_entities < 2:
        veto.append("insufficient independent entity breadth")
    if concentrated:
        veto.append("flow is dominated by one entity")
    if unresolved_fraction >= 0.50:
        veto.append("too many unresolved whale observations")

    return SmartMoneyConsensus(
        asset=asset,
        observed_at=max(float(row.observed_at) for row in observations),
        qualified_observations=len(qualified),
        unique_entities=unique_entities,
        bullish_entities=bullish,
        bearish_entities=bearish,
        unresolved_observations=unresolved,
        gross_usd=float(gross),
        directional_usd=float(directional_usd),
        single_entity_share=float(single_share),
        breadth_score=float(breadth),
        direction=float(direction),
        confidence=float(confidence),
        accumulation_score=float(max(0.0, min(1.0, accumulation))),
        distribution_score=float(max(0.0, min(1.0, distribution))),
        concentrated=concentrated,
        veto_reasons=tuple(veto),
    )
