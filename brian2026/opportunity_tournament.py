from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence
import hashlib
import json
import math

from .global_sensor_mesh import SensorObservation, PROSPECTIVE_EVIDENCE_CLASS

TOURNAMENT_SCHEMA_VERSION = "brian.opportunity-tournament.v1"


def _hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TournamentConfig:
    min_independent_groups: int = 2
    min_opportunity_score: float = 0.20
    max_candidates: int = 25
    total_virtual_equity: float = 500.0
    max_virtual_candidate_ticket: float = 20.0
    max_virtual_gross_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.min_independent_groups < 1 or self.max_candidates < 1:
            raise ValueError("invalid tournament breadth")
        if not 0 <= self.min_opportunity_score <= 1:
            raise ValueError("min_opportunity_score must be in [0,1]")
        if self.total_virtual_equity <= 0 or self.max_virtual_candidate_ticket <= 0:
            raise ValueError("virtual equity/ticket must be positive")
        if not 0 < self.max_virtual_gross_fraction <= 1:
            raise ValueError("tournament remains unlevered")


@dataclass(frozen=True, slots=True)
class TournamentCandidate:
    asset_id: str
    horizon: str
    observed_at: float
    direction: int
    opportunity_score: float
    independent_groups: tuple[str, ...]
    supporting_observation_ids: tuple[str, ...]
    conflicting_observation_ids: tuple[str, ...]
    virtual_ticket_usd: float
    eligible: bool
    veto_reasons: tuple[str, ...]
    shadow_only: bool = True
    live_execution: bool = False
    evidence_class: str = PROSPECTIVE_EVIDENCE_CLASS
    schema_version: str = TOURNAMENT_SCHEMA_VERSION

    @property
    def candidate_id(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True, slots=True)
class TournamentRound:
    observed_at: float
    candidates: tuple[TournamentCandidate, ...]
    virtual_allocated_usd: float
    virtual_unallocated_usd: float
    shadow_only: bool = True
    live_execution: bool = False
    evidence_class: str = PROSPECTIVE_EVIDENCE_CLASS
    schema_version: str = TOURNAMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.virtual_allocated_usd < 0 or self.virtual_unallocated_usd < 0:
            raise ValueError("virtual capital accounting cannot be negative")
        if not self.shadow_only or self.live_execution:
            raise ValueError("tournament cannot execute live orders")

    @property
    def round_id(self) -> str:
        return _hash(asdict(self))


def _best_per_independent_group(observations: Sequence[SensorObservation]) -> tuple[SensorObservation, ...]:
    """Prevent one evidence family/source-cluster from pretending to be many votes."""
    chosen: dict[str, SensorObservation] = {}
    for row in observations:
        if not row.available or row.direction == 0:
            continue
        current = chosen.get(row.independent_group)
        quality = row.strength * row.confidence * row.reliability
        current_quality = current.strength * current.confidence * current.reliability if current else -1.0
        if quality > current_quality or (math.isclose(quality, current_quality) and row.observation_id < current.observation_id):
            chosen[row.independent_group] = row
    return tuple(chosen[key] for key in sorted(chosen))


def _ticket(score: float, cap: float) -> float:
    if score >= 0.80:
        return min(20.0, cap)
    if score >= 0.60:
        return min(10.0, cap)
    if score >= 0.40:
        return min(5.0, cap)
    if score >= 0.20:
        return min(3.0, cap)
    return min(2.0, cap)


def build_candidate(
    observations: Sequence[SensorObservation],
    *,
    config: TournamentConfig = TournamentConfig(),
) -> TournamentCandidate:
    if not observations:
        raise ValueError("candidate requires observations")
    first = observations[0]
    if any(row.asset_id != first.asset_id or row.horizon != first.horizon for row in observations):
        raise ValueError("candidate observations must share asset and horizon")
    if any(abs(row.observed_at - first.observed_at) > 1e-6 for row in observations):
        raise ValueError("candidate observations must share the same PIT decision time")

    independent = _best_per_independent_group(observations)
    veto: list[str] = []
    if len(independent) < config.min_independent_groups:
        veto.append("insufficient independent evidence groups")

    signed = [row.direction * row.strength * row.confidence * row.reliability for row in independent]
    weights = [max(1e-12, row.confidence * row.reliability) for row in independent]
    aggregate = sum(s * w for s, w in zip(signed, weights)) / sum(weights) if weights else 0.0
    direction = 1 if aggregate > 0 else -1 if aggregate < 0 else 0
    support = tuple(row for row in independent if row.direction == direction and direction != 0)
    conflicts = tuple(row for row in independent if row.direction != direction and row.direction != 0)
    independence_scale = min(1.0, len(independent) / max(config.min_independent_groups + 2, 1))
    agreement_scale = 0.0 if not independent else len(support) / len(independent)
    score = min(1.0, abs(aggregate) * (0.50 + 0.50 * agreement_scale) * (0.50 + 0.50 * independence_scale))

    if direction == 0:
        veto.append("no directional aggregate")
    if score < config.min_opportunity_score:
        veto.append("opportunity score below preregistered threshold")
    eligible = not veto
    return TournamentCandidate(
        asset_id=first.asset_id,
        horizon=first.horizon,
        observed_at=first.observed_at,
        direction=direction,
        opportunity_score=score,
        independent_groups=tuple(row.independent_group for row in independent),
        supporting_observation_ids=tuple(row.observation_id for row in support),
        conflicting_observation_ids=tuple(row.observation_id for row in conflicts),
        virtual_ticket_usd=_ticket(score, config.max_virtual_candidate_ticket) if eligible else 0.0,
        eligible=eligible,
        veto_reasons=tuple(veto),
    )


def run_tournament(
    observations: Sequence[SensorObservation],
    *,
    config: TournamentConfig = TournamentConfig(),
) -> TournamentRound:
    if not observations:
        raise ValueError("tournament requires observations")
    observed_at = max(row.observed_at for row in observations)
    groups: dict[tuple[str, str, float], list[SensorObservation]] = {}
    for row in observations:
        groups.setdefault((row.asset_id, row.horizon, row.observed_at), []).append(row)
    candidates = [build_candidate(rows, config=config) for _, rows in sorted(groups.items())]
    candidates.sort(key=lambda row: (-int(row.eligible), -row.opportunity_score, row.asset_id, row.horizon))
    candidates = candidates[: config.max_candidates]

    max_virtual = config.total_virtual_equity * config.max_virtual_gross_fraction
    allocated = 0.0
    final: list[TournamentCandidate] = []
    for row in candidates:
        ticket = row.virtual_ticket_usd
        if row.eligible:
            ticket = min(ticket, max(0.0, max_virtual - allocated))
            allocated += ticket
        if not math.isclose(ticket, row.virtual_ticket_usd):
            row = TournamentCandidate(
                row.asset_id, row.horizon, row.observed_at, row.direction, row.opportunity_score,
                row.independent_groups, row.supporting_observation_ids, row.conflicting_observation_ids,
                ticket, row.eligible and ticket > 0, row.veto_reasons + (("virtual capital exhausted",) if ticket <= 0 else ()),
            )
        final.append(row)
    return TournamentRound(
        observed_at=observed_at,
        candidates=tuple(final),
        virtual_allocated_usd=allocated,
        virtual_unallocated_usd=max(0.0, config.total_virtual_equity - allocated),
    )
