from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from brian2026.global_sensor_mesh import SensorObservation, SensorTemplate

INTRABAR_REACTION_SCHEMA_VERSION = "brian.phase40-intrabar-reaction.v1"
INTRABAR_EXPERIMENT_ID = "phase40-intrabar-reaction-v1"


@dataclass(frozen=True, slots=True)
class IntrabarReactionConfig:
    cadence_seconds: int = 60
    scan_top_n: int = 50
    min_support_groups: int = 2
    min_consensus_score: float = 0.18
    overextension_sigma: float = 3.5
    fee_bps: float = 10.0
    slippage_bps: float = 1.0

    def __post_init__(self) -> None:
        if self.cadence_seconds < 30:
            raise ValueError("intrabar cadence below 30s is not preregistered")
        if not 1 <= self.scan_top_n <= 100:
            raise ValueError("scan_top_n must be in [1,100]")
        if self.min_support_groups < 2:
            raise ValueError("intrabar reaction requires independent confirmation")
        if not 0 <= self.min_consensus_score <= 1:
            raise ValueError("min_consensus_score must be in [0,1]")
        if self.overextension_sigma <= 0:
            raise ValueError("overextension_sigma must be positive")
        if min(self.fee_bps, self.slippage_bps) < 0:
            raise ValueError("trading costs cannot be negative")


@dataclass(frozen=True, slots=True)
class IntrabarConsensus:
    direction: int
    score: float
    support_groups: tuple[str, ...]
    conflict_groups: tuple[str, ...]
    eligible: bool
    late_chase: bool
    status: str


def intrabar_reaction_templates() -> tuple[SensorTemplate, ...]:
    """Preregistered fast specialists for the prospective intrabar shadow.

    These eyes are additive. They do not mutate the frozen Phase 3.7 learner,
    thresholds, learning state, or live-shadow portfolio experiment.
    """
    return (
        SensorTemplate("velocity-micro", "price_structure", "crypto", "MICRO_1_5M", "micro_velocity", "live_trade_velocity_5_60s", 3.0),
        SensorTemplate("volume-burst-micro", "price_structure", "crypto", "MICRO_1_5M", "micro_volume", "partial_1m_relative_volume", 3.0),
        SensorTemplate("breakout-micro", "price_structure", "crypto", "MICRO_1_5M", "micro_breakout", "preclose_1m_structure_break", 5.0),
        SensorTemplate("reclaim-micro", "price_structure", "crypto", "MICRO_1_5M", "micro_reclaim", "liquidity_sweep_reclaim", 3.0),
        SensorTemplate("taker-flow-micro", "orderbook", "crypto", "MICRO_1_5M", "micro_taker_flow", "partial_1m_taker_imbalance", 3.0),
    )


def build_intrabar_consensus(
    observations: Sequence[SensorObservation],
    *,
    config: IntrabarReactionConfig | None = None,
    extension_sigma: float = 0.0,
    decelerating: bool = False,
    fresh_velocity_direction: int = 0,
    taker_flow_direction: int = 0,
) -> IntrabarConsensus:
    cfg = config or IntrabarReactionConfig()
    by_group: dict[str, SensorObservation] = {}
    for row in observations:
        if not row.available or row.direction == 0:
            continue
        current = by_group.get(row.independent_group)
        quality = row.strength * row.confidence * row.reliability
        if current is None or quality > current.strength * current.confidence * current.reliability:
            by_group[row.independent_group] = row

    evidence = tuple(by_group.values())
    if not evidence:
        return IntrabarConsensus(0, 0.0, (), (), False, False, "WATCH")

    signed = [row.direction * row.strength * row.confidence * row.reliability for row in evidence]
    aggregate = sum(signed) / len(signed)
    direction = 1 if aggregate > 0 else -1 if aggregate < 0 else 0
    support = tuple(sorted(row.independent_group for row in evidence if row.direction == direction))
    conflicts = tuple(sorted(row.independent_group for row in evidence if row.direction != direction))
    support_ratio = len(support) / len(evidence) if evidence else 0.0
    breadth = 0.4 + 0.6 * min(1.0, len(evidence) / 3.0)
    score = max(0.0, min(1.0, abs(aggregate) * (0.5 + 0.5 * support_ratio) * breadth))
    eligible = direction != 0 and len(support) >= cfg.min_support_groups and score >= cfg.min_consensus_score

    flow_conflict = taker_flow_direction == -direction and taker_flow_direction != 0
    stale_velocity = fresh_velocity_direction not in (0, direction)
    late_chase = bool(
        eligible
        and extension_sigma >= cfg.overextension_sigma
        and (decelerating or flow_conflict or stale_velocity)
    )
    status = "ACTIONABLE_SHADOW" if eligible and not late_chase else "VETOED_LATE_CHASE" if late_chase else "WATCH"
    return IntrabarConsensus(direction, score, support, conflicts, eligible, late_chase, status)
