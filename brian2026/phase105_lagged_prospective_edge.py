from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

PHASE105_SCHEMA_VERSION = "brian.phase105-lagged-prospective-edge.v1"
UPSTREAM_EDGE_MODEL_VERSION = "brian.alpha-intelligence-challenger.v1"

Recommendation = Literal[
    "ALLOW_EDGE",
    "DOWNGRADE_TO_WAIT",
    "COST_UNAVAILABLE",
    "INSUFFICIENT_LAGGED_EVIDENCE",
    "CONTAMINATED_EVIDENCE",
]

PRIOR_RELIABILITY = 0.50
SHRINK_SAMPLES = 200
MAX_RELIABILITY_DEVIATION = 0.15
MIN_MATURE_SAMPLES = 100
MIN_MATURE_GROUPS = 2
GROSS_CAP_BPS = 75.0
UNCERTAINTY_FLOOR_BPS = 1.5
IMMATURITY_PENALTY_BPS = 8.0


class LaggedProspectiveEdgeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LaggedReliabilityEvidence:
    group: str
    sample_count: int
    bayesian_hit_rate: float
    avg_signed_bps: float
    avg_cost_adjusted_signed_bps: float
    outcome_horizon_seconds: int
    snapshot_window_end: float
    snapshot_generated_at: float
    evidence_class: str = "PROSPECTIVE_DEVELOPMENT_SHADOW"
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.group.strip():
            raise ValueError("group is required")
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if not math.isfinite(self.bayesian_hit_rate) or not 0 <= self.bayesian_hit_rate <= 1:
            raise ValueError("bayesian_hit_rate must be in [0,1]")
        for label, value in (
            ("avg_signed_bps", self.avg_signed_bps),
            ("avg_cost_adjusted_signed_bps", self.avg_cost_adjusted_signed_bps),
            ("snapshot_window_end", self.snapshot_window_end),
            ("snapshot_generated_at", self.snapshot_generated_at),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        if self.outcome_horizon_seconds <= 0:
            raise ValueError("outcome_horizon_seconds must be positive")
        if self.evidence_class != "PROSPECTIVE_DEVELOPMENT_SHADOW":
            raise ValueError("reliability evidence must be prospective shadow evidence")
        if not self.shadow_only or self.live_execution:
            raise ValueError("reliability evidence must remain shadow-only")


@dataclass(frozen=True, slots=True)
class EvidenceFreshness:
    group: str
    observed_at: float
    horizon: str

    def __post_init__(self) -> None:
        if not self.group.strip():
            raise ValueError("freshness group is required")
        if not math.isfinite(float(self.observed_at)):
            raise ValueError("observed_at must be finite")
        if not self.horizon.strip():
            raise ValueError("horizon is required")


@dataclass(frozen=True, slots=True)
class EdgeContribution:
    group: str
    samples: int
    measured_reliability: float
    bounded_reliability: float
    maturity: float
    historical_gross_signed_bps: float
    historical_after_cost_signed_bps: float
    contribution_weight: float
    contribution_gross_bps: float
    outcome_horizon_seconds: int
    snapshot_window_end: float
    snapshot_generated_at: float


@dataclass(frozen=True, slots=True)
class LaggedExpectedEdgeEstimate:
    decision_timestamp: float
    direction: int
    evidence_score: float
    expected_gross_move_bps: float | None
    estimated_round_trip_cost_bps: float | None
    uncertainty_penalty_bps: float | None
    event_decay_penalty_bps: float | None
    expected_net_edge_bps: float | None
    minimum_net_margin_bps: float
    eligible: bool
    recommendation: Recommendation
    mature_group_count: int
    contributions: tuple[EdgeContribution, ...]
    reliability_weights: tuple[tuple[str, float], ...]
    pit_clear: bool
    reasons: tuple[str, ...]
    model_version: str = UPSTREAM_EDGE_MODEL_VERSION
    schema_version: str = PHASE105_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False
    automatic_promotion: bool = False

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        if not math.isfinite(float(self.decision_timestamp)):
            raise ValueError("decision_timestamp must be finite")
        if not math.isfinite(float(self.evidence_score)) or not 0 <= self.evidence_score <= 1:
            raise ValueError("evidence_score must be in [0,1]")
        if self.minimum_net_margin_bps < 0:
            raise ValueError("minimum_net_margin_bps must be non-negative")
        if self.eligible:
            if self.recommendation != "ALLOW_EDGE":
                raise ValueError("eligible estimate must use ALLOW_EDGE")
            if not self.pit_clear or self.expected_net_edge_bps is None:
                raise ValueError("eligible estimate requires PIT-clear net edge")
        if not self.shadow_only or self.live_execution or self.automatic_promotion:
            raise ValueError("Phase105 must remain hard shadow-only")


def bounded_prospective_reliability(sample_count: int, measured_reliability: float) -> float:
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if not math.isfinite(measured_reliability) or not 0 <= measured_reliability <= 1:
        raise ValueError("measured_reliability must be in [0,1]")
    maturity = sample_count / (sample_count + SHRINK_SAMPLES)
    raw = PRIOR_RELIABILITY + (measured_reliability - PRIOR_RELIABILITY) * maturity
    return max(
        PRIOR_RELIABILITY - MAX_RELIABILITY_DEVIATION,
        min(PRIOR_RELIABILITY + MAX_RELIABILITY_DEVIATION, raw),
    )


def freshness_budget_seconds(horizon: str) -> float | None:
    return {
        "MICRO_1_5M": 5 * 60.0,
        "FAST_5_30M": 30 * 60.0,
        "EVENT_DRIVEN": 60 * 60.0,
        "INTRADAY_30M_6H": 6 * 60 * 60.0,
        "DAILY": 36 * 60 * 60.0,
        "SWING_6H_7D": 7 * 24 * 60 * 60.0,
        "MACRO_1D_PLUS": 7 * 24 * 60 * 60.0,
    }.get(horizon)


def _weighted_mean(values: Sequence[tuple[float, float]]) -> float | None:
    total = sum(weight for _, weight in values)
    if total <= 0:
        return None
    return sum(value * weight for value, weight in values) / total


def _weighted_mean_absolute_deviation(
    values: Sequence[tuple[float, float]],
    mean: float,
) -> float:
    total = sum(weight for _, weight in values) or 1.0
    return sum(abs(value - mean) * weight for value, weight in values) / total


def estimate_lagged_expected_edge(
    *,
    decision_timestamp: float,
    direction: int,
    evidence_score: float,
    round_trip_cost_bps: float | None,
    reliability: Sequence[LaggedReliabilityEvidence],
    freshness: Sequence[EvidenceFreshness],
    minimum_net_margin_bps: float = 2.0,
) -> LaggedExpectedEdgeEstimate:
    """Python runtime port of the proven ALPHA expected-edge challenger semantics.

    avg_signed_bps is already aligned to each sensor's historical direction.
    It is intentionally *not* multiplied by the current trade direction again.
    Only reliability snapshots generated/windowed no later than the decision
    timestamp may contribute.
    """
    timestamp = float(decision_timestamp)
    if not math.isfinite(timestamp):
        raise ValueError("decision_timestamp must be finite")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    score = float(evidence_score)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("evidence_score must be in [0,1]")
    margin = float(minimum_net_margin_bps)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("minimum_net_margin_bps must be finite and non-negative")

    if round_trip_cost_bps is None:
        return LaggedExpectedEdgeEstimate(
            decision_timestamp=timestamp,
            direction=direction,
            evidence_score=score,
            expected_gross_move_bps=None,
            estimated_round_trip_cost_bps=None,
            uncertainty_penalty_bps=None,
            event_decay_penalty_bps=None,
            expected_net_edge_bps=None,
            minimum_net_margin_bps=margin,
            eligible=False,
            recommendation="COST_UNAVAILABLE",
            mature_group_count=0,
            contributions=(),
            reliability_weights=(),
            pit_clear=True,
            reasons=("decision-time round-trip cost is unavailable",),
        )
    cost = float(round_trip_cost_bps)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("round_trip_cost_bps must be finite and non-negative")

    reasons: list[str] = []
    pit_clear = True
    by_group: dict[str, LaggedReliabilityEvidence] = {}
    for row in reliability:
        if (
            row.snapshot_window_end > timestamp
            or row.snapshot_generated_at > timestamp
        ):
            pit_clear = False
            reasons.append(f"post-decision reliability rejected:{row.group}")
            continue
        prior = by_group.get(row.group)
        if prior is None or (
            row.sample_count,
            row.snapshot_generated_at,
            row.snapshot_window_end,
        ) > (
            prior.sample_count,
            prior.snapshot_generated_at,
            prior.snapshot_window_end,
        ):
            by_group[row.group] = row

    contributions: list[EdgeContribution] = []
    for group in sorted(by_group):
        row = by_group[group]
        maturity = max(
            0.0,
            min(1.0, row.sample_count / (row.sample_count + SHRINK_SAMPLES)),
        )
        bounded = bounded_prospective_reliability(
            row.sample_count,
            row.bayesian_hit_rate,
        )
        reliability_scale = max(0.7, min(1.3, bounded / PRIOR_RELIABILITY))
        gross = max(-GROSS_CAP_BPS, min(GROSS_CAP_BPS, row.avg_signed_bps))
        after_cost = max(
            -GROSS_CAP_BPS,
            min(GROSS_CAP_BPS, row.avg_cost_adjusted_signed_bps),
        )
        weight = maturity * reliability_scale
        contributions.append(EdgeContribution(
            group=group,
            samples=row.sample_count,
            measured_reliability=row.bayesian_hit_rate,
            bounded_reliability=bounded,
            maturity=maturity,
            historical_gross_signed_bps=gross,
            historical_after_cost_signed_bps=after_cost,
            contribution_weight=weight,
            contribution_gross_bps=gross * weight,
            outcome_horizon_seconds=row.outcome_horizon_seconds,
            snapshot_window_end=row.snapshot_window_end,
            snapshot_generated_at=row.snapshot_generated_at,
        ))

    mature = tuple(
        row for row in contributions
        if row.samples >= MIN_MATURE_SAMPLES
    )
    weights = tuple(
        (row.group, row.bounded_reliability)
        for row in contributions
    )
    if len(mature) < MIN_MATURE_GROUPS:
        reasons.append(
            f"only {len(mature)} mature independent groups; "
            f"{MIN_MATURE_GROUPS} required"
        )
        return LaggedExpectedEdgeEstimate(
            decision_timestamp=timestamp,
            direction=direction,
            evidence_score=score,
            expected_gross_move_bps=None,
            estimated_round_trip_cost_bps=cost,
            uncertainty_penalty_bps=None,
            event_decay_penalty_bps=None,
            expected_net_edge_bps=None,
            minimum_net_margin_bps=margin,
            eligible=False,
            recommendation=(
                "INSUFFICIENT_LAGGED_EVIDENCE"
                if pit_clear
                else "CONTAMINATED_EVIDENCE"
            ),
            mature_group_count=len(mature),
            contributions=tuple(contributions),
            reliability_weights=weights,
            pit_clear=pit_clear,
            reasons=tuple(reasons),
        )

    weighted = tuple(
        (row.historical_gross_signed_bps, row.contribution_weight)
        for row in mature
    )
    gross = _weighted_mean(weighted)
    if gross is None:
        raise LaggedProspectiveEdgeError(
            "mature expected-edge contributions have zero total weight"
        )
    dispersion = _weighted_mean_absolute_deviation(weighted, gross)
    avg_maturity = sum(row.maturity for row in mature) / len(mature)
    uncertainty = (
        UNCERTAINTY_FLOOR_BPS
        + 0.5 * dispersion
        + (1.0 - avg_maturity) * IMMATURITY_PENALTY_BPS
    )

    fractions: list[float] = []
    for row in freshness:
        budget = freshness_budget_seconds(row.horizon)
        if budget is None:
            fractions.append(1.0)
            reasons.append(f"unknown freshness for {row.group}")
            continue
        if row.observed_at > timestamp:
            pit_clear = False
            reasons.append(f"future observation detected for {row.group}")
            continue
        age = max(0.0, timestamp - row.observed_at)
        fractions.append(max(0.0, min(1.0, age / budget)))
    if not freshness:
        freshness_fraction = 0.25
        reasons.append(
            "source-observation freshness unavailable; conservative decay prior applied"
        )
    else:
        freshness_fraction = (
            sum(fractions) / len(fractions)
            if fractions
            else 1.0
        )
    decay = max(0.0, gross) * 0.25 * freshness_fraction
    net = gross - cost - uncertainty - decay
    eligible = pit_clear and net > margin

    if gross <= 0:
        reasons.append(
            "lagged prospective evidence implies non-positive gross directional edge"
        )
    if net <= margin:
        reasons.append(
            f"expected net edge {net:.2f} bps does not clear "
            f"{margin:.2f} bps margin"
        )
    if eligible:
        reasons.append(
            f"expected net edge clears margin with "
            f"{len(mature)} mature independent groups"
        )

    return LaggedExpectedEdgeEstimate(
        decision_timestamp=timestamp,
        direction=direction,
        evidence_score=score,
        expected_gross_move_bps=gross,
        estimated_round_trip_cost_bps=cost,
        uncertainty_penalty_bps=uncertainty,
        event_decay_penalty_bps=decay,
        expected_net_edge_bps=net,
        minimum_net_margin_bps=margin,
        eligible=eligible,
        recommendation=(
            "CONTAMINATED_EVIDENCE"
            if not pit_clear
            else "ALLOW_EDGE"
            if eligible
            else "DOWNGRADE_TO_WAIT"
        ),
        mature_group_count=len(mature),
        contributions=tuple(contributions),
        reliability_weights=weights,
        pit_clear=pit_clear,
        reasons=tuple(reasons),
    )


def eligible_expected_edge_bps_by_asset(
    estimates: Mapping[str, LaggedExpectedEdgeEstimate],
) -> dict[str, float]:
    """Return only PIT-clear, eligible *net* edges for new-risk compilation.

    Missing assets remain missing on purpose. Phase55 then fails closed for a
    new/increasing-risk leg instead of inventing an edge. Risk-reduction legs do
    not require alpha edge and continue to use their existing reduce-only path.
    """
    result: dict[str, float] = {}
    for raw_asset, estimate in estimates.items():
        asset = str(raw_asset).strip()
        if not asset:
            raise ValueError("edge estimate asset id cannot be blank")
        if not estimate.eligible:
            continue
        if (
            estimate.recommendation != "ALLOW_EDGE"
            or not estimate.pit_clear
            or estimate.expected_net_edge_bps is None
            or not math.isfinite(estimate.expected_net_edge_bps)
        ):
            raise LaggedProspectiveEdgeError(
                f"{asset} has inconsistent eligible edge estimate"
            )
        result[asset] = float(estimate.expected_net_edge_bps)
    return result
