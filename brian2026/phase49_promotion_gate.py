from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Literal, Sequence
import math

from .phase41_robustness_lab import RobustnessReport
from .phase42_autonomous_alpha_lab import AlphaExperimentOutcome
from .phase48_causality_auditor import CausalityPromotionGate

PHASE49_SCHEMA_VERSION = "brian.phase49-promotion-gate.v1"
PromotionStatus = Literal[
    "RESEARCH_BLOCKED",
    "SHADOW_ONLY",
    "PAPER_PARITY_PENDING",
    "MICRO_LIVE_ELIGIBLE",
]


def _percentile(values: Sequence[float], q: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("percentile requires observations")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0,1]")
    if len(rows) == 1:
        return rows[0]
    position = (len(rows) - 1) * q
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return rows[left]
    weight = position - left
    return rows[left] * (1.0 - weight) + rows[right] * weight


@dataclass(frozen=True, slots=True)
class ShadowPaperObservation:
    intent_id: str
    shadow_direction: int
    paper_direction: int
    reference_price: float
    paper_fill_price: float | None
    shadow_fill_fraction: float
    paper_fill_fraction: float
    paper_acknowledged: bool
    reconciliation_complete: bool
    ambiguous_outcome: bool
    observed_at: float

    def __post_init__(self) -> None:
        if not self.intent_id.strip():
            raise ValueError("intent_id is required")
        if self.shadow_direction not in (-1, 0, 1) or self.paper_direction not in (-1, 0, 1):
            raise ValueError("directions must be -1, 0 or 1")
        if not math.isfinite(self.reference_price) or self.reference_price <= 0:
            raise ValueError("reference_price must be positive")
        if self.paper_fill_price is not None and (
            not math.isfinite(self.paper_fill_price) or self.paper_fill_price <= 0
        ):
            raise ValueError("paper_fill_price must be positive when present")
        for label, value in (
            ("shadow_fill_fraction", self.shadow_fill_fraction),
            ("paper_fill_fraction", self.paper_fill_fraction),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{label} must be in [0,1]")
        if not math.isfinite(self.observed_at):
            raise ValueError("observed_at must be finite")

    @property
    def direction_matches(self) -> bool:
        return self.shadow_direction == self.paper_direction

    @property
    def execution_drift_bps(self) -> float | None:
        if self.paper_fill_price is None or self.paper_fill_fraction <= 0:
            return None
        if self.shadow_direction == 0:
            return 0.0
        if self.shadow_direction > 0:
            return (self.paper_fill_price / self.reference_price - 1.0) * 10_000.0
        return (self.reference_price / self.paper_fill_price - 1.0) * 10_000.0


@dataclass(frozen=True, slots=True)
class PaperParityPolicy:
    min_observations: int = 30
    min_direction_match_rate: float = 0.98
    min_acknowledgement_rate: float = 0.99
    min_reconciliation_rate: float = 1.0
    min_mean_fill_fraction_ratio: float = 0.90
    max_p95_adverse_execution_drift_bps: float = 12.0
    max_ambiguous_outcomes: int = 0

    def __post_init__(self) -> None:
        if self.min_observations < 5:
            raise ValueError("min_observations must be at least 5")
        for label, value in (
            ("min_direction_match_rate", self.min_direction_match_rate),
            ("min_acknowledgement_rate", self.min_acknowledgement_rate),
            ("min_reconciliation_rate", self.min_reconciliation_rate),
            ("min_mean_fill_fraction_ratio", self.min_mean_fill_fraction_ratio),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{label} must be in [0,1]")
        if self.max_p95_adverse_execution_drift_bps < 0:
            raise ValueError("execution drift threshold must be non-negative")
        if self.max_ambiguous_outcomes < 0:
            raise ValueError("max_ambiguous_outcomes must be non-negative")


@dataclass(frozen=True, slots=True)
class PaperParityReport:
    observations: int
    direction_match_rate: float
    acknowledgement_rate: float
    reconciliation_rate: float
    mean_shadow_fill_fraction: float
    mean_paper_fill_fraction: float
    mean_fill_fraction_ratio: float
    median_execution_drift_bps: float | None
    p95_adverse_execution_drift_bps: float | None
    ambiguous_outcomes: int
    checks: tuple[tuple[str, bool], ...]
    status: str
    schema_version: str = PHASE49_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "observations": self.observations,
            "direction_match_rate": self.direction_match_rate,
            "acknowledgement_rate": self.acknowledgement_rate,
            "reconciliation_rate": self.reconciliation_rate,
            "mean_shadow_fill_fraction": self.mean_shadow_fill_fraction,
            "mean_paper_fill_fraction": self.mean_paper_fill_fraction,
            "mean_fill_fraction_ratio": self.mean_fill_fraction_ratio,
            "median_execution_drift_bps": self.median_execution_drift_bps,
            "p95_adverse_execution_drift_bps": self.p95_adverse_execution_drift_bps,
            "ambiguous_outcomes": self.ambiguous_outcomes,
            "checks": dict(self.checks),
            "status": self.status,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }


def evaluate_shadow_paper_parity(
    observations: Sequence[ShadowPaperObservation],
    *,
    policy: PaperParityPolicy = PaperParityPolicy(),
) -> PaperParityReport:
    """Evaluate sandbox/paper parity before any real-money eligibility.

    The gate follows production execution-system behavior rather than PnL:
    state must reconcile, acknowledgements must be observed, ambiguous outcomes
    must be resolved, and simulated-vs-paper direction/fill behavior must remain
    within a preregistered tolerance.
    """
    rows = tuple(observations)
    if not rows:
        raise ValueError("paper parity requires observations")
    ids = [row.intent_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("paper parity observations require unique intent ids")

    n = len(rows)
    direction_match_rate = sum(row.direction_matches for row in rows) / n
    acknowledgement_rate = sum(row.paper_acknowledged for row in rows) / n
    reconciliation_rate = sum(row.reconciliation_complete for row in rows) / n
    ambiguous = sum(row.ambiguous_outcome for row in rows)

    shadow_fill = sum(row.shadow_fill_fraction for row in rows) / n
    paper_fill = sum(row.paper_fill_fraction for row in rows) / n
    if shadow_fill <= 1e-12:
        fill_ratio = 1.0 if paper_fill <= 1e-12 else 0.0
    else:
        fill_ratio = min(1.0, paper_fill / shadow_fill)

    drift = tuple(
        value
        for row in rows
        if (value := row.execution_drift_bps) is not None
    )
    median_drift = median(drift) if drift else None
    # Only adverse positive drift matters for the promotion threshold.
    adverse = tuple(max(0.0, value) for value in drift)
    p95_adverse = _percentile(adverse, 0.95) if adverse else None

    checks = (
        ("minimum_observations", n >= policy.min_observations),
        ("direction_parity", direction_match_rate >= policy.min_direction_match_rate),
        ("acknowledgements_complete_enough", acknowledgement_rate >= policy.min_acknowledgement_rate),
        ("reconciliation_complete_enough", reconciliation_rate >= policy.min_reconciliation_rate),
        ("fill_fraction_parity", fill_ratio >= policy.min_mean_fill_fraction_ratio),
        (
            "execution_drift_within_limit",
            p95_adverse is not None and p95_adverse <= policy.max_p95_adverse_execution_drift_bps,
        ),
        ("no_unresolved_ambiguous_outcomes", ambiguous <= policy.max_ambiguous_outcomes),
    )
    return PaperParityReport(
        observations=n,
        direction_match_rate=direction_match_rate,
        acknowledgement_rate=acknowledgement_rate,
        reconciliation_rate=reconciliation_rate,
        mean_shadow_fill_fraction=shadow_fill,
        mean_paper_fill_fraction=paper_fill,
        mean_fill_fraction_ratio=fill_ratio,
        median_execution_drift_bps=median_drift,
        p95_adverse_execution_drift_bps=p95_adverse,
        ambiguous_outcomes=ambiguous,
        checks=checks,
        status="PASS_PAPER_PARITY" if all(value for _, value in checks) else "FAIL_PAPER_PARITY",
    )


@dataclass(frozen=True, slots=True)
class PromotionReceipt:
    status: PromotionStatus
    candidate_id: str
    checks: tuple[tuple[str, bool], ...]
    reasons: tuple[str, ...]
    research_experiment_id: str
    paper_observations: int
    human_authorization_required: bool = True
    exchange_adapter_enabled: bool = False
    automatic_activation: bool = False
    shadow_only: bool = True
    live_execution: bool = False
    schema_version: str = PHASE49_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_micro_live_eligibility(
    research: AlphaExperimentOutcome,
    robustness: RobustnessReport,
    causality: CausalityPromotionGate,
    paper_parity: PaperParityReport | None,
) -> PromotionReceipt:
    """Aggregate independent gates without enabling live execution.

    MICRO_LIVE_ELIGIBLE means only that the research artifact passed all
    preregistered pre-live checks. It does not enable an exchange adapter,
    authorize capital, or submit an order. An explicit later authorization
    boundary is mandatory.
    """
    checks = (
        (
            "research_challenger_passed",
            research.decision == "RESEARCH_CHALLENGER_CANDIDATE"
            and all(value for _, value in research.checks),
        ),
        (
            "robustness_passed",
            robustness.status == "ROBUSTNESS_CANDIDATE"
            and all(value for _, value in robustness.checks),
        ),
        ("causality_passed", bool(causality.eligible)),
        (
            "paper_parity_passed",
            paper_parity is not None
            and paper_parity.status == "PASS_PAPER_PARITY"
            and all(value for _, value in paper_parity.checks),
        ),
    )
    reasons = tuple(name for name, passed in checks if not passed)

    if not checks[0][1] or not checks[1][1] or not checks[2][1]:
        status: PromotionStatus = "RESEARCH_BLOCKED"
    elif paper_parity is None:
        status = "PAPER_PARITY_PENDING"
    elif not checks[3][1]:
        status = "SHADOW_ONLY"
    else:
        status = "MICRO_LIVE_ELIGIBLE"

    return PromotionReceipt(
        status=status,
        candidate_id=research.candidate_id,
        checks=checks,
        reasons=reasons,
        research_experiment_id=research.experiment_id,
        paper_observations=0 if paper_parity is None else paper_parity.observations,
    )
