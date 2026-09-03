from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence
import math

import numpy as np

from .adaptive_quant import ChallengerVote, DriftAssessment, FamiliarityAssessment
from .portfolio import DEVELOPMENT_CUTOFF

Action = Literal["BUY", "SELL", "WAIT"]


@dataclass(frozen=True, slots=True)
class OpportunityConfig:
    min_abs_edge: float = 0.03
    drift_penalty: float = 0.50
    familiarity_floor: float = 0.35
    dispersion_penalty: float = 0.50

    def __post_init__(self) -> None:
        if not 0 <= self.min_abs_edge <= 1:
            raise ValueError("min_abs_edge must be in [0,1]")
        if not 0 < self.drift_penalty <= 1 or not 0 < self.familiarity_floor <= 1:
            raise ValueError("invalid opportunity penalties")
        if not 0 <= self.dispersion_penalty <= 1:
            raise ValueError("dispersion_penalty must be in [0,1]")


@dataclass(frozen=True, slots=True)
class OpportunityDecision:
    timestamp: float
    proposed_action: Action
    edge: float
    confidence: float
    directional_support: float
    dispersion: float
    opportunity_score: float
    familiarity: float
    out_of_distribution: bool
    drifted: bool
    hard_drift: bool
    veto_reasons: tuple[str, ...]
    shadow_only: bool = True
    schema_version: str = "brian.opportunity-score.v1"


def score_opportunity(
    timestamp: float,
    votes: Sequence[ChallengerVote],
    familiarity: FamiliarityAssessment,
    drift: DriftAssessment,
    config: OpportunityConfig = OpportunityConfig(),
) -> OpportunityDecision:
    """Soft committee score. It ranks evidence; it never sends an order.

    Unlike Phase 2.9's hard agreement gate, disagreement is a continuous penalty.
    OOD and hard drift remain hard vetoes. All inputs must be available at t.
    """
    if timestamp >= DEVELOPMENT_CUTOFF:
        raise ValueError("2026 data is INVALID_CONTAMINATED and forbidden")
    active = tuple(v for v in votes if v.validation_weight > 0)
    veto: list[str] = []
    if not active:
        veto.append("no validation-qualified challengers")
        return OpportunityDecision(timestamp, "WAIT", 0.0, 0.0, 0.0, 1.0, 0.0,
                                   familiarity.familiarity, familiarity.out_of_distribution,
                                   drift.drifted, drift.hard_drift, tuple(veto))

    total = sum(v.validation_weight for v in active)
    edge = sum(v.edge * v.validation_weight for v in active) / total
    confidence = sum(v.confidence * v.validation_weight for v in active) / total
    direction = 1 if edge > 0 else -1 if edge < 0 else 0

    directional = tuple(v for v in active if v.action != "WAIT")
    directional_total = sum(v.validation_weight for v in directional)
    support = 0.0
    if direction and directional_total > 0:
        support = sum(
            v.validation_weight for v in directional
            if (1 if v.edge > 0 else -1 if v.edge < 0 else 0) == direction
        ) / directional_total

    variance = sum(v.validation_weight * (v.edge - edge) ** 2 for v in active) / total
    dispersion = min(1.0, math.sqrt(max(0.0, variance)))
    familiarity_scale = max(config.familiarity_floor, familiarity.familiarity)
    drift_scale = config.drift_penalty if drift.drifted and not drift.hard_drift else 1.0
    disagreement_scale = max(0.0, 1.0 - config.dispersion_penalty * dispersion)
    score = abs(edge) * confidence * (0.50 + 0.50 * support) * familiarity_scale * drift_scale * disagreement_scale

    if familiarity.out_of_distribution:
        veto.append("market familiarity gate: out-of-distribution")
    if drift.hard_drift:
        veto.append("concept drift gate: hard distribution shift")
    proposed: Action = "WAIT"
    if not veto and abs(edge) >= config.min_abs_edge:
        proposed = "BUY" if edge > 0 else "SELL"

    return OpportunityDecision(
        float(timestamp), proposed, float(max(-1.0, min(1.0, edge))),
        float(max(0.0, min(1.0, confidence))), float(max(0.0, min(1.0, support))),
        float(dispersion), float(max(0.0, min(1.0, score))),
        float(familiarity.familiarity), bool(familiarity.out_of_distribution),
        bool(drift.drifted), bool(drift.hard_drift), tuple(veto), True,
    )


@dataclass(frozen=True, slots=True)
class SelectionReceipt:
    threshold: float
    quantile: float
    eligible: int
    observations: int
    selection_partition: str = "policy_validation_only"


def select_score_threshold(
    scores: Sequence[float],
    proposed_actions: Sequence[Action],
    quantile: float,
) -> SelectionReceipt:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0,1)")
    if len(scores) != len(proposed_actions) or not scores:
        raise ValueError("scores/actions must be non-empty and aligned")
    eligible_scores = np.asarray([
        float(score) for score, action in zip(scores, proposed_actions)
        if action != "WAIT" and math.isfinite(float(score))
    ], dtype=float)
    if not len(eligible_scores):
        return SelectionReceipt(float("inf"), quantile, 0, len(scores))
    threshold = float(np.quantile(eligible_scores, quantile))
    eligible = sum(
        action != "WAIT" and math.isfinite(float(score)) and float(score) >= threshold
        for score, action in zip(scores, proposed_actions)
    )
    return SelectionReceipt(threshold, quantile, eligible, len(scores))


def apply_score_threshold(decisions: Sequence[OpportunityDecision], threshold: float) -> list[Action]:
    if math.isnan(float(threshold)):
        raise ValueError("threshold cannot be NaN")
    return [
        row.proposed_action if row.proposed_action != "WAIT" and row.opportunity_score >= threshold else "WAIT"
        for row in decisions
    ]


@dataclass(frozen=True, slots=True)
class CapitalPolicy:
    starting_equity: float
    equity_fraction: float
    ruin_probability: float
    validation_net_pnl: float
    validation_profit_factor: float | None
    validation_entries: int
    validation_max_drawdown_pct: float
    deployable: bool
    reason: str
    schema_version: str = "brian.capital-policy.v1"

    def manifest(self) -> dict:
        return asdict(self)


def bootstrap_risk_of_ruin(
    trade_return_fractions: Sequence[float],
    equity_fraction: float,
    *,
    seed: int,
    trials: int = 2000,
    horizon_trades: int = 200,
    ruin_drawdown_fraction: float = 0.30,
) -> float:
    if not 0 <= equity_fraction <= 1:
        raise ValueError("equity_fraction must be in [0,1]")
    if not 0 < ruin_drawdown_fraction < 1 or trials <= 0 or horizon_trades <= 0:
        raise ValueError("invalid ruin simulation settings")
    returns = np.asarray([float(x) for x in trade_return_fractions if math.isfinite(float(x))], dtype=float)
    if equity_fraction == 0:
        return 0.0
    if not len(returns):
        return 1.0
    rng = np.random.default_rng(seed)
    ruined = 0
    for _ in range(trials):
        equity = peak = 1.0
        for r in rng.choice(returns, size=horizon_trades, replace=True):
            equity *= max(0.0, 1.0 + equity_fraction * float(r))
            peak = max(peak, equity)
            if equity <= peak * (1.0 - ruin_drawdown_fraction):
                ruined += 1
                break
    return ruined / trials


def choose_capital_policy(
    candidates: Sequence[dict],
    *,
    starting_equity: float = 500.0,
    max_ruin_probability: float = 0.05,
    min_entries: int = 20,
    min_profit_factor: float = 1.05,
    max_drawdown_pct: float = 10.0,
) -> CapitalPolicy:
    """Choose only among precomputed policy-validation candidates.

    A candidate must contain fraction, ruin_probability, net_pnl, profit_factor,
    entries and max_drawdown_pct. No test result belongs here.
    """
    if starting_equity <= 0:
        raise ValueError("starting_equity must be positive")
    valid = []
    for row in candidates:
        fraction = float(row["fraction"])
        ruin = float(row["ruin_probability"])
        pnl = float(row["net_pnl"])
        entries = int(row["entries"])
        drawdown = float(row["max_drawdown_pct"])
        pf = row.get("profit_factor")
        pf_value = float(pf) if pf is not None and math.isfinite(float(pf)) else 0.0
        if (
            0 < fraction <= 1 and pnl > 0 and entries >= min_entries and
            pf_value >= min_profit_factor and drawdown <= max_drawdown_pct and
            ruin <= max_ruin_probability
        ):
            valid.append((pnl, -ruin, -drawdown, -fraction, row))
    if not valid:
        return CapitalPolicy(starting_equity, 0.0, 0.0, 0.0, None, 0, 0.0, False,
                             "validation edge/risk gates not met; capital remains flat")
    row = max(valid, key=lambda item: item[:4])[-1]
    return CapitalPolicy(
        starting_equity, float(row["fraction"]), float(row["ruin_probability"]),
        float(row["net_pnl"]), float(row["profit_factor"]), int(row["entries"]),
        float(row["max_drawdown_pct"]), True,
        "selected on policy-validation only; no test tuning",
    )
