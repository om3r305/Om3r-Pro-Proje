from __future__ import annotations

from dataclasses import asdict, dataclass
from random import Random
from statistics import fmean
from typing import Sequence
import math

PHASE41_SCHEMA_VERSION = "brian.phase41-robustness-lab.v1"


def _finite_series(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    rows = tuple(float(value) for value in values)
    if not rows:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in rows):
        raise ValueError(f"{name} must contain only finite values")
    return rows


def _percentile(values: Sequence[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0,1]")
    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("percentile requires observations")
    if len(rows) == 1:
        return rows[0]
    position = (len(rows) - 1) * q
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return rows[left]
    weight = position - left
    return rows[left] * (1.0 - weight) + rows[right] * weight


def _path_metrics(pnls: Sequence[float], *, starting_equity: float, ruin_fraction: float) -> tuple[float, float, bool]:
    if starting_equity <= 0:
        raise ValueError("starting_equity must be positive")
    if not 0.0 < ruin_fraction < 1.0:
        raise ValueError("ruin_fraction must be in (0,1)")
    equity = peak = float(starting_equity)
    max_drawdown_pct = 0.0
    ruined = False
    ruin_level = starting_equity * ruin_fraction
    for pnl in pnls:
        equity += float(pnl)
        peak = max(peak, equity)
        max_drawdown_pct = max(
            max_drawdown_pct,
            100.0 * max(0.0, peak - equity) / max(peak, 1e-12),
        )
        ruined = ruined or equity <= ruin_level
    return (equity / starting_equity - 1.0) * 100.0, max_drawdown_pct, ruined


@dataclass(frozen=True, slots=True)
class MonteCarloSummary:
    method: str
    trials: int
    seed: int
    median_return_pct: float
    p05_return_pct: float
    p95_return_pct: float
    median_max_drawdown_pct: float
    p95_max_drawdown_pct: float
    loss_probability: float
    ruin_probability: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuleSignificanceResult:
    trials: int
    seed: int
    active_count: int
    opportunity_count: int
    observed_mean_return: float
    random_mean_return: float
    lift: float
    p_value: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RobustnessPolicy:
    min_trials: int = 250
    min_block_p05_return_pct: float = 0.0
    max_trade_order_p95_drawdown_pct: float = 15.0
    max_block_loss_probability: float = 0.25
    max_ruin_probability: float = 0.01
    max_rule_p_value: float = 0.05

    def __post_init__(self) -> None:
        if self.min_trials < 50:
            raise ValueError("min_trials must be at least 50")
        if self.max_trade_order_p95_drawdown_pct < 0:
            raise ValueError("drawdown limit must be non-negative")
        for value in (self.max_block_loss_probability, self.max_ruin_probability, self.max_rule_p_value):
            if not 0.0 <= value <= 1.0:
                raise ValueError("probability thresholds must be in [0,1]")


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    trade_order: MonteCarloSummary
    block_bootstrap: MonteCarloSummary
    rule_significance: RuleSignificanceResult
    checks: tuple[tuple[str, bool], ...]
    status: str
    schema_version: str = PHASE41_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False
    automatic_promotion: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trade_order": self.trade_order.to_dict(),
            "block_bootstrap": self.block_bootstrap.to_dict(),
            "rule_significance": self.rule_significance.to_dict(),
            "checks": dict(self.checks),
            "status": self.status,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
            "automatic_promotion": self.automatic_promotion,
        }


def _summary(method: str, paths: Sequence[tuple[float, float, bool]], *, trials: int, seed: int) -> MonteCarloSummary:
    returns = tuple(row[0] for row in paths)
    drawdowns = tuple(row[1] for row in paths)
    return MonteCarloSummary(
        method=method,
        trials=trials,
        seed=seed,
        median_return_pct=_percentile(returns, 0.50),
        p05_return_pct=_percentile(returns, 0.05),
        p95_return_pct=_percentile(returns, 0.95),
        median_max_drawdown_pct=_percentile(drawdowns, 0.50),
        p95_max_drawdown_pct=_percentile(drawdowns, 0.95),
        loss_probability=sum(value < 0.0 for value in returns) / len(returns),
        ruin_probability=sum(row[2] for row in paths) / len(paths),
    )


def trade_order_monte_carlo(
    pnls: Sequence[float],
    *,
    trials: int = 1000,
    seed: int = 4101,
    starting_equity: float = 10_000.0,
    ruin_fraction: float = 0.50,
) -> MonteCarloSummary:
    """Shuffle the realized trade order without changing the trade population.

    This mirrors the trade-order Monte Carlo idea used by mature trading
    research stacks: total PnL is fixed, but path-dependent drawdown and ruin
    behavior are stress-tested under alternate orderings.
    """
    rows = _finite_series(pnls, name="pnls")
    if trials < 50:
        raise ValueError("trials must be at least 50")
    rng = Random(seed)
    paths: list[tuple[float, float, bool]] = []
    for _ in range(trials):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        paths.append(_path_metrics(shuffled, starting_equity=starting_equity, ruin_fraction=ruin_fraction))
    return _summary("trade_order_shuffle", paths, trials=trials, seed=seed)


def block_bootstrap_monte_carlo(
    pnls: Sequence[float],
    *,
    block_size: int = 5,
    trials: int = 1000,
    seed: int = 4102,
    starting_equity: float = 10_000.0,
    ruin_fraction: float = 0.50,
) -> MonteCarloSummary:
    """Resample contiguous PnL blocks to preserve short-run dependence.

    Blocks, rather than individual trades, are sampled with replacement. This
    produces alternate but locally coherent paths and avoids pretending that
    adjacent trading outcomes are independent.
    """
    rows = _finite_series(pnls, name="pnls")
    if trials < 50:
        raise ValueError("trials must be at least 50")
    if block_size <= 0 or block_size > len(rows):
        raise ValueError("block_size must be in [1, len(pnls)]")
    starts = tuple(range(0, len(rows) - block_size + 1))
    rng = Random(seed)
    paths: list[tuple[float, float, bool]] = []
    for _ in range(trials):
        sampled: list[float] = []
        while len(sampled) < len(rows):
            start = starts[rng.randrange(len(starts))]
            sampled.extend(rows[start:start + block_size])
        sampled = sampled[:len(rows)]
        paths.append(_path_metrics(sampled, starting_equity=starting_equity, ruin_fraction=ruin_fraction))
    return _summary("contiguous_block_bootstrap", paths, trials=trials, seed=seed)


def rule_significance_test(
    opportunity_returns: Sequence[float],
    active_mask: Sequence[bool],
    *,
    trials: int = 2000,
    seed: int = 4103,
) -> RuleSignificanceResult:
    """Compare the observed rule entries against same-count random entries.

    Inputs must come from an already locked evaluation partition. The function
    never tunes a threshold and never mutates the rule; it only asks whether the
    rule's selected opportunities beat random selection on the same market path.
    """
    returns = _finite_series(opportunity_returns, name="opportunity_returns")
    mask = tuple(bool(value) for value in active_mask)
    if len(mask) != len(returns):
        raise ValueError("active_mask must match opportunity_returns")
    active = tuple(index for index, enabled in enumerate(mask) if enabled)
    if len(active) < 2:
        raise ValueError("rule significance requires at least two active opportunities")
    if len(active) >= len(returns):
        raise ValueError("rule significance requires inactive opportunities for a random baseline")
    if trials < 100:
        raise ValueError("trials must be at least 100")

    observed = fmean(returns[index] for index in active)
    rng = Random(seed)
    population = tuple(range(len(returns)))
    random_means: list[float] = []
    for _ in range(trials):
        indices = rng.sample(population, len(active))
        random_means.append(fmean(returns[index] for index in indices))
    random_mean = fmean(random_means)
    p_value = (1 + sum(value >= observed for value in random_means)) / (trials + 1)
    return RuleSignificanceResult(
        trials=trials,
        seed=seed,
        active_count=len(active),
        opportunity_count=len(returns),
        observed_mean_return=observed,
        random_mean_return=random_mean,
        lift=observed - random_mean,
        p_value=p_value,
    )


def evaluate_robustness(
    pnls: Sequence[float],
    opportunity_returns: Sequence[float],
    active_mask: Sequence[bool],
    *,
    block_size: int = 5,
    policy: RobustnessPolicy = RobustnessPolicy(),
    starting_equity: float = 10_000.0,
    ruin_fraction: float = 0.50,
    seed: int = 4100,
) -> RobustnessReport:
    """Run preregistered robustness checks without promoting or executing."""

    trade_order = trade_order_monte_carlo(
        pnls,
        trials=max(1000, policy.min_trials),
        seed=seed + 1,
        starting_equity=starting_equity,
        ruin_fraction=ruin_fraction,
    )
    bootstrap = block_bootstrap_monte_carlo(
        pnls,
        block_size=block_size,
        trials=max(1000, policy.min_trials),
        seed=seed + 2,
        starting_equity=starting_equity,
        ruin_fraction=ruin_fraction,
    )
    significance = rule_significance_test(
        opportunity_returns,
        active_mask,
        trials=max(2000, policy.min_trials),
        seed=seed + 3,
    )
    checks = (
        ("block_p05_return_meets_floor", bootstrap.p05_return_pct >= policy.min_block_p05_return_pct),
        ("trade_order_drawdown_within_limit", trade_order.p95_max_drawdown_pct <= policy.max_trade_order_p95_drawdown_pct),
        ("block_loss_probability_within_limit", bootstrap.loss_probability <= policy.max_block_loss_probability),
        ("trade_order_ruin_within_limit", trade_order.ruin_probability <= policy.max_ruin_probability),
        ("block_ruin_within_limit", bootstrap.ruin_probability <= policy.max_ruin_probability),
        ("rule_significance_passes", significance.p_value <= policy.max_rule_p_value and significance.lift > 0.0),
    )
    return RobustnessReport(
        trade_order=trade_order,
        block_bootstrap=bootstrap,
        rule_significance=significance,
        checks=checks,
        status="ROBUSTNESS_CANDIDATE" if all(value for _, value in checks) else "INSUFFICIENT_ROBUSTNESS",
    )
