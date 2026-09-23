from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import comb, exp, isfinite, log, sqrt
from random import Random
from statistics import NormalDist
from typing import Sequence
import math

import numpy as np

PHASE58_SCHEMA_VERSION = "brian.phase58-advanced-overfit-audit.v1"
_EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


@dataclass(frozen=True, slots=True)
class SampleInterval:
    start: float
    end: float

    def __post_init__(self) -> None:
        if not isfinite(self.start) or not isfinite(self.end):
            raise ValueError("sample interval bounds must be finite")
        if self.end < self.start:
            raise ValueError("sample interval end cannot precede start")


@dataclass(frozen=True, slots=True)
class CPCVSplit:
    test_blocks: tuple[int, ...]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    purged_indices: tuple[int, ...]
    embargoed_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CPCVPlan:
    n_splits: int
    n_test_splits: int
    pct_embargo: float
    combinations_count: int
    backtest_paths: int
    splits: tuple[CPCVSplit, ...]
    schema_version: str = PHASE58_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "n_splits": self.n_splits,
            "n_test_splits": self.n_test_splits,
            "pct_embargo": self.pct_embargo,
            "combinations_count": self.combinations_count,
            "backtest_paths": self.backtest_paths,
            "splits": [row.to_dict() for row in self.splits],
        }


def _contiguous_blocks(n_observations: int, n_splits: int) -> tuple[tuple[int, ...], ...]:
    raw = np.array_split(np.arange(n_observations, dtype=int), n_splits)
    blocks = tuple(tuple(int(value) for value in block) for block in raw)
    if any(not block for block in blocks):
        raise ValueError("n_splits cannot exceed observation count")
    return blocks


def _overlaps(left: SampleInterval, right: SampleInterval) -> bool:
    return left.start <= right.end and right.start <= left.end


def combinatorial_purged_cv(
    intervals: Sequence[SampleInterval],
    *,
    n_splits: int = 6,
    n_test_splits: int = 2,
    pct_embargo: float = 0.0,
) -> CPCVPlan:
    """Generate CPCV splits with interval purging and index embargo.

    Clean-room implementation of the public CPCV behavior:
    - observations are divided into contiguous time blocks;
    - every combination of K test blocks is evaluated;
    - train samples whose label-information interval overlaps a test interval
      are purged;
    - a forward index embargo after every test block is excluded from training.
    """
    rows = tuple(intervals)
    if len(rows) < 2:
        raise ValueError("CPCV requires multiple observations")
    if n_splits < 2 or n_test_splits < 1 or n_test_splits >= n_splits:
        raise ValueError("CPCV requires 1 <= n_test_splits < n_splits")
    if n_splits > len(rows):
        raise ValueError("n_splits cannot exceed observations")
    if not 0.0 <= pct_embargo < 1.0:
        raise ValueError("pct_embargo must be in [0,1)")
    if any(rows[index].start > rows[index + 1].start for index in range(len(rows) - 1)):
        raise ValueError("sample intervals must be chronological by start")

    blocks = _contiguous_blocks(len(rows), n_splits)
    embargo_count = int(len(rows) * pct_embargo)
    all_indices = set(range(len(rows)))
    output: list[CPCVSplit] = []

    for test_blocks in combinations(range(n_splits), n_test_splits):
        test_indices = tuple(sorted(index for block_id in test_blocks for index in blocks[block_id]))
        test_set = set(test_indices)

        # One information interval per contiguous test block, spanning sample
        # information start through label-resolution end.
        test_windows = tuple(
            SampleInterval(
                rows[blocks[block_id][0]].start,
                max(rows[index].end for index in blocks[block_id]),
            )
            for block_id in test_blocks
        )

        purged = {
            index
            for index in all_indices - test_set
            if any(_overlaps(rows[index], window) for window in test_windows)
        }

        embargoed: set[int] = set()
        if embargo_count > 0:
            for block_id in test_blocks:
                end_exclusive = blocks[block_id][-1] + 1
                embargoed.update(
                    range(
                        end_exclusive,
                        min(len(rows), end_exclusive + embargo_count),
                    )
                )
            embargoed.difference_update(test_set)

        train = tuple(sorted(all_indices - test_set - purged - embargoed))
        output.append(CPCVSplit(
            test_blocks=tuple(int(value) for value in test_blocks),
            train_indices=train,
            test_indices=test_indices,
            purged_indices=tuple(sorted(purged)),
            embargoed_indices=tuple(sorted(embargoed)),
        ))

    combinations_count = comb(n_splits, n_test_splits)
    paths_float = combinations_count * n_test_splits / n_splits
    if not math.isclose(paths_float, round(paths_float), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("CPCV configuration does not produce an integer number of backtest paths")

    return CPCVPlan(
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        pct_embargo=pct_embargo,
        combinations_count=combinations_count,
        backtest_paths=int(round(paths_float)),
        splits=tuple(output),
    )


def _sharpe_vector(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=1)
    if np.any(~np.isfinite(means)) or np.any(~np.isfinite(stds)):
        raise ValueError("return matrix produced non-finite Sharpe inputs")
    if np.any(stds <= 1e-15):
        raise ValueError("strategy variants must have non-zero sample variance")
    return means / stds


@dataclass(frozen=True, slots=True)
class PBOResult:
    pbo: float
    logits: tuple[float, ...]
    in_sample_best_sharpes: tuple[float, ...]
    oos_sharpes_of_in_sample_winner: tuple[float, ...]
    n_splits: int
    n_blocks: int
    n_strategies: int
    schema_version: str = PHASE58_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probability_of_backtest_overfitting(
    returns_matrix: Sequence[Sequence[float]],
    *,
    n_blocks: int = 16,
    max_splits: int | None = None,
    seed: int = 5801,
) -> PBOResult:
    """CSCV probability of backtest overfitting across strategy variants."""
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (time, strategy)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("returns_matrix must contain only finite values")
    n_obs, n_strategies = matrix.shape
    if n_strategies < 2:
        raise ValueError("PBO requires at least two strategy variants")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even integer >= 2")
    block_len = n_obs // n_blocks
    if block_len < 3:
        raise ValueError("PBO requires at least three observations per block")
    if max_splits is not None and max_splits <= 0:
        raise ValueError("max_splits must be positive when set")

    usable = matrix[: block_len * n_blocks]
    blocks = usable.reshape(n_blocks, block_len, n_strategies)
    all_combos = list(combinations(range(n_blocks), n_blocks // 2))
    if max_splits is not None and len(all_combos) > max_splits:
        rng = Random(seed)
        chosen = sorted(rng.sample(range(len(all_combos)), max_splits))
        split_combos = [all_combos[index] for index in chosen]
    else:
        split_combos = all_combos

    everything = np.arange(n_blocks)
    logits: list[float] = []
    is_best: list[float] = []
    oos_best: list[float] = []

    for combo in split_combos:
        is_blocks = np.asarray(combo, dtype=int)
        oos_blocks = np.setdiff1d(everything, is_blocks)
        ins = blocks[is_blocks].reshape(-1, n_strategies)
        oos = blocks[oos_blocks].reshape(-1, n_strategies)
        sr_is = _sharpe_vector(ins)
        sr_oos = _sharpe_vector(oos)
        winner = int(np.argmax(sr_is))

        # 1=worst ... N=best. Relative rank stays strictly within (0,1).
        rank = int(np.sum(sr_oos < sr_oos[winner])) + 1
        omega = rank / (n_strategies + 1.0)
        logits.append(log(omega / (1.0 - omega)))
        is_best.append(float(sr_is[winner]))
        oos_best.append(float(sr_oos[winner]))

    logit_array = np.asarray(logits, dtype=float)
    return PBOResult(
        pbo=float(np.mean(logit_array <= 0.0)),
        logits=tuple(float(value) for value in logits),
        in_sample_best_sharpes=tuple(is_best),
        oos_sharpes_of_in_sample_winner=tuple(oos_best),
        n_splits=len(split_combos),
        n_blocks=n_blocks,
        n_strategies=n_strategies,
    )


def _sample_skew(values: np.ndarray) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    centered = values - values.mean()
    m2 = float(np.mean(centered ** 2))
    if m2 <= 1e-30:
        return 0.0
    m3 = float(np.mean(centered ** 3))
    # Bias-corrected Fisher-Pearson standardized moment.
    g1 = m3 / (m2 ** 1.5)
    return float(sqrt(n * (n - 1)) / (n - 2) * g1)


def _sample_raw_kurtosis(values: np.ndarray) -> float:
    n = len(values)
    if n < 4:
        return 3.0
    centered = values - values.mean()
    m2 = float(np.mean(centered ** 2))
    if m2 <= 1e-30:
        return 3.0
    m4 = float(np.mean(centered ** 4))
    # Use raw moment kurtosis; DSR correction expects normal ~= 3.
    return float(m4 / (m2 ** 2))


def sharpe_standard_error(
    observed_sr: float,
    n_obs: int,
    *,
    skew: float = 0.0,
    raw_kurtosis: float = 3.0,
) -> float:
    if n_obs < 2:
        raise ValueError("Sharpe standard error requires at least two observations")
    variance = (
        1.0
        - skew * observed_sr
        + (raw_kurtosis - 1.0) / 4.0 * observed_sr ** 2
    ) / (n_obs - 1)
    if variance <= 0 or not isfinite(variance):
        raise ValueError("Sharpe standard-error variance is invalid")
    return sqrt(variance)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    *,
    skew: float = 0.0,
    raw_kurtosis: float = 3.0,
) -> float:
    se = sharpe_standard_error(
        observed_sr,
        n_obs,
        skew=skew,
        raw_kurtosis=raw_kurtosis,
    )
    return float(_NORMAL.cdf((observed_sr - benchmark_sr) / se))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if not isfinite(sr_variance) or sr_variance < 0:
        raise ValueError("sr_variance must be finite and non-negative")
    if n_trials == 1 or sr_variance == 0:
        return 0.0
    z1 = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * exp(1.0)))
    return sqrt(sr_variance) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    best_strategy_index: int
    observed_sharpe: float
    naive_psr: float
    deflated_sharpe_probability: float
    selection_bias_benchmark_sharpe: float
    n_observations: int
    n_trials: int
    sharpe_variance_across_trials: float
    skew: float
    raw_kurtosis: float
    schema_version: str = PHASE58_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def deflated_sharpe_of_best(
    returns_matrix: Sequence[Sequence[float]],
) -> DeflatedSharpeResult:
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("returns_matrix must be 2-D with strategy columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("returns_matrix must contain only finite values")
    n_obs, n_trials = matrix.shape
    if n_obs < 4:
        raise ValueError("deflated Sharpe requires at least four observations")

    sharpes = _sharpe_vector(matrix)
    best = int(np.argmax(sharpes))
    best_returns = matrix[:, best]
    observed = float(sharpes[best])
    skew = _sample_skew(best_returns)
    kurtosis = _sample_raw_kurtosis(best_returns)
    sr_variance = float(np.var(sharpes, ddof=1)) if n_trials > 1 else 0.0
    benchmark = expected_max_sharpe(n_trials, sr_variance)
    return DeflatedSharpeResult(
        best_strategy_index=best,
        observed_sharpe=observed,
        naive_psr=probabilistic_sharpe_ratio(
            observed, 0.0, n_obs, skew=skew, raw_kurtosis=kurtosis
        ),
        deflated_sharpe_probability=probabilistic_sharpe_ratio(
            observed, benchmark, n_obs, skew=skew, raw_kurtosis=kurtosis
        ),
        selection_bias_benchmark_sharpe=benchmark,
        n_observations=n_obs,
        n_trials=n_trials,
        sharpe_variance_across_trials=sr_variance,
        skew=skew,
        raw_kurtosis=kurtosis,
    )


@dataclass(frozen=True, slots=True)
class AdvancedOverfitPolicy:
    max_pbo: float = 0.20
    min_deflated_sharpe_probability: float = 0.95
    min_cpcv_backtest_paths: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.max_pbo <= 1:
            raise ValueError("max_pbo must be in [0,1]")
        if not 0 <= self.min_deflated_sharpe_probability <= 1:
            raise ValueError("min_deflated_sharpe_probability must be in [0,1]")
        if self.min_cpcv_backtest_paths < 1:
            raise ValueError("min_cpcv_backtest_paths must be positive")


@dataclass(frozen=True, slots=True)
class AdvancedOverfitReport:
    cpcv: CPCVPlan
    pbo: PBOResult
    deflated_sharpe: DeflatedSharpeResult
    checks: tuple[tuple[str, bool], ...]
    status: str
    schema_version: str = PHASE58_SCHEMA_VERSION
    research_only: bool = True
    automatic_promotion: bool = False
    live_execution: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cpcv": self.cpcv.to_dict(),
            "pbo": self.pbo.to_dict(),
            "deflated_sharpe": self.deflated_sharpe.to_dict(),
            "checks": dict(self.checks),
            "status": self.status,
            "research_only": self.research_only,
            "automatic_promotion": self.automatic_promotion,
            "live_execution": self.live_execution,
        }


def evaluate_advanced_overfit_risk(
    intervals: Sequence[SampleInterval],
    returns_matrix: Sequence[Sequence[float]],
    *,
    cpcv_n_splits: int = 6,
    cpcv_n_test_splits: int = 2,
    pct_embargo: float = 0.01,
    pbo_n_blocks: int = 16,
    pbo_max_splits: int | None = 2000,
    pbo_seed: int = 5801,
    policy: AdvancedOverfitPolicy = AdvancedOverfitPolicy(),
) -> AdvancedOverfitReport:
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(intervals):
        raise ValueError("interval count must match return-matrix observations")

    cpcv = combinatorial_purged_cv(
        intervals,
        n_splits=cpcv_n_splits,
        n_test_splits=cpcv_n_test_splits,
        pct_embargo=pct_embargo,
    )
    pbo = probability_of_backtest_overfitting(
        matrix,
        n_blocks=pbo_n_blocks,
        max_splits=pbo_max_splits,
        seed=pbo_seed,
    )
    dsr = deflated_sharpe_of_best(matrix)
    checks = (
        ("cpcv_has_enough_paths", cpcv.backtest_paths >= policy.min_cpcv_backtest_paths),
        ("pbo_within_limit", pbo.pbo <= policy.max_pbo),
        (
            "deflated_sharpe_significant",
            dsr.deflated_sharpe_probability >= policy.min_deflated_sharpe_probability,
        ),
    )
    return AdvancedOverfitReport(
        cpcv=cpcv,
        pbo=pbo,
        deflated_sharpe=dsr,
        checks=checks,
        status="ADVANCED_ROBUSTNESS_CANDIDATE" if all(value for _, value in checks) else "OVERFIT_RISK",
    )
