from __future__ import annotations

import math

import numpy as np
import pytest

from brian2026.phase58_advanced_overfit_audit import (
    AdvancedOverfitPolicy,
    SampleInterval,
    combinatorial_purged_cv,
    deflated_sharpe_of_best,
    evaluate_advanced_overfit_risk,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    probabilistic_sharpe_ratio,
)


def _intervals(n: int, span: float = 1.0) -> tuple[SampleInterval, ...]:
    return tuple(SampleInterval(float(i), float(i) + span) for i in range(n))


def _strong_matrix(n_obs: int = 960, n_strategies: int = 20) -> np.ndarray:
    rng = np.random.default_rng(5)
    matrix = rng.normal(0.0, 0.01, size=(n_obs, n_strategies))
    matrix[:, 0] += 0.004
    return matrix


def test_cpcv_6_2_matches_reference_split_and_path_counts() -> None:
    plan = combinatorial_purged_cv(
        _intervals(120, span=0.25),
        n_splits=6,
        n_test_splits=2,
        pct_embargo=0.0,
    )
    assert plan.combinations_count == 15
    assert len(plan.splits) == 15
    assert plan.backtest_paths == 5
    assert plan.splits[0].test_blocks == (0, 1)


def test_cpcv_purges_interval_overlap_and_applies_forward_embargo() -> None:
    intervals = list(_intervals(12, span=0.0))
    # Sample 3's label resolves deep into the following block, so whenever
    # that future test window is selected it must be purged from training.
    intervals[3] = SampleInterval(3.0, 6.5)
    plan = combinatorial_purged_cv(
        tuple(intervals),
        n_splits=4,
        n_test_splits=1,
        pct_embargo=1 / 12,
    )
    split = next(row for row in plan.splits if row.test_blocks == (1,))
    assert 3 in split.test_indices
    # First index after test block is embargoed.
    assert 6 in split.embargoed_indices
    assert 6 not in split.train_indices

    # No train interval is allowed to overlap the selected test information window.
    test_window = SampleInterval(
        intervals[split.test_indices[0]].start,
        max(intervals[index].end for index in split.test_indices),
    )
    assert all(
        not (
            intervals[index].start <= test_window.end
            and test_window.start <= intervals[index].end
        )
        for index in split.train_indices
    )


def test_pbo_split_count_and_strong_strategy_behavior() -> None:
    matrix = _strong_matrix()
    result = probability_of_backtest_overfitting(
        matrix,
        n_blocks=12,
    )
    assert result.n_splits == math.comb(12, 6)
    assert result.n_strategies == 20
    assert 0.0 <= result.pbo <= 1.0
    assert result.pbo < 0.10
    assert np.mean(result.oos_sharpes_of_in_sample_winner) > 0.20


def test_pbo_max_splits_sampling_is_seeded_and_bounded() -> None:
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(320, 5))
    first = probability_of_backtest_overfitting(
        matrix,
        n_blocks=8,
        max_splits=20,
        seed=7,
    )
    second = probability_of_backtest_overfitting(
        matrix,
        n_blocks=8,
        max_splits=20,
        seed=7,
    )
    assert first == second
    assert first.n_splits == 20


def test_deflated_sharpe_penalizes_multiple_testing() -> None:
    matrix = _strong_matrix(n_obs=500, n_strategies=20)
    result = deflated_sharpe_of_best(matrix)
    assert result.best_strategy_index == 0
    assert 0.0 <= result.deflated_sharpe_probability <= 1.0
    assert result.selection_bias_benchmark_sharpe > 0.0
    assert result.deflated_sharpe_probability <= result.naive_psr + 1e-12


def test_expected_max_sharpe_grows_with_trials_and_variance() -> None:
    values = [expected_max_sharpe(n, 1.0) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)
    assert expected_max_sharpe(50, 4.0) == pytest.approx(
        2.0 * expected_max_sharpe(50, 1.0)
    )
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_single_trial_deflation_reduces_to_psr_against_zero() -> None:
    rng = np.random.default_rng(2)
    returns = rng.normal(0.002, 0.01, size=(300, 1))
    result = deflated_sharpe_of_best(returns)
    expected = probabilistic_sharpe_ratio(
        result.observed_sharpe,
        0.0,
        result.n_observations,
        skew=result.skew,
        raw_kurtosis=result.raw_kurtosis,
    )
    assert result.selection_bias_benchmark_sharpe == pytest.approx(0.0)
    assert result.deflated_sharpe_probability == pytest.approx(expected)


def test_advanced_overfit_report_can_pass_but_never_promotes_or_executes() -> None:
    matrix = _strong_matrix()
    report = evaluate_advanced_overfit_risk(
        _intervals(len(matrix), span=0.25),
        matrix,
        cpcv_n_splits=6,
        cpcv_n_test_splits=2,
        pct_embargo=0.01,
        pbo_n_blocks=12,
        policy=AdvancedOverfitPolicy(
            max_pbo=0.10,
            min_deflated_sharpe_probability=0.90,
            min_cpcv_backtest_paths=5,
        ),
    )
    assert report.status == "ADVANCED_ROBUSTNESS_CANDIDATE"
    assert all(dict(report.checks).values())
    assert report.research_only is True
    assert report.automatic_promotion is False
    assert report.live_execution is False


def test_bad_shapes_and_variance_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least two strategy variants"):
        probability_of_backtest_overfitting(np.ones((100, 1)), n_blocks=4)

    matrix = np.ones((100, 2))
    with pytest.raises(ValueError, match="non-zero sample variance"):
        probability_of_backtest_overfitting(matrix, n_blocks=4)

    with pytest.raises(ValueError, match="interval count"):
        evaluate_advanced_overfit_risk(
            _intervals(10),
            np.random.default_rng(0).normal(size=(12, 3)),
            cpcv_n_splits=2,
            cpcv_n_test_splits=1,
            pbo_n_blocks=2,
        )
