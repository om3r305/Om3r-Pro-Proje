from __future__ import annotations

import pytest

from brian2026.phase41_robustness_lab import (
    RobustnessPolicy,
    block_bootstrap_monte_carlo,
    evaluate_robustness,
    rule_significance_test,
    trade_order_monte_carlo,
)


def test_trade_order_monte_carlo_is_reproducible_and_preserves_total_return() -> None:
    pnls = (12.0, -8.0, 7.0, -2.0, 5.0, 4.0, -1.0, 3.0)
    first = trade_order_monte_carlo(pnls, trials=250, seed=7, starting_equity=1000.0)
    second = trade_order_monte_carlo(pnls, trials=250, seed=7, starting_equity=1000.0)
    assert first == second
    expected = sum(pnls) / 1000.0 * 100.0
    assert first.p05_return_pct == pytest.approx(expected)
    assert first.p95_return_pct == pytest.approx(expected)
    assert first.p95_max_drawdown_pct >= first.median_max_drawdown_pct


def test_block_bootstrap_preserves_local_blocks_but_changes_possible_returns() -> None:
    pnls = (5.0, 4.0, -3.0, -2.0, 6.0, 5.0, -4.0, -1.0, 3.0, 2.0)
    summary = block_bootstrap_monte_carlo(
        pnls,
        block_size=2,
        trials=300,
        seed=11,
        starting_equity=1000.0,
    )
    assert summary.method == "contiguous_block_bootstrap"
    assert summary.p05_return_pct < summary.p95_return_pct
    assert 0.0 <= summary.loss_probability <= 1.0
    assert 0.0 <= summary.ruin_probability <= 1.0


def test_rule_significance_uses_same_count_random_entries() -> None:
    returns = tuple([2.0] * 10 + [-0.2] * 90)
    mask = tuple([True] * 10 + [False] * 90)
    result = rule_significance_test(returns, mask, trials=2000, seed=13)
    assert result.active_count == 10
    assert result.opportunity_count == 100
    assert result.lift > 1.0
    assert result.p_value <= 0.05


def test_phase41_report_can_pass_but_never_promotes_or_executes() -> None:
    pnls = tuple([2.0] * 40)
    opportunities = tuple([3.0] * 10 + [0.0] * 30)
    mask = tuple([True] * 10 + [False] * 30)
    report = evaluate_robustness(
        pnls,
        opportunities,
        mask,
        block_size=4,
        policy=RobustnessPolicy(
            min_trials=250,
            min_block_p05_return_pct=0.0,
            max_trade_order_p95_drawdown_pct=1.0,
            max_block_loss_probability=0.0,
            max_ruin_probability=0.0,
            max_rule_p_value=0.05,
        ),
        starting_equity=500.0,
        seed=41,
    )
    assert report.status == "ROBUSTNESS_CANDIDATE"
    assert all(dict(report.checks).values())
    assert report.shadow_only is True
    assert report.live_execution is False
    assert report.automatic_promotion is False
