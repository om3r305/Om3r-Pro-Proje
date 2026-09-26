from __future__ import annotations

import math

import pytest

from brian2026.phase44_portfolio_brain import (
    PortfolioRiskLimits,
    PortfolioSignal,
    construct_portfolio_book,
)
from brian2026.phase52_covariance_risk import (
    CovarianceRiskConfig,
    apply_covariance_risk_overlay,
    estimate_shrunk_covariance,
)


def _returns(n: int = 80):
    btc = []
    eth = []
    sol = []
    for index in range(n):
        base = ((index % 10) - 4.5) / 1000.0
        btc.append(base)
        eth.append(base * 0.92 + (0.0002 if index % 2 else -0.0002))
        sol.append(-base * 0.35 + ((index % 3) - 1) * 0.0003)
    return {"BTC": btc, "ETH": eth, "SOL": sol}


def _plan():
    signals = (
        PortfolioSignal("model", "BTC", 0.9, evidence_ids=("ev-btc",)),
        PortfolioSignal("model", "ETH", 0.7, evidence_ids=("ev-eth",)),
        PortfolioSignal("model", "SOL", -0.4, evidence_ids=("ev-sol",)),
    )
    return construct_portfolio_book(
        signals,
        {"model": 1.0},
        gross_target=0.9,
        limits=PortfolioRiskLimits(max_position_pct=0.6, max_gross_exposure=0.9),
    )


def test_shrunk_covariance_is_symmetric_and_lw_alpha_is_bounded() -> None:
    estimate = estimate_shrunk_covariance(
        _returns(),
        config=CovarianceRiskConfig(alpha="lw", min_observations=30, max_period_volatility=0.02),
    )
    covariance = estimate.covariance
    assert 0.0 <= estimate.shrinkage_alpha <= 1.0
    assert estimate.observations == 80
    for i in range(len(covariance)):
        assert covariance[i][i] >= 0.0
        for j in range(len(covariance)):
            assert covariance[i][j] == pytest.approx(covariance[j][i], abs=1e-14)


def test_highly_related_assets_show_positive_correlation() -> None:
    estimate = estimate_shrunk_covariance(
        _returns(),
        assets=("BTC", "ETH", "SOL"),
        config=CovarianceRiskConfig(alpha=0.0, min_observations=30, max_period_volatility=0.02),
    )
    btc_eth = estimate.correlation[0][1]
    btc_sol = estimate.correlation[0][2]
    assert btc_eth > 0.9
    assert btc_sol < 0.0


def test_covariance_overlay_only_scales_down_and_keeps_directions() -> None:
    plan = _plan()
    overlay = apply_covariance_risk_overlay(
        plan,
        _returns(),
        config=CovarianceRiskConfig(
            alpha="lw",
            min_observations=30,
            max_period_volatility=0.001,
        ),
    )
    assert overlay.volatility_scale < 1.0
    assert overlay.portfolio_volatility_after <= 0.001 + 1e-12
    assert overlay.released_weight_to_cash > 0.0
    for asset, original in overlay.original_weights.items():
        scaled = overlay.scaled_weights[asset]
        assert abs(scaled) <= abs(original) + 1e-12
        if original != 0 and scaled != 0:
            assert math.copysign(1.0, scaled) == math.copysign(1.0, original)
    assert overlay.risk_only_shrinks is True
    assert overlay.shadow_only is True
    assert overlay.live_execution is False


def test_overlay_leaves_book_unchanged_when_under_risk_limit() -> None:
    plan = _plan()
    overlay = apply_covariance_risk_overlay(
        plan,
        _returns(),
        config=CovarianceRiskConfig(
            alpha=0.25,
            min_observations=30,
            max_period_volatility=1.0,
        ),
    )
    assert overlay.volatility_scale == pytest.approx(1.0)
    assert overlay.scaled_weights == pytest.approx(overlay.original_weights)
    assert overlay.released_weight_to_cash == pytest.approx(0.0)


def test_normalized_risk_contributions_reconcile_portfolio_variance() -> None:
    overlay = apply_covariance_risk_overlay(
        _plan(),
        _returns(),
        config=CovarianceRiskConfig(alpha=0.0, min_observations=30, max_period_volatility=1.0),
    )
    if overlay.portfolio_variance_before > 1e-18:
        assert sum(overlay.normalized_risk_contributions.values()) == pytest.approx(1.0, abs=1e-10)


def test_missing_or_misaligned_return_history_fails_closed() -> None:
    with pytest.raises(KeyError, match="missing return history"):
        estimate_shrunk_covariance(
            {"BTC": [0.0] * 40},
            assets=("BTC", "ETH"),
            config=CovarianceRiskConfig(min_observations=30),
        )

    with pytest.raises(ValueError, match="aligned to equal length"):
        estimate_shrunk_covariance(
            {"BTC": [0.0] * 40, "ETH": [0.0] * 39},
            config=CovarianceRiskConfig(min_observations=30),
        )


def test_nonfinite_returns_can_be_rejected_or_zero_filled_explicitly() -> None:
    data = {"BTC": [0.001] * 39 + [float("nan")], "ETH": [0.002] * 40}
    with pytest.raises(ValueError, match="non-finite"):
        estimate_shrunk_covariance(
            data,
            config=CovarianceRiskConfig(nan_policy="reject", min_observations=30),
        )
    estimate = estimate_shrunk_covariance(
        data,
        config=CovarianceRiskConfig(nan_policy="fill_zero", min_observations=30),
    )
    assert estimate.observations == 40
