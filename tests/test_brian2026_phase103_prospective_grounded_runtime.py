from __future__ import annotations

import pytest

from brian2026.expert_reasoner import reason_market
from brian2026.global_sensor_mesh import SensorObservation
from brian2026.phase44_portfolio_brain import PortfolioRiskLimits
from brian2026.phase52_covariance_risk import CovarianceRiskConfig
from brian2026.phase53_turnover_rebalance import TurnoverConfig
from brian2026.phase54_integrated_shadow_decision import (
    AssetDecisionInput,
    IntegratedShadowConfig,
)
from brian2026.phase103_prospective_grounded_runtime import (
    ProspectiveGroundedRuntimeError,
    run_prospective_grounded_phase43,
    run_prospective_integrated_shadow_decision,
)
from brian2026.portfolio import DEVELOPMENT_CUTOFF


TS = 1_790_000_000.0


def _snapshot(direction: int = 1) -> dict[str, float]:
    bull = direction > 0
    return {
        "structure_state": float(direction),
        "structure_15m": float(direction),
        "structure_1h": float(direction),
        "ema_slope": 0.01 * direction,
        "relative_volume": 1.6,
        "volume_zscore": 1.1 * direction,
        "acceleration": 0.4 * direction,
        "return_1": 0.3 * direction,
        "rsi": 62.0 if bull else 38.0,
        "support_distance_atr": 0.5,
        "resistance_distance_atr": 1.2,
        "nearest_support": 100.0,
        "nearest_resistance": 110.0,
        "dip_score": 0.8 if bull else 0.2,
        "rally_score": 0.2 if bull else 0.8,
        "bullish_breakout_retest": 1.0 if bull else 0.0,
        "bearish_breakout_retest": 0.0 if bull else 1.0,
        "range_expansion": 1.0,
        "zscore": 0.4 * direction,
        "bb_position": 0.65 if bull else 0.35,
    }


def _obs(
    asset: str,
    eye: str,
    group: str,
    direction: int,
    *,
    observed_at: float = TS - 60,
    strength: float = 0.9,
) -> SensorObservation:
    return SensorObservation(
        eye_id=f"{asset}-{eye}",
        asset_id=asset,
        observed_at=observed_at,
        direction=direction,
        strength=strength,
        confidence=0.9,
        reliability=0.85,
        available=True,
        independent_group=group,
        source_ids=(f"source-{asset}-{eye}",),
        horizon="FAST_5_30M",
        reason="prospective phase103 fixture",
    )


def _asset_input(asset: str, direction: int) -> AssetDecisionInput:
    return AssetDecisionInput(
        snapshot=_snapshot(direction),
        observations=(
            _obs(asset, "market", "price_structure", direction, strength=0.95),
            _obs(asset, "news", "news_verified", direction, strength=0.85),
            _obs(asset, "deriv", "derivatives", direction, strength=0.75),
        ),
        source_kind_by_eye={
            f"{asset}-market": "market_snapshot",
            f"{asset}-news": "verified_news",
            f"{asset}-deriv": "derivatives",
        },
    )


def _returns():
    btc = []
    eth = []
    for index in range(80):
        base = ((index % 12) - 5.5) / 2000.0
        btc.append(base)
        eth.append(base * 0.4 + ((index % 3) - 1) * 0.0004)
    return {"BTCUSDT": btc, "ETHUSDT": eth}


def _config() -> IntegratedShadowConfig:
    return IntegratedShadowConfig(
        gross_target=0.60,
        position_limits=PortfolioRiskLimits(
            max_position_pct=0.40,
            max_gross_exposure=0.60,
        ),
        covariance=CovarianceRiskConfig(
            alpha="lw",
            min_observations=30,
            max_period_volatility=1.0,
        ),
        turnover=TurnoverConfig(
            max_l1_turnover=1.0,
            risk_reduction_bypass=True,
        ),
    )


def _weights():
    return {
        "market_snapshot_analyst": 1.0,
        "verified_news_analyst": 1.0,
        "derivatives_analyst": 1.0,
    }


def test_frozen_reasoner_still_rejects_post_cutoff_data() -> None:
    with pytest.raises(ValueError, match="INVALID_CONTAMINATED"):
        reason_market(
            _snapshot(),
            timestamp=TS,
        )


def test_prospective_phase43_accepts_current_shadow_observations() -> None:
    result = run_prospective_grounded_phase43(
        _snapshot(),
        (
            _obs("BTCUSDT", "market", "price_structure", 1),
            _obs("BTCUSDT", "news", "news_verified", 1),
            _obs("BTCUSDT", "deriv", "derivatives", 1),
        ),
        timestamp=TS,
        source_kind_by_eye={
            "BTCUSDT-market": "market_snapshot",
            "BTCUSDT-news": "verified_news",
            "BTCUSDT-deriv": "derivatives",
        },
    )

    assert result.analyst_direction == 1
    assert result.packet.decision_timestamp == TS
    assert result.expert_decision.timestamp == TS
    assert result.shadow_only is True
    assert result.live_execution is False
    assert result.automatic_promotion is False


def test_prospective_phase43_rejects_pre_cutoff_reused_evidence() -> None:
    old = _obs(
        "BTCUSDT",
        "old",
        "price_structure",
        1,
        observed_at=DEVELOPMENT_CUTOFF - 1,
    )
    with pytest.raises(
        ProspectiveGroundedRuntimeError,
        match="pre-cutoff development evidence",
    ):
        run_prospective_grounded_phase43(
            _snapshot(),
            (old,),
            timestamp=TS,
        )


def test_prospective_phase43_rejects_future_observation() -> None:
    future = _obs(
        "BTCUSDT",
        "future",
        "price_structure",
        1,
        observed_at=TS + 1,
    )
    with pytest.raises(
        ProspectiveGroundedRuntimeError,
        match="from the future",
    ):
        run_prospective_grounded_phase43(
            _snapshot(),
            (future,),
            timestamp=TS,
        )


def test_prospective_phase43_requires_post_cutoff_decision_timestamp() -> None:
    with pytest.raises(
        ProspectiveGroundedRuntimeError,
        match="post-cutoff decision timestamp",
    ):
        run_prospective_grounded_phase43(
            _snapshot(),
            (
                _obs(
                    "BTCUSDT",
                    "market",
                    "price_structure",
                    1,
                    observed_at=DEVELOPMENT_CUTOFF - 60,
                ),
            ),
            timestamp=DEVELOPMENT_CUTOFF - 1,
        )


def test_prospective_phase54_runs_real_current_grounded_pipeline() -> None:
    result = run_prospective_integrated_shadow_decision(
        {
            "BTCUSDT": _asset_input("BTCUSDT", 1),
            "ETHUSDT": _asset_input("ETHUSDT", -1),
        },
        timestamp=TS,
        model_weights=_weights(),
        current_weights={},
        returns_by_asset=_returns(),
        config=_config(),
    )

    assert result.status == "REBALANCE_PLANNED"
    assert result.portfolio_book is not None
    assert result.covariance_overlay is not None
    assert result.turnover_plan is not None
    assert result.asset_results["BTCUSDT"].analyst_direction == 1
    assert result.asset_results["ETHUSDT"].analyst_direction == -1
    assert result.pipeline_id
    assert result.shadow_only is True
    assert result.live_execution is False
    assert result.automatic_promotion is False


def test_prospective_phase54_does_not_mutate_frozen_phase54_default_lane() -> None:
    from brian2026.phase54_integrated_shadow_decision import (
        run_integrated_shadow_decision,
    )

    with pytest.raises(ValueError, match="INVALID_CONTAMINATED"):
        run_integrated_shadow_decision(
            {"BTCUSDT": _asset_input("BTCUSDT", 1)},
            timestamp=TS,
            model_weights=_weights(),
            current_weights={},
            returns_by_asset={"BTCUSDT": _returns()["BTCUSDT"]},
            config=_config(),
        )


def test_prospective_phase54_rejects_pre_cutoff_timestamp_even_with_valid_shapes() -> None:
    old_ts = DEVELOPMENT_CUTOFF - 10
    old_input = AssetDecisionInput(
        snapshot=_snapshot(1),
        observations=(
            _obs(
                "BTCUSDT",
                "market",
                "price_structure",
                1,
                observed_at=old_ts - 10,
            ),
        ),
        source_kind_by_eye={"BTCUSDT-market": "market_snapshot"},
    )
    with pytest.raises(
        ProspectiveGroundedRuntimeError,
        match="post-cutoff decision timestamp",
    ):
        run_prospective_integrated_shadow_decision(
            {"BTCUSDT": old_input},
            timestamp=old_ts,
            model_weights={"market_snapshot_analyst": 1.0},
            current_weights={},
            returns_by_asset={"BTCUSDT": _returns()["BTCUSDT"]},
            config=_config(),
        )
