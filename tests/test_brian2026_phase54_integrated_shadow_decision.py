from __future__ import annotations

import pytest

from brian2026.global_sensor_mesh import SensorObservation
from brian2026.phase44_portfolio_brain import PortfolioRiskLimits
from brian2026.phase52_covariance_risk import CovarianceRiskConfig
from brian2026.phase53_turnover_rebalance import TurnoverConfig
from brian2026.phase54_integrated_shadow_decision import (
    AssetDecisionInput,
    IntegratedShadowConfig,
    run_integrated_shadow_decision,
)


TS = 1_760_000_000.0


def _snapshot(direction: int) -> dict[str, float]:
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


def _obs(asset: str, eye: str, group: str, direction: int, strength: float = 0.9) -> SensorObservation:
    return SensorObservation(
        eye_id=f"{asset}-{eye}",
        asset_id=asset,
        observed_at=TS - 60,
        direction=direction,
        strength=strength,
        confidence=0.9,
        reliability=0.85,
        available=True,
        independent_group=group,
        source_ids=(f"source-{asset}-{eye}",),
        horizon="FAST_5_30M",
        reason="integrated phase54 fixture",
    )


def _asset_input(asset: str, direction: int) -> AssetDecisionInput:
    return AssetDecisionInput(
        snapshot=_snapshot(direction),
        observations=(
            _obs(asset, "market", "price_structure", direction, 0.95),
            _obs(asset, "news", "news_verified", direction, 0.85),
            _obs(asset, "deriv", "derivatives", direction, 0.75),
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


def test_integrated_chain_runs_real_phase_outputs_end_to_end() -> None:
    result = run_integrated_shadow_decision(
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
    assert set(result.asset_results) == {"BTCUSDT", "ETHUSDT"}
    assert result.asset_results["BTCUSDT"].analyst_direction == 1
    assert result.asset_results["ETHUSDT"].analyst_direction == -1
    assert result.final_planned_weights["BTCUSDT"] > 0
    assert result.final_planned_weights["ETHUSDT"] < 0
    assert sum(abs(value) for value in result.final_planned_weights.values()) <= 0.60 + 1e-12
    assert result.source_evidence_ids
    assert result.pipeline_id
    assert result.shadow_only is True
    assert result.live_execution is False
    assert result.automatic_promotion is False


def test_pipeline_id_is_deterministic_for_same_inputs() -> None:
    kwargs = dict(
        asset_inputs={
            "BTCUSDT": _asset_input("BTCUSDT", 1),
            "ETHUSDT": _asset_input("ETHUSDT", -1),
        },
        timestamp=TS,
        model_weights=_weights(),
        current_weights={},
        returns_by_asset=_returns(),
        config=_config(),
    )
    first = run_integrated_shadow_decision(**kwargs)
    second = run_integrated_shadow_decision(**kwargs)
    assert first.pipeline_id == second.pipeline_id
    assert first.final_planned_weights == pytest.approx(second.final_planned_weights)


def test_missing_analyst_weight_fails_before_portfolio_construction() -> None:
    weights = _weights()
    del weights["verified_news_analyst"]
    with pytest.raises(ValueError, match="missing model weights"):
        run_integrated_shadow_decision(
            {"BTCUSDT": _asset_input("BTCUSDT", 1)},
            timestamp=TS,
            model_weights=weights,
            current_weights={},
            returns_by_asset={"BTCUSDT": _returns()["BTCUSDT"]},
            config=_config(),
        )


def test_asset_key_and_observation_identity_must_match() -> None:
    with pytest.raises(ValueError, match="does not match observation assets"):
        run_integrated_shadow_decision(
            {"WRONG": _asset_input("BTCUSDT", 1)},
            timestamp=TS,
            model_weights=_weights(),
            current_weights={},
            returns_by_asset={"WRONG": _returns()["BTCUSDT"]},
            config=_config(),
        )


def test_no_grounded_directional_sources_holds_current_book() -> None:
    neutral_observations = (
        SensorObservation(
            eye_id="BTCUSDT-neutral",
            asset_id="BTCUSDT",
            observed_at=TS - 60,
            direction=0,
            strength=0.0,
            confidence=0.8,
            reliability=0.8,
            available=True,
            independent_group="news_verified",
            source_ids=("neutral-source",),
            horizon="FAST_5_30M",
            reason="no directional evidence",
        ),
    )
    result = run_integrated_shadow_decision(
        {
            "BTCUSDT": AssetDecisionInput(
                snapshot=_snapshot(1),
                observations=neutral_observations,
                source_kind_by_eye={"BTCUSDT-neutral": "verified_news"},
            ),
        },
        timestamp=TS,
        model_weights={},
        current_weights={"BTCUSDT": 0.15},
        returns_by_asset={"BTCUSDT": _returns()["BTCUSDT"]},
        config=_config(),
    )
    assert result.status == "WAIT_NO_GROUNDED_SIGNALS"
    assert result.portfolio_book is None
    assert result.covariance_overlay is None
    assert result.turnover_plan is None
    assert result.final_planned_weights == {"BTCUSDT": 0.15}


def test_existing_asset_without_new_target_is_reduced_to_zero_by_rebalance_stage() -> None:
    result = run_integrated_shadow_decision(
        {
            "BTCUSDT": _asset_input("BTCUSDT", 1),
            "ETHUSDT": _asset_input("ETHUSDT", -1),
        },
        timestamp=TS,
        model_weights=_weights(),
        current_weights={"SOLUSDT": 0.10},
        returns_by_asset=_returns(),
        config=_config(),
    )
    assert result.turnover_plan is not None
    assert result.final_planned_weights["SOLUSDT"] == pytest.approx(0.0)
