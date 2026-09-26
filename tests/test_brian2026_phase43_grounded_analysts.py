from __future__ import annotations

import pytest

from brian2026.global_sensor_mesh import SensorObservation
from brian2026.phase43_grounded_analysts import (
    GroundedAnalystClaim,
    compile_prefetched_analyst_claims,
    prefetch_structured_evidence,
    route_specialists,
    run_grounded_phase43,
    validate_grounded_claim,
)


TS = 1_760_000_000.0


def _obs(
    *,
    eye: str,
    group: str,
    direction: int,
    observed_at: float = TS - 60,
    available: bool = True,
    strength: float = 0.9,
    confidence: float = 0.9,
    reliability: float = 0.8,
    horizon: str = "FAST_5_30M",
) -> SensorObservation:
    return SensorObservation(
        eye_id=eye,
        asset_id="crypto:BTCUSDT",
        observed_at=observed_at,
        direction=direction,
        strength=strength if available else 0.0,
        confidence=confidence if available else 0.0,
        reliability=reliability if available else 0.0,
        available=available,
        independent_group=group,
        source_ids=(f"src-{eye}",) if available else (),
        horizon=horizon,
        reason=f"{eye} fixture",
    )


def _snapshot(**overrides) -> dict[str, float]:
    base = {
        "structure_state": 1.0,
        "structure_15m": 1.0,
        "structure_1h": 1.0,
        "ema_slope": 0.01,
        "relative_volume": 1.5,
        "volume_zscore": 1.2,
        "acceleration": 0.4,
        "return_1": 0.3,
        "rsi": 61.0,
        "support_distance_atr": 0.5,
        "resistance_distance_atr": 1.2,
        "nearest_support": 100.0,
        "nearest_resistance": 110.0,
        "dip_score": 0.8,
        "rally_score": 0.2,
        "bullish_breakout_retest": 1.0,
        "range_expansion": 1.0,
        "zscore": 0.4,
        "bb_position": 0.65,
    }
    base.update(overrides)
    return base


def test_prefetch_rejects_future_evidence_instead_of_hiding_leakage() -> None:
    with pytest.raises(ValueError, match="future evidence forbidden"):
        prefetch_structured_evidence(
            (_obs(eye="future", group="news", direction=1, observed_at=TS + 1),),
            decision_timestamp=TS,
        )


def test_prefetch_keeps_unavailable_source_as_explicit_placeholder() -> None:
    packet = prefetch_structured_evidence(
        (
            _obs(eye="news", group="news_verified", direction=1),
            _obs(eye="social", group="social", direction=0, available=False),
        ),
        decision_timestamp=TS,
        source_kind_by_eye={"news": "verified_news", "social": "social"},
    )
    assert packet.external_tools_allowed_after_prefetch is False
    assert packet.unavailable_source_kinds == ("social",)
    assert len(packet.blocks) == 2
    assert sum(block.available for block in packet.blocks) == 1


def test_directional_claim_cannot_reference_unknown_stale_or_discovery_only_evidence() -> None:
    packet = prefetch_structured_evidence(
        (
            _obs(eye="fresh", group="news_verified", direction=1),
            _obs(eye="stale", group="derivatives", direction=1, observed_at=TS - 3600),
            _obs(eye="gdelt", group="news_gdelt", direction=1),
        ),
        decision_timestamp=TS,
        source_kind_by_eye={"fresh": "verified_news", "stale": "derivatives", "gdelt": "news_gdelt"},
    )
    by_eye = {block.observation_id: block for block in packet.blocks}

    with pytest.raises(ValueError, match="unknown evidence"):
        validate_grounded_claim(
            packet,
            GroundedAnalystClaim("news", 1, 0.8, "fabricated", ("missing-id",)),
        )

    with pytest.raises(ValueError, match="unavailable/stale/non-directional"):
        validate_grounded_claim(
            packet,
            GroundedAnalystClaim("derivatives", 1, 0.8, "stale", (by_eye[_obs(eye="stale", group="derivatives", direction=1, observed_at=TS - 3600).observation_id].evidence_id,)),
        )

    with pytest.raises(ValueError, match="unavailable/stale/non-directional"):
        validate_grounded_claim(
            packet,
            GroundedAnalystClaim("gdelt", 1, 0.8, "discovery only", (by_eye[_obs(eye="gdelt", group="news_gdelt", direction=1).observation_id].evidence_id,)),
        )


def test_compiler_builds_source_level_claims_only_from_prefetched_grounded_rows() -> None:
    packet = prefetch_structured_evidence(
        (
            _obs(eye="news-a", group="news_verified", direction=1, strength=0.9),
            _obs(eye="news-b", group="news_verified", direction=1, strength=0.7),
            _obs(eye="deriv", group="derivatives", direction=-1, strength=0.5),
        ),
        decision_timestamp=TS,
        source_kind_by_eye={"news-a": "verified_news", "news-b": "verified_news", "deriv": "derivatives"},
    )
    claims = compile_prefetched_analyst_claims(packet)
    assert {claim.analyst for claim in claims} == {"verified_news_analyst", "derivatives_analyst"}
    news = next(claim for claim in claims if claim.analyst == "verified_news_analyst")
    assert news.direction == 1
    assert len(news.independent_support_groups) == 1
    assert news.status == "GROUNDED_DIRECTIONAL"


def test_regime_router_caps_features_and_changes_specialists_by_context() -> None:
    trend = route_specialists(_snapshot())
    assert trend.regime == "ALIGNED_UPTREND"
    assert "mean_reversion_expert" not in trend.selected_experts
    assert len(trend.selected_features) <= 8

    ranging = route_specialists(_snapshot(structure_state=0.0, structure_15m=0.0, structure_1h=0.0))
    assert ranging.regime == "RANGE"
    assert "mean_reversion_expert" in ranging.selected_experts
    assert len(ranging.selected_features) <= 8

    shock = route_specialists(_snapshot(range_expansion=3.5))
    assert shock.regime == "VOLATILITY_SHOCK"
    assert "mean_reversion_expert" not in shock.selected_experts


def test_phase43_routes_real_expert_reasoner_and_remains_shadow_only() -> None:
    observations = (
        _obs(eye="price", group="price_structure", direction=1),
        _obs(eye="news", group="news_verified", direction=1, strength=0.8),
        _obs(eye="deriv", group="derivatives", direction=1, strength=0.7),
    )
    result = run_grounded_phase43(
        _snapshot(),
        observations,
        timestamp=TS,
        source_kind_by_eye={
            "price": "market_snapshot",
            "news": "verified_news",
            "deriv": "derivatives",
        },
    )
    expert_names = {expert.name for expert in result.expert_decision.experts}
    assert "mean_reversion_expert" not in expert_names
    assert {"structure_expert", "trend_expert", "momentum_expert", "volume_expert"} <= expert_names
    assert "risk_critic" in expert_names
    assert result.analyst_direction == 1
    assert result.analyst_support_evidence_ids
    assert all(
        evidence_id in {block.evidence_id for block in result.packet.blocks}
        for evidence_id in result.analyst_support_evidence_ids
    )
    assert result.shadow_only is True
    assert result.live_execution is False
    assert result.automatic_promotion is False
