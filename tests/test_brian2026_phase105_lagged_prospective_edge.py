from __future__ import annotations

import pytest

from brian2026.phase105_lagged_prospective_edge import (
    EvidenceFreshness,
    LaggedReliabilityEvidence,
    bounded_prospective_reliability,
    eligible_expected_edge_bps_by_asset,
    estimate_lagged_expected_edge,
)


TS = 1_790_000_000.0


def _row(
    group: str,
    *,
    samples: int = 200,
    hit: float = 0.62,
    signed_bps: float = 30.0,
    after_cost_bps: float = 24.0,
    window_end: float = TS - 3600,
    generated_at: float = TS - 1800,
) -> LaggedReliabilityEvidence:
    return LaggedReliabilityEvidence(
        group=group,
        sample_count=samples,
        bayesian_hit_rate=hit,
        avg_signed_bps=signed_bps,
        avg_cost_adjusted_signed_bps=after_cost_bps,
        outcome_horizon_seconds=900,
        snapshot_window_end=window_end,
        snapshot_generated_at=generated_at,
    )


def _fresh(group: str, *, observed_at: float = TS - 60) -> EvidenceFreshness:
    return EvidenceFreshness(
        group=group,
        observed_at=observed_at,
        horizon="FAST_5_30M",
    )


def test_bounded_reliability_shrinks_toward_half() -> None:
    assert bounded_prospective_reliability(0, 1.0) == pytest.approx(0.5)
    mature = bounded_prospective_reliability(10_000, 1.0)
    assert 0.5 < mature <= 0.65


def test_two_mature_pit_groups_produce_reconciled_allow_edge() -> None:
    estimate = estimate_lagged_expected_edge(
        decision_timestamp=TS,
        direction=1,
        evidence_score=0.8,
        round_trip_cost_bps=5.0,
        reliability=(
            _row("price_structure", signed_bps=32.0),
            _row("derivatives", signed_bps=24.0, hit=0.58),
        ),
        freshness=(
            _fresh("price_structure"),
            _fresh("derivatives", observed_at=TS - 120),
        ),
        minimum_net_margin_bps=2.0,
    )

    assert estimate.pit_clear is True
    assert estimate.mature_group_count == 2
    assert estimate.expected_gross_move_bps is not None
    assert estimate.uncertainty_penalty_bps is not None
    assert estimate.event_decay_penalty_bps is not None
    assert estimate.expected_net_edge_bps == pytest.approx(
        estimate.expected_gross_move_bps
        - estimate.estimated_round_trip_cost_bps
        - estimate.uncertainty_penalty_bps
        - estimate.event_decay_penalty_bps
    )
    assert estimate.eligible is True
    assert estimate.recommendation == "ALLOW_EDGE"
    assert estimate.shadow_only is True
    assert estimate.live_execution is False
    assert estimate.automatic_promotion is False


def test_insufficient_mature_groups_fails_closed_without_edge() -> None:
    estimate = estimate_lagged_expected_edge(
        decision_timestamp=TS,
        direction=1,
        evidence_score=0.7,
        round_trip_cost_bps=4.0,
        reliability=(
            _row("price_structure", samples=99),
            _row("derivatives", samples=250),
        ),
        freshness=(_fresh("price_structure"), _fresh("derivatives")),
    )

    assert estimate.eligible is False
    assert estimate.recommendation == "INSUFFICIENT_LAGGED_EVIDENCE"
    assert estimate.expected_net_edge_bps is None
    assert estimate.mature_group_count == 1


def test_post_decision_reliability_is_rejected_and_marks_contamination() -> None:
    estimate = estimate_lagged_expected_edge(
        decision_timestamp=TS,
        direction=-1,
        evidence_score=0.8,
        round_trip_cost_bps=5.0,
        reliability=(
            _row("price_structure"),
            _row("derivatives", generated_at=TS + 1),
            _row("news_verified", signed_bps=20.0),
        ),
        freshness=(
            _fresh("price_structure"),
            _fresh("news_verified"),
        ),
    )

    assert estimate.pit_clear is False
    assert estimate.eligible is False
    assert estimate.recommendation == "CONTAMINATED_EVIDENCE"
    assert any("post-decision reliability" in reason for reason in estimate.reasons)


def test_future_source_observation_contaminates_otherwise_positive_edge() -> None:
    estimate = estimate_lagged_expected_edge(
        decision_timestamp=TS,
        direction=1,
        evidence_score=0.9,
        round_trip_cost_bps=3.0,
        reliability=(
            _row("price_structure", signed_bps=40.0),
            _row("derivatives", signed_bps=35.0),
        ),
        freshness=(
            _fresh("price_structure"),
            _fresh("derivatives", observed_at=TS + 0.001),
        ),
    )

    assert estimate.pit_clear is False
    assert estimate.eligible is False
    assert estimate.recommendation == "CONTAMINATED_EVIDENCE"
    assert any("future observation" in reason for reason in estimate.reasons)


def test_missing_cost_is_explicit_cost_unavailable() -> None:
    estimate = estimate_lagged_expected_edge(
        decision_timestamp=TS,
        direction=1,
        evidence_score=0.9,
        round_trip_cost_bps=None,
        reliability=(
            _row("price_structure"),
            _row("derivatives"),
        ),
        freshness=(
            _fresh("price_structure"),
            _fresh("derivatives"),
        ),
    )

    assert estimate.recommendation == "COST_UNAVAILABLE"
    assert estimate.eligible is False
    assert estimate.expected_gross_move_bps is None
    assert estimate.expected_net_edge_bps is None


def test_nonpositive_lagged_gross_edge_downgrades_to_wait() -> None:
    estimate = estimate_lagged_expected_edge(
        decision_timestamp=TS,
        direction=-1,
        evidence_score=0.9,
        round_trip_cost_bps=2.0,
        reliability=(
            _row("price_structure", signed_bps=-10.0),
            _row("derivatives", signed_bps=-5.0),
        ),
        freshness=(
            _fresh("price_structure"),
            _fresh("derivatives"),
        ),
    )

    assert estimate.eligible is False
    assert estimate.recommendation == "DOWNGRADE_TO_WAIT"
    assert estimate.expected_net_edge_bps is not None
    assert estimate.expected_net_edge_bps < 0
    assert any("non-positive gross" in reason for reason in estimate.reasons)


def test_duplicate_group_counts_once_using_strongest_mature_snapshot() -> None:
    estimate = estimate_lagged_expected_edge(
        decision_timestamp=TS,
        direction=1,
        evidence_score=0.8,
        round_trip_cost_bps=4.0,
        reliability=(
            _row("price_structure", samples=120, signed_bps=10.0),
            _row("price_structure", samples=300, signed_bps=35.0),
            _row("derivatives", samples=250, signed_bps=25.0),
        ),
        freshness=(
            _fresh("price_structure"),
            _fresh("derivatives"),
        ),
    )

    assert len(estimate.contributions) == 2
    selected = {row.group: row for row in estimate.contributions}
    assert selected["price_structure"].samples == 300
    assert selected["price_structure"].historical_gross_signed_bps == pytest.approx(35.0)


def test_eligible_mapping_contains_only_pit_clear_allow_edge_net_values() -> None:
    allowed = estimate_lagged_expected_edge(
        decision_timestamp=TS,
        direction=1,
        evidence_score=0.9,
        round_trip_cost_bps=3.0,
        reliability=(
            _row("price_structure", signed_bps=35.0),
            _row("derivatives", signed_bps=30.0),
        ),
        freshness=(
            _fresh("price_structure"),
            _fresh("derivatives"),
        ),
    )
    denied = estimate_lagged_expected_edge(
        decision_timestamp=TS,
        direction=1,
        evidence_score=0.9,
        round_trip_cost_bps=50.0,
        reliability=(
            _row("price_structure", signed_bps=5.0),
            _row("derivatives", signed_bps=5.0),
        ),
        freshness=(
            _fresh("price_structure"),
            _fresh("derivatives"),
        ),
    )

    mapping = eligible_expected_edge_bps_by_asset({
        "BTCUSDT": allowed,
        "ETHUSDT": denied,
    })

    assert mapping == {
        "BTCUSDT": pytest.approx(allowed.expected_net_edge_bps),
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"direction": 0}, "direction"),
        ({"evidence_score": 1.1}, "evidence_score"),
        ({"round_trip_cost_bps": -1.0}, "round_trip_cost_bps"),
        ({"minimum_net_margin_bps": -1.0}, "minimum_net_margin_bps"),
    ],
)
def test_invalid_estimator_inputs_fail_closed(kwargs, message) -> None:
    values = dict(
        decision_timestamp=TS,
        direction=1,
        evidence_score=0.8,
        round_trip_cost_bps=3.0,
        reliability=(
            _row("price_structure"),
            _row("derivatives"),
        ),
        freshness=(
            _fresh("price_structure"),
            _fresh("derivatives"),
        ),
        minimum_net_margin_bps=2.0,
    )
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        estimate_lagged_expected_edge(**values)
