from __future__ import annotations

import pytest

from brian2026.phase44_portfolio_brain import (
    PortfolioRiskLimits,
    PortfolioSignal,
    apply_hard_limits,
    blend_signals,
    construct_portfolio_book,
)


def _sig(model: str, asset: str, conviction: float, *, abstained: bool = False) -> PortfolioSignal:
    return PortfolioSignal(
        model_name=model,
        asset_id=asset,
        conviction=conviction,
        abstained=abstained,
        evidence_ids=() if abstained or conviction == 0.0 else (f"ev-{model}-{asset}",),
    )


def test_weighted_mean_and_abstain_semantics_match_upstream_behavior() -> None:
    result = blend_signals(
        (_sig("a", "BTC", 1.0), _sig("b", "BTC", 0.0)),
        {"a": 3.0, "b": 1.0},
        gross_target=1.0,
    )
    assert result.convictions["BTC"] == pytest.approx(0.75)

    result = blend_signals(
        (_sig("a", "BTC", 1.0), _sig("b", "BTC", 0.0, abstained=True)),
        {"a": 1.0, "b": 1.0},
        gross_target=1.0,
    )
    assert result.convictions["BTC"] == pytest.approx(1.0)


def test_real_zero_vote_dilutes_but_all_abstain_stays_flat() -> None:
    diluted = blend_signals(
        (_sig("a", "BTC", 1.0), _sig("b", "BTC", 0.0)),
        {"a": 1.0, "b": 1.0},
        gross_target=1.0,
    )
    assert diluted.convictions["BTC"] == pytest.approx(0.5)

    flat = blend_signals(
        (_sig("a", "BTC", 0.0, abstained=True), _sig("b", "BTC", 0.0, abstained=True)),
        {"a": 1.0, "b": 1.0},
        gross_target=1.0,
    )
    assert flat.convictions == {"BTC": 0.0}
    assert flat.requested_weights == {"BTC": 0.0}


def test_market_neutral_demeans_cross_section_and_preserves_ranking() -> None:
    result = blend_signals(
        (
            _sig("a", "BTC", 1.0),
            _sig("a", "ETH", 0.2),
            _sig("a", "SOL", -0.6),
        ),
        {"a": 1.0},
        gross_target=1.0,
        market_neutral=True,
    )
    assert sum(result.requested_weights.values()) == pytest.approx(0.0)
    assert sum(abs(value) for value in result.requested_weights.values()) == pytest.approx(1.0)
    assert result.requested_weights["BTC"] > 0 > result.requested_weights["SOL"]


def test_position_then_gross_clamp_only_shrinks_and_never_redistributes() -> None:
    limits = PortfolioRiskLimits(max_position_pct=0.25, max_gross_exposure=1.0)
    result = apply_hard_limits(
        {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5, "E": 0.5, "F": 0.5},
        limits,
    )
    assert all(abs(weight) <= 0.25 + 1e-12 for weight in result.weights.values())
    assert result.gross_exposure == pytest.approx(1.0)
    assert [event.limit for event in result.clamps].count("max_position_pct") == 6
    assert [event.limit for event in result.clamps].count("max_gross_exposure") == 1

    asymmetric = apply_hard_limits({"BTC": 0.9, "ETH": 0.05}, limits)
    assert asymmetric.weights["BTC"] == pytest.approx(0.25)
    assert asymmetric.weights["ETH"] == pytest.approx(0.05)
    assert asymmetric.cash_weight == pytest.approx(0.70)


def test_risk_stage_is_idempotent() -> None:
    limits = PortfolioRiskLimits(max_position_pct=0.30, max_gross_exposure=0.80)
    first = apply_hard_limits({"BTC": 0.7, "ETH": -0.5, "SOL": 0.2}, limits)
    second = apply_hard_limits(first.weights, limits)
    assert second.weights == pytest.approx(first.weights)
    assert second.clamps == ()


def test_constructed_book_preserves_evidence_and_never_promotes_or_executes() -> None:
    signals = (
        _sig("news", "BTC", 0.9),
        _sig("technical", "BTC", 0.7),
        _sig("news", "ETH", -0.4),
        _sig("technical", "ETH", -0.6),
    )
    plan = construct_portfolio_book(
        signals,
        {"news": 1.0, "technical": 2.0},
        gross_target=1.0,
        limits=PortfolioRiskLimits(max_position_pct=0.35, max_gross_exposure=0.65),
    )
    assert plan.risk.gross_exposure <= 0.65 + 1e-12
    assert plan.released_to_cash >= 0.0
    assert set(plan.source_evidence_ids) == {
        "ev-news-BTC", "ev-technical-BTC", "ev-news-ETH", "ev-technical-ETH"
    }
    assert plan.shadow_only is True
    assert plan.live_execution is False
    assert plan.automatic_promotion is False
