from __future__ import annotations

import pytest

from brian2026.global_sensor_mesh import (
    PROSPECTIVE_EVIDENCE_CLASS,
    SensorObservation,
    SensorTemplate,
    VirtualMicroBookReceipt,
    default_global_sensor_templates,
    expand_logical_eyes,
    mesh_manifest,
)
from brian2026.opportunity_tournament import TournamentConfig, build_candidate, run_tournament


def _obs(*, group: str, direction: int, strength: float, confidence: float = 0.8, reliability: float = 0.8, eye: str | None = None):
    return SensorObservation(
        eye_id=eye or f"eye-{group}",
        asset_id="crypto:XRPUSDT",
        observed_at=1_788_448_000.0,
        direction=direction,
        strength=strength,
        confidence=confidence,
        reliability=reliability,
        available=True,
        independent_group=group,
        source_ids=(f"source-{group}",),
        horizon="FAST_5_30M",
        reason="prospective test evidence",
    )


def test_default_crypto_templates_expand_to_700_logical_eyes_for_70_assets():
    templates = default_global_sensor_templates()
    assets = [f"crypto:ASSET{i}USDT" for i in range(70)]
    eyes = expand_logical_eyes(templates, assets, market_domain="crypto")
    assert len(templates) == 10
    assert len(eyes) == 700
    assert len({row.eye_id for row in eyes}) == 700
    assert all(row.shadow_only and not row.live_execution for row in eyes)


def test_mesh_manifest_counts_logical_eyes_without_spawning_processes():
    templates = default_global_sensor_templates()
    manifest = mesh_manifest(templates, {"crypto": ["BTC", "ETH", "XRP"]})
    assert manifest["logical_eye_count"] == 30
    assert manifest["virtual_micro_books"] is True
    assert manifest["live_execution"] is False


def test_unavailable_sensor_cannot_invent_direction():
    with pytest.raises(ValueError, match="unavailable sensors"):
        SensorObservation(
            eye_id="missing-news",
            asset_id="crypto:BTCUSDT",
            observed_at=1.0,
            direction=1,
            strength=0.5,
            confidence=0.5,
            reliability=0.5,
            available=False,
            independent_group="news",
            source_ids=(),
            horizon="FAST_5_30M",
            reason="provider unavailable",
        )


def test_available_sensor_requires_provenance():
    with pytest.raises(ValueError, match="provenance"):
        SensorObservation(
            eye_id="bad",
            asset_id="crypto:BTCUSDT",
            observed_at=1.0,
            direction=0,
            strength=0.0,
            confidence=0.0,
            reliability=0.0,
            available=True,
            independent_group="price",
            source_ids=(),
            horizon="FAST_5_30M",
            reason="missing provenance",
        )


def test_duplicate_independent_group_counts_as_one_vote():
    rows = (
        _obs(group="price", direction=1, strength=0.9, eye="price-a"),
        _obs(group="price", direction=1, strength=0.8, eye="price-b"),
        _obs(group="news", direction=1, strength=0.7),
    )
    candidate = build_candidate(rows)
    assert candidate.independent_groups == ("news", "price")
    assert len(candidate.supporting_observation_ids) == 2
    assert candidate.eligible is True


def test_one_family_alone_cannot_promote_to_tournament_candidate():
    candidate = build_candidate((_obs(group="price", direction=1, strength=1.0),))
    assert candidate.eligible is False
    assert candidate.virtual_ticket_usd == 0
    assert "insufficient independent evidence groups" in candidate.veto_reasons


def test_conflict_reduces_score_and_is_preserved():
    aligned = build_candidate((
        _obs(group="price", direction=1, strength=0.9),
        _obs(group="news", direction=1, strength=0.9),
    ))
    conflicted = build_candidate((
        _obs(group="price", direction=1, strength=0.9),
        _obs(group="news", direction=-1, strength=0.7),
    ))
    assert conflicted.opportunity_score < aligned.opportunity_score
    assert conflicted.conflicting_observation_ids


def test_tournament_virtual_cap_never_exceeds_500_and_has_no_live_execution():
    observations = []
    for i in range(40):
        asset = f"crypto:A{i}USDT"
        for group in ("price", "news", "orderbook"):
            observations.append(SensorObservation(
                eye_id=f"{asset}-{group}",
                asset_id=asset,
                observed_at=100.0,
                direction=1,
                strength=1.0,
                confidence=1.0,
                reliability=1.0,
                available=True,
                independent_group=group,
                source_ids=(f"src-{asset}-{group}",),
                horizon="FAST_5_30M",
                reason="strong test",
            ))
    result = run_tournament(observations, config=TournamentConfig(max_candidates=40))
    assert result.virtual_allocated_usd <= 500
    assert result.virtual_allocated_usd + result.virtual_unallocated_usd == pytest.approx(500)
    assert result.shadow_only is True
    assert result.live_execution is False


def test_micro_book_receipt_is_virtual_and_reconciles_pnl():
    receipt = VirtualMicroBookReceipt(
        eye_id="eye-1",
        asset_id="crypto:XRPUSDT",
        starting_equity=5.0,
        ending_equity=5.12,
        net_pnl=0.12,
        max_drawdown_pct=0.8,
        turnover_notional=10.0,
        trading_cost=0.01,
        active_decisions=4,
        wins=2,
        losses=1,
        horizon="FAST_5_30M",
    )
    assert receipt.evidence_class == PROSPECTIVE_EVIDENCE_CLASS
    assert receipt.shadow_only and not receipt.live_execution


def test_micro_book_rejects_non_preregistered_ticket():
    with pytest.raises(ValueError, match="preregistered"):
        VirtualMicroBookReceipt(
            eye_id="eye-1",
            asset_id="crypto:XRPUSDT",
            starting_equity=7.0,
            ending_equity=7.0,
            net_pnl=0.0,
            max_drawdown_pct=0.0,
            turnover_notional=0.0,
            trading_cost=0.0,
            active_decisions=0,
            wins=0,
            losses=0,
            horizon="FAST_5_30M",
        )


def test_sensor_template_cannot_enable_live_execution():
    with pytest.raises(ValueError, match="shadow-only"):
        SensorTemplate(
            "bad-live",
            "price_structure",
            "crypto",
            "FAST_5_30M",
            "price",
            "bad",
            5.0,
            shadow_only=True,
            live_execution=True,
        )
