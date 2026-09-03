from __future__ import annotations

import json
from pathlib import Path

import pytest

from brian2026.global_sensor_mesh import SensorObservation
from brian2026.intrabar_reaction import (
    INTRABAR_EXPERIMENT_ID,
    IntrabarReactionConfig,
    build_intrabar_consensus,
    intrabar_reaction_templates,
)

ROOT = Path(__file__).resolve().parents[1]
EDGE = ROOT / "supabase" / "functions" / "brian-intrabar-eye" / "index.ts"
MIGRATION = ROOT / "supabase" / "migrations" / "202609030008_brian_phase40_intrabar_reaction.sql"
CONSENSUS_FIXTURES = ROOT / "tests" / "fixtures" / "intrabar_consensus_cases.json"


def _obs(group: str, direction: int, strength: float, *, confidence: float = 0.95, reliability: float = 0.5) -> SensorObservation:
    return SensorObservation(
        eye_id=f"eye-{group}",
        asset_id="crypto:XRPUSDT",
        observed_at=1_788_450_000.0,
        direction=direction,
        strength=strength,
        confidence=confidence,
        reliability=reliability,
        available=True,
        independent_group=group,
        source_ids=(f"source-{group}",),
        horizon="MICRO_1_5M",
        reason="prospective intrabar test",
    )


def test_intrabar_templates_are_independent_micro_shadow_eyes():
    templates = intrabar_reaction_templates()
    assert len(templates) == 5
    assert len({row.independent_group for row in templates}) == 5
    assert all(row.horizon == "MICRO_1_5M" for row in templates)
    assert all(row.shadow_only and not row.live_execution for row in templates)
    assert {row.template_id for row in templates} == {
        "velocity-micro",
        "volume-burst-micro",
        "breakout-micro",
        "reclaim-micro",
        "taker-flow-micro",
    }


def test_intrabar_consensus_requires_two_independent_groups():
    one = build_intrabar_consensus((_obs("micro_velocity", 1, 1.0),))
    assert one.eligible is False
    two = build_intrabar_consensus((
        _obs("micro_velocity", 1, 1.0),
        _obs("micro_taker_flow", 1, 1.0),
    ))
    assert two.direction == 1
    assert two.eligible is True
    assert two.status == "ACTIONABLE_SHADOW"


def test_intrabar_conflict_reduces_consensus_quality():
    aligned = build_intrabar_consensus((
        _obs("micro_velocity", 1, 1.0),
        _obs("micro_volume", 1, 1.0),
        _obs("micro_taker_flow", 1, 1.0),
    ))
    conflicted = build_intrabar_consensus((
        _obs("micro_velocity", 1, 1.0),
        _obs("micro_volume", 1, 1.0),
        _obs("micro_taker_flow", -1, 1.0),
    ))
    assert conflicted.score < aligned.score
    assert conflicted.conflict_groups


def test_overextended_decelerating_move_is_not_chased():
    result = build_intrabar_consensus(
        (
            _obs("micro_velocity", 1, 1.0),
            _obs("micro_volume", 1, 1.0),
            _obs("micro_breakout", 1, 1.0),
        ),
        extension_sigma=4.2,
        decelerating=True,
        fresh_velocity_direction=1,
        taker_flow_direction=1,
    )
    assert result.eligible is True
    assert result.late_chase is True
    assert result.status == "VETOED_LATE_CHASE"


def test_phase40_runtime_is_one_minute_partial_bar_and_public_market_only():
    text = EDGE.read_text(encoding="utf-8")
    assert 'const EXPERIMENT_ID = "phase40-intrabar-reaction-v1"' in text
    assert 'const TOP_N = 50' in text
    assert 'interval=1m' in text
    assert '/api/v3/aggTrades' in text
    assert 'current_1m_partial' in text
    assert 'MIN_INTERVAL_SECONDS = 50' in text
    assert 'VETOED_LATE_CHASE' in text
    assert 'live_execution: false' in text
    assert '/api/v3/order' not in text
    assert 'createOrder' not in text
    assert 'place_order' not in text


def test_intrabar_prior_lookup_is_batched_and_errors_are_serialized():
    text = EDGE.read_text(encoding="utf-8")
    assert 'const PRIOR_LOOKUP_BATCH = 40' in text
    assert 'i += PRIOR_LOOKUP_BATCH' in text
    assert '.in("eye_id", chunk)' in text
    assert '.in("eye_id", allEyeIds)' not in text
    assert 'function errorText(error: unknown): string' in text
    assert 'error: message' in text


def test_phase40_migration_is_append_only_shadow_and_does_not_mutate_phase37():
    text = MIGRATION.read_text(encoding="utf-8")
    assert INTRABAR_EXPERIMENT_ID in text
    assert "brian-intrabar-eye-1m" in text
    assert "'* * * * *'" in text
    assert "brian_intrabar_reaction_events_append_only" in text
    assert "live_execution boolean not null default false check (not live_execution)" in text
    assert "learning_enabled', false" in text
    assert "historical_backfill_allowed', false" in text
    forbidden_mutations = (
        "update public.brian_live_shadow_ticks",
        "delete from public.brian_live_shadow_ticks",
        "truncate public.brian_live_shadow_ticks",
        "update public.brian_live_shadow_experiments",
        "delete from public.brian_live_shadow_experiments",
    )
    lowered = text.lower()
    assert all(token not in lowered for token in forbidden_mutations)


def test_intrabar_config_keeps_independent_confirmation_and_no_sub_30s_cron_assumption():
    cfg = IntrabarReactionConfig()
    assert cfg.cadence_seconds == 60
    assert cfg.scan_top_n == 50
    assert cfg.min_support_groups >= 2
    assert cfg.overextension_sigma == 3.5


def _load_consensus_fixtures() -> list[dict]:
    payload = json.loads(CONSENSUS_FIXTURES.read_text(encoding="utf-8"))
    return payload["cases"]


def _obs_from_fixture(row: dict) -> SensorObservation:
    return SensorObservation(
        eye_id=f"eye-{row['independent_group']}",
        asset_id="crypto:XRPUSDT",
        observed_at=1_788_450_000.0,
        direction=row["direction"],
        strength=row["strength"],
        confidence=row["confidence"],
        reliability=row["reliability"],
        available=True,
        independent_group=row["independent_group"],
        source_ids=(f"source-{row['independent_group']}",),
        horizon="MICRO_1_5M",
        reason="shared fixture case",
    )


@pytest.mark.parametrize("case", _load_consensus_fixtures(), ids=lambda case: case["name"])
def test_build_intrabar_consensus_matches_shared_fixtures(case: dict) -> None:
    """Runs brian2026/intrabar_reaction.py::build_intrabar_consensus against the same JSON
    fixtures consumed by supabase/functions/brian-intrabar-eye/logic.test.ts (deployed TS), so
    the Python reference model and the production runtime are checked against identical inputs.
    Cases marked shared_semantics: False are the one currently-known divergence (brian-2026 issue
    #32 Sec 3.4) and assert this implementation's real, current behavior via python_expected --
    not a target the two implementations have been made to agree on yet.
    """
    observations = [_obs_from_fixture(row) for row in case["observations"]]
    result = build_intrabar_consensus(
        observations,
        extension_sigma=case["extension_sigma"],
        decelerating=case["decelerating"],
        fresh_velocity_direction=case["fresh_velocity_direction"],
        taker_flow_direction=case["taker_flow_direction"],
    )
    expected = case["expected"] if case["shared_semantics"] else case["python_expected"]

    assert result.direction == expected["direction"], case["name"]
    assert result.eligible == expected["eligible"], case["name"]
    if "late_chase" in expected:
        assert result.late_chase == expected["late_chase"], case["name"]
    if "status" in expected:
        assert result.status == expected["status"], case["name"]
    if "score" in expected:
        assert result.score == pytest.approx(expected["score"], abs=1e-9), case["name"]
    if "support_groups" in expected:
        assert list(result.support_groups) == expected["support_groups"], case["name"]
    if "conflict_groups" in expected:
        assert list(result.conflict_groups) == expected["conflict_groups"], case["name"]
