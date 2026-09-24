from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / (
    "supabase/migrations/"
    "20260924192000_brian_referenced_sensor_evidence_bridge.sql"
)
EXPORT = ROOT / (
    "supabase/functions/"
    "brian-realtime-referenced-sensor-export/index.ts"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8")


def _export() -> str:
    return EXPORT.read_text(encoding="utf-8")


def test_bridge_syncs_only_missing_sensor_ids_referenced_by_recent_alpha() -> None:
    sql = _sql().lower()

    assert "unnest(coalesce(d.source_observation_ids" in sql
    assert "left join public.brian_sensor_observations" in sql
    assert "s.observation_id is null" in sql
    assert "d.observed_at >= clock_timestamp() - p_lookback" in sql
    assert "limit v_limit" in sql
    assert "least(coalesce(p_limit, 500), 500)" in sql


def test_bridge_validates_returned_ids_and_shadow_boundary_before_insert() -> None:
    sql = _sql().lower()

    assert "brian_referenced_sensor_export_id_mismatch" in sql
    assert "observation_id = any(v_ids)" in sql
    assert "evidence_class = 'prospective_development_shadow'" in sql
    assert "shadow_only is true" in sql
    assert "live_execution is false" in sql
    assert "on conflict (observation_id) do nothing" in sql


def test_bridge_uses_internal_auth_and_exact_realtime_endpoint() -> None:
    sql = _sql()

    assert "where name = 'brian_cron_key'" in sql
    assert (
        "https://dliediwlldojkfjzlznm.supabase.co/functions/v1/"
        "brian-realtime-referenced-sensor-export"
    ) in sql
    assert "extensions.http_header('x-brian-internal-key',v_key)" in sql
    assert "CURLOPT_CONNECTTIMEOUT_MS" in sql
    assert "CURLOPT_TIMEOUT_MS" in sql


def test_bridge_runs_bounded_every_two_minutes_and_logs_outcomes() -> None:
    sql = _sql()

    assert "'brian-referenced-sensor-evidence-sync-2m'" in sql
    assert "'1-59/2 * * * *'" in sql
    assert "'brian-referenced-sensor-evidence-sync-v1'" in sql
    assert "'REFERENCED_SENSOR_SYNC_ERROR'" in sql
    assert "'remaining_selected'" in sql


def test_realtime_export_accepts_only_bounded_hash_ids() -> None:
    source = _export()

    assert "const MAX_IDS = 500;" in source
    assert "/^[a-f0-9]{64}$/i.test(item)" in source
    assert "[...new Set(ids)].slice(0, MAX_IDS)" in source
    assert '.in("observation_id", ids)' in source
    assert '.eq("evidence_class", "PROSPECTIVE_DEVELOPMENT_SHADOW")' in source
    assert '.eq("shadow_only", true)' in source
    assert '.eq("live_execution", false)' in source


def test_realtime_export_is_internal_only_and_has_no_trade_surface() -> None:
    source = _export().lower()

    assert 'x-brian-internal-key' in source
    assert '"unauthorized"' in source
    for forbidden in (
        "place_order",
        "create_order",
        "exchange_api_key",
        "live_execution: true",
        "automatic_promotion",
    ):
        assert forbidden not in source


def test_phase123_adds_no_mutation_of_existing_sensor_rows() -> None:
    sql = _sql().lower()

    assert "on conflict (observation_id) do nothing" in sql
    assert "update public.brian_sensor_observations" not in sql
    assert "delete from public.brian_sensor_observations" not in sql
