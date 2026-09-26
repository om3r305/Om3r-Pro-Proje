from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / (
    "supabase/migrations/"
    "20260924134000_brian_alpha_auditor_freshness_recovery.sql"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8")


def test_auditor_recovery_reserves_current_and_backlog_capacity() -> None:
    sql = _sql().lower()

    assert "recent_quota" in sql
    assert "/ 3.0" in sql
    assert "lane_quota" in sql
    assert "/ 6.0" in sql
    assert "order by e.observed_at desc" in sql
    assert "order by e.observed_at asc" in sql
    assert "missing_3600" in sql
    assert "missing_900" in sql
    assert "missing_300" in sql


def test_auditor_recovery_does_not_action_prioritize_calibration_queue() -> None:
    sql = _sql().lower()

    assert "action_priority" not in sql
    assert "case when d.action" not in sql


def test_auditor_schedule_is_restored_to_five_minutes_via_aux_backpressure() -> None:
    sql = _sql()

    assert "'brian-missed-opportunity-auditor-v3-5m'" in sql
    assert "'2-59/5 * * * *'" in sql
    assert "brian_private.enqueue_aux_service('missed_auditor')" in sql
    assert "/functions/v1/brian-missed-opportunity-auditor-v3" not in sql


def test_pending_audit_rpc_remains_service_role_only_and_bounded() -> None:
    sql = _sql().lower()

    assert "least(coalesce(p_limit, 120), 500)" in sql
    assert "revoke all on function public.brian_alpha_pending_audit_decisions" in sql
    assert "from anon;" in sql
    assert "from authenticated;" in sql
    assert "grant execute on function public.brian_alpha_pending_audit_decisions" in sql
    assert "to service_role;" in sql


def test_migration_keeps_shadow_only_scope_and_adds_no_trade_surface() -> None:
    sql = _sql().lower()

    assert "shadow only" in sql
    for forbidden in (
        "place_order",
        "create_order",
        "live_execution = true",
        "automatic_promotion",
        "exchange_api_key",
    ):
        assert forbidden not in sql

def test_reliability_measurement_cadences_return_to_declared_repository_rates() -> None:
    sql = _sql()

    assert "'brian-sensor-reliability-shadow-hourly'" in sql
    assert "'12 * * * *'" in sql
    assert "brian_refresh_sensor_reliability_shadow" in sql
    assert "'brian-sensor-reliability-calibration-5m'" in sql
    assert "'4-59/5 * * * *'" in sql
    assert "brian_resolve_sensor_reliability_prospective_calibration" in sql
