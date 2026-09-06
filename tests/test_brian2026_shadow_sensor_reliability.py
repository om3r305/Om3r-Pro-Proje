from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "202609060005_brian_shadow_sensor_reliability_v1.sql"


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_shadow_learner_never_mutates_alpha_or_sensor_weights():
    sql = _sql()
    assert "measurement_only_no_alpha_weight_change" in sql
    assert "update public.brian_sensor_observations" not in sql
    assert "update public.brian_alpha_decisions" not in sql
    assert "delete from public.brian_sensor_observations" not in sql
    assert "live_execution boolean not null default false check (not live_execution)" in sql
    assert "shadow_only boolean not null default true check (shadow_only)" in sql


def test_shadow_learner_dedupes_reused_observations_and_is_causal():
    sql = _sql()
    assert "partition by s.observation_id, o.horizon_seconds" in sql
    assert "order by d.observed_at asc, d.decision_id asc" in sql
    assert "s.observed_at <= d.observed_at" in sql
    assert "o.resolved_at >= d.observed_at" in sql
    assert "s.direction <> 0" in sql
    assert "s.independent_group <> 'news_gdelt'" in sql


def test_shadow_learner_records_all_outcome_horizons_without_auto_mapping():
    sql = _sql()
    assert "o.horizon_seconds in (300, 900, 3600)" in sql
    assert "sensor_horizon" in sql
    assert "outcome_horizon_seconds" in sql
    assert "bayesian_hit_rate_beta10_10" in sql
    assert "avg_signed_bps" in sql
    assert "median_signed_bps" in sql
    assert "avg_cost_adjusted_signed_bps" in sql


def test_shadow_learner_runs_hourly_and_snapshots_are_append_only():
    sql = _sql()
    assert "brian-sensor-reliability-shadow-hourly" in sql
    assert "'12 * * * *'" in sql
    assert "brian_sensor_reliability_shadow_snapshots_append_only" in sql
    assert "execute function public.brian_reject_mutation()" in sql
    assert "on conflict (snapshot_id) do nothing" in sql
