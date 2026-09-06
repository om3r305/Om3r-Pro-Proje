from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "202609060007_brian_sensor_reliability_prospective_calibration_v1.sql"


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_calibration_is_measurement_only_and_never_mutates_alpha_or_reliability():
    sql = _sql()
    assert "prospective_calibration_measurement_only" in sql
    assert "reliability_mutation_enabled', false" in sql
    assert "alpha_action_mutation_enabled', false" in sql
    assert "update public.brian_sensor_observations" not in sql
    assert "update public.brian_alpha_decisions" not in sql
    assert "delete from public.brian_sensor_observations" not in sql
    assert "live_execution boolean not null default false check (not live_execution)" in sql
    assert "shadow_only boolean not null default true check (shadow_only)" in sql


def test_calibration_uses_frozen_features_then_future_outcomes_with_causal_checks():
    sql = _sql()
    assert "from public.brian_alpha_reliability_shadow_features f" in sql
    assert "join public.brian_alpha_decision_outcomes o" in sql
    assert "rs.generated_at <= f.decision_observed_at" in sql
    assert "rs.window_end <= f.decision_observed_at" in sql
    assert "o.resolved_at >= f.decision_observed_at" in sql
    assert "snapshot_generated_at <= decision_observed_at" in sql
    assert "resolved_at >= decision_observed_at" in sql


def test_calibration_dedupes_reused_observations_per_outcome_horizon():
    sql = _sql()
    assert "partition by feat->>'observation_id', o.horizon_seconds" in sql
    assert "order by f.decision_observed_at asc, f.decision_id asc" in sql
    assert "unique (observation_id, outcome_horizon_seconds)" in sql
    assert "on conflict (observation_id, outcome_horizon_seconds) do nothing" in sql


def test_calibration_keeps_all_three_horizons_and_prior_metrics():
    sql = _sql()
    assert "o.horizon_seconds in (300, 900, 3600)" in sql
    assert "prior_bayesian_hit_rate_beta10_10" in sql
    assert "prior_avg_signed_bps" in sql
    assert "prior_median_signed_bps" in sql
    assert "prior_avg_cost_adjusted_signed_bps" in sql
    assert "realized_sensor_signed_bps" in sql


def test_calibration_is_append_only_and_runs_after_auditor_cadence():
    sql = _sql()
    assert "brian_sensor_reliability_prospective_calibration_append_only" in sql
    assert "execute function public.brian_reject_mutation()" in sql
    assert "brian-sensor-reliability-calibration-5m" in sql
    assert "'4-59/5 * * * *'" in sql
