from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "202609060006_brian_alpha_reliability_feature_freeze_v1.sql"


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_feature_freeze_is_strictly_outcome_blind_and_no_effect():
    sql = _sql()
    assert "prospective_feature_freeze_no_alpha_effect" in sql
    assert "outcomes_consulted', false" in sql
    assert "brian_alpha_decision_outcomes" not in sql
    assert "update public.brian_alpha_decisions" not in sql
    assert "update public.brian_sensor_observations" not in sql
    assert "alpha_reliability_mutation_enabled', false" in sql
    assert "live_execution boolean not null default false check (not live_execution)" in sql
    assert "shadow_only boolean not null default true check (shadow_only)" in sql


def test_feature_freeze_only_uses_snapshots_available_before_decision():
    sql = _sql()
    assert "r.window_end <= d.observed_at" in sql
    assert "r.generated_at <= d.observed_at" in sql
    assert "snapshot_window_end <= decision_observed_at" in sql
    assert "snapshot_generated_at <= decision_observed_at" in sql
    assert "rs.generated_at <= c.decision_observed_at" in sql


def test_feature_freeze_preserves_all_three_shadow_outcome_metrics():
    sql = _sql()
    assert "outcome_horizon_seconds" in sql
    assert "bayesian_hit_rate_beta10_10" in sql
    assert "avg_signed_bps" in sql
    assert "median_signed_bps" in sql
    assert "avg_cost_adjusted_signed_bps" in sql
    assert "compiler_canonical_group" in sql
    assert "intrabar_tape" in sql


def test_feature_freeze_is_append_only_idempotent_and_frequent():
    sql = _sql()
    assert "brian_alpha_reliability_shadow_features_append_only" in sql
    assert "execute function public.brian_reject_mutation()" in sql
    assert "on conflict (decision_id) do nothing" in sql
    assert "brian-alpha-reliability-feature-freeze-1m" in sql
    assert "'* * * * *'" in sql
