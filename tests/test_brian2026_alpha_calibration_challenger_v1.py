from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUNCTION = (ROOT / "supabase" / "functions" / "brian-alpha-calibration-challenger" / "index.ts").read_text(encoding="utf-8")
MIGRATION = (ROOT / "supabase" / "migrations" / "202609070072_brian_alpha_calibration_challenger_v1.sql").read_text(encoding="utf-8")


def test_challenger_is_shadow_only_and_never_mutates_canonical_alpha():
    assert 'canonical_mutation: false' in FUNCTION
    assert 'shadow_only: true' in FUNCTION
    assert 'live_execution: false' in FUNCTION
    assert '/api/v3/order' not in FUNCTION
    assert '/fapi/v1/order' not in FUNCTION
    assert '.update(' not in FUNCTION
    assert '.delete(' not in FUNCTION


def test_challenger_uses_mature_prospective_reliability_and_cost_adjusted_evidence():
    assert 'brian_sensor_reliability_shadow_snapshots' in FUNCTION
    assert 'MIN_MATURE_SAMPLES = 100' in FUNCTION
    assert 'bayesian_hit_rate_beta10_10' in FUNCTION
    assert 'avg_cost_adjusted_signed_bps' in FUNCTION
    for action in ['KEEP_VETO', 'ALLOW_ACTION', 'DOWNGRADE_TO_WAIT', 'PROMOTE_CANDIDATE_LONG', 'PROMOTE_CANDIDATE_SHORT', 'KEEP_WAIT']:
        assert action in FUNCTION


def test_challenger_table_is_append_only_rls_and_hard_shadow_bounded():
    assert 'create table if not exists public.brian_alpha_calibration_challenger' in MIGRATION
    assert 'enable row level security' in MIGRATION
    assert 'brian_alpha_calibration_challenger_append_only' in MIGRATION
    assert 'public.brian_reject_mutation()' in MIGRATION
    assert "check (shadow_only)" in MIGRATION
    assert "check (not live_execution)" in MIGRATION
    assert 'grant select,insert' in MIGRATION
    assert 'revoke update,delete,truncate,references,trigger' in MIGRATION


def test_challenger_cron_is_idempotent_and_calls_only_its_shadow_function():
    assert "brian-alpha-calibration-challenger-v1-5m" in MIGRATION
    assert "'4-59/5 * * * *'" in MIGRATION
    assert "/functions/v1/brian-alpha-calibration-challenger" in MIGRATION
    assert "x-brian-cron-key" in MIGRATION
