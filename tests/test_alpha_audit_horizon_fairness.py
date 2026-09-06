from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "202609060006_brian_alpha_audit_fair_horizon_queue.sql"


def test_audit_queue_reserves_capacity_for_all_horizons():
    src = MIGRATION.read_text(encoding="utf-8")
    assert "missing_300" in src
    assert "missing_900" in src
    assert "missing_3600" in src
    assert "q60 as" in src
    assert "q15 as" in src
    assert "q5 as" in src
    assert "limit (select quota from cfg)" in src


def test_auditor_runs_each_minute_without_mutating_alpha_policy():
    src = MIGRATION.read_text(encoding="utf-8")
    assert "jobname='brian-missed-opportunity-auditor-v3-5m'" in src
    assert "schedule := '* * * * *'" in src
    assert "brian_alpha_pending_audit_decisions" in src
    # This migration is measurement/audit only; decision compiler policy must not be changed here.
    forbidden = ["evidence_score =", "estimated_round_trip_cost_bps =", "brian_alpha_shadow_position_book", "live_execution = true"]
    for token in forbidden:
        assert token not in src
