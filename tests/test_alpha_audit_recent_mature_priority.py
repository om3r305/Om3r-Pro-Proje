from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "202609060007_brian_alpha_audit_recent_mature_priority.sql"


def test_each_horizon_has_reserved_recent_mature_priority():
    src = MIGRATION.read_text(encoding="utf-8")
    assert "q60 as" in src and "q15 as" in src and "q5 as" in src
    assert src.count("order by e.observed_at desc, e.decision_id desc") >= 3
    assert "missing_300" in src and "missing_900" in src and "missing_3600" in src


def test_policy_and_execution_are_not_mutated():
    src = MIGRATION.read_text(encoding="utf-8")
    forbidden = ["live_execution = true", "update public.brian_alpha_decisions", "evidence_score =", "phase37"]
    for token in forbidden:
        assert token not in src.lower()
