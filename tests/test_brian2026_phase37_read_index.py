from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "202609060004_brian_phase37_recent_read_partial_index.sql"


def test_phase37_read_index_is_partial_and_behavior_neutral():
    sql = MIGRATION.read_text(encoding="utf-8").lower()
    assert "brian_live_shadow_ticks" in sql
    assert "(observed_at desc)" in sql
    assert "phase37-prospective-live-20260903" in sql
    assert "policy_kind in ('native', 'profit')" in sql
    assert "update public.brian_live_shadow" not in sql
    assert "delete from public.brian_live_shadow" not in sql
    assert "truncate" not in sql
