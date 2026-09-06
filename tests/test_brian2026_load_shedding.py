from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "202609060002_brian_disable_gdelt_news_cron.sql"
ALPHA = ROOT / "supabase" / "functions" / "brian-alpha-decision-compiler" / "index.ts"


def test_legacy_gdelt_cron_is_disabled_not_deleted():
    sql = MIGRATION.read_text(encoding="utf-8").lower()
    assert "brian-news-eye-10m" in sql
    assert "cron.alter_job" in sql
    assert "active := false" in sql
    assert "cron.unschedule" not in sql
    assert "drop" not in sql


def test_alpha_still_excludes_gdelt_from_directional_evidence():
    source = ALPHA.read_text(encoding="utf-8")
    assert '.neq("independent_group", "news_gdelt")' in source
    assert 'gdelt_role: "discovery_only_no_direction_vote"' in source
