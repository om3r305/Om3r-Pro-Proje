from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "202609060003_brian_alpha_daily_sensor_partial_index.sql"


def test_daily_sensor_index_is_narrow_and_semantically_aligned():
    sql = MIGRATION.read_text(encoding="utf-8").lower()
    assert "brian_sensor_observations" in sql
    assert "(asset_id, observed_at desc)" in sql
    assert "horizon = 'daily'" in sql
    assert "available = true" in sql
    assert "independent_group <> 'news_gdelt'" in sql
    assert "micro_1_5m" not in sql.split("create index", 1)[1]
    assert "fast_5_30m" not in sql.split("create index", 1)[1]
