from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_alpha_uses_canonical_radar_and_causal_windows():
    source = (ROOT / "supabase/functions/brian-alpha-decision-compiler/index.ts").read_text()
    assert 'from("brian_universe_snapshots")' in source
    assert 'brian_emergent_mover_frames' not in source
    assert '.eq("horizon", horizon)' in source
    assert '.limit(6000)' not in source
    assert 'ENABLE_DIP_DIRECTIONAL_EVIDENCE = false' in source
    assert 'if (ENABLE_DIP_DIRECTIONAL_EVIDENCE) await addDipEvidence' in source


def test_micro_latest_state_consumers_use_rpc():
    intrabar = (ROOT / "supabase/functions/brian-intrabar-eye/index.ts").read_text()
    sensor = (ROOT / "supabase/functions/brian-sensor-mesh/index.ts").read_text()
    for source in (intrabar, sensor):
        assert 'rpc("brian_latest_micro_book_ticks"' in source
    assert '.limit(1000)' not in intrabar
    assert '.limit(500)' not in sensor


def test_latest_state_migration_is_lateral_and_index_neutral():
    sql = (ROOT / "supabase/migrations/202609060001_brian_latest_micro_book_state_rpc.sql").read_text().lower()
    assert "cross join lateral" in sql
    assert "order by t.observed_at desc" in sql
    assert "limit 1" in sql
    assert "create index" not in sql
    assert "grant execute" in sql
