from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_dip_resume_preserves_session_and_runtime_snapshot():
    ui = read("monster-coins-pro/dip-expert-v4-session.js")
    edge = read("supabase/functions/brian-dip-trader/index.ts")
    assert "d.status==='RESUMED'" in ui
    assert "await status(false)" in ui
    assert "DIP_RESUME_FAILED_CLOSED" in ui
    assert 'const resumed=!restart&&!!previous&&!previous.active' in edge
    assert 'status:restart||previous?.active?"RESTARTED":resumed?"RESUMED":"STARTED"' in edge
    assert 'started_at:resumed&&previous?previous.start.requested_at' in edge
    assert 'starting_equity:Number(row?.starting_equity??starting)' in edge
    assert "history=[]" in ui  # fresh Start/Restart still gets a clean current-session view


def test_main_and_dip_pause_start_is_append_only_resume():
    sql = read("supabase/migrations/202609060070_brian_session_resume_contract.sql")
    assert "original.session_id,'START'" in sql
    assert "where session_id=latest.session_id and event_kind='START'" in sql
    assert "brian_dashboard_start_session" in sql
    assert "brian_dip_start_session" in sql
    lowered = sql.lower()
    assert "delete from public.brian_dashboard_session_events" not in lowered
    assert "delete from public.brian_dip_session_events" not in lowered
    assert "truncate" not in lowered


def test_shadow_only_contract_remains_explicit():
    edge = read("supabase/functions/brian-dip-trader/index.ts")
    sql = read("supabase/migrations/202609060070_brian_session_resume_contract.sql")
    assert "shadow_only:true,live_execution:false" in edge
    assert "true,false" in sql
    assert "/api/v3/order" not in edge
    assert "/fapi/v1/order" not in edge
