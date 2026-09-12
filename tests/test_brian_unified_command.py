from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "monster-coins-pro" / "frontier-v3.html").read_text(encoding="utf-8")
JS = (ROOT / "monster-coins-pro" / "frontier-v3.js").read_text(encoding="utf-8")
EDGE = (ROOT / "supabase" / "functions" / "brian-system-control" / "index.ts").read_text(encoding="utf-8")
SQL = (ROOT / "supabase" / "migrations" / "202609121730_brian_unified_system_control.sql").read_text(encoding="utf-8")
CRON_FIX = (ROOT / "supabase" / "migrations" / "202609121740_brian_unified_cron_control_fix.sql").read_text(encoding="utf-8")


def test_product_navigation_exposes_brian_and_dip_without_old_classic_dashboard():
    assert 'class="active" href="/">Brian' in HTML
    assert 'href="/dip"' in HTML
    assert '/classic.html' not in HTML
    assert '/evolution.html' not in HTML
    assert '/world.html' not in HTML
    assert '/treasury.html' not in HTML
    assert '/ocean.html' not in HTML


def test_frontier_has_real_treasury_and_global_controls():
    for amount in ("1000", "2000", "3000", "5000", "10000"):
        assert f'data-amount="{amount}"' in HTML
    assert 'id="startSystem"' in HTML
    assert 'id="restartSystem"' in HTML
    assert 'id="stopSystem"' in HTML
    assert "brian-system-control" in JS
    assert "starting_equity=S.selectedAmount" in JS


def test_brain_visual_uses_continuous_flows_and_animated_packets():
    assert 'class="brain-svg"' in HTML
    assert 'class="stream-core"' in HTML
    assert '<animateMotion' in HTML
    assert 'stroke-dasharray' not in HTML.split('class="flow-map"', 1)[1].split('</svg>', 1)[0]


def test_global_control_is_fail_closed_and_never_targets_dip_jobs():
    assert "j.jobname not like 'brian-dip-%'" in SQL
    assert "j.jobname not like 'brian-dip-%'" in CRON_FIX
    assert "cron.alter_job" in CRON_FIX
    assert "update cron.job" not in CRON_FIX.lower()
    assert "dip_touched',false" in SQL.lower()
    assert "dip_touched',false" in CRON_FIX.lower()
    assert 'TREASURY_REBASE_OPEN_POSITIONS' in SQL
    assert 'TREASURY_REBASE_REQUIRES_STOPPED_SYSTEM' in SQL
    assert 'shadow_only: true' in EDGE
    assert 'live_execution: false' in EDGE
    assert 'UNAUTHORIZED_DASHBOARD' in EDGE


def test_treasury_rebase_preserves_append_only_history():
    lower = SQL.lower()
    assert "previous_snapshot_id" in lower
    assert "operator_treasury_rebase" in lower
    assert "pg_advisory_xact_lock" in lower
    assert "insert into public.brian_treasury_shadow_snapshots" in lower
    assert "delete from public.brian_treasury_shadow_snapshots" not in lower
    assert "truncate public.brian_treasury_shadow_snapshots" not in lower
