from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORKER = (ROOT / "supabase" / "functions" / "brian-dip-shadow-worker" / "index.ts").read_text(encoding="utf-8")
MIGRATION = (ROOT / "supabase" / "migrations" / "202609070071_brian_dip_server_authoritative_v7.sql").read_text(encoding="utf-8")
HANDOFF_MIGRATION = (ROOT / "supabase" / "migrations" / "202609070073_brian_dip_v7_handoff_lease_fence.sql").read_text(encoding="utf-8")
UI = (ROOT / "monster-coins-pro" / "dip-server-authoritative-v7.js").read_text(encoding="utf-8")
HTML = (ROOT / "monster-coins-pro" / "dip.html").read_text(encoding="utf-8")


def test_server_worker_is_cron_authorized_leased_and_shadow_only():
    assert 'requireCronAuth' in WORKER
    assert 'withCollectorLease' in WORKER
    assert 'brian-dip-shadow-worker-v7' in WORKER
    assert 'shadow_only: true' in WORKER
    assert 'live_execution: false' in WORKER
    assert '/api/v3/order' not in WORKER
    assert '/fapi/v1/order' not in WORKER
    assert 'BINANCE_PUBLIC_REST' in WORKER


def test_worker_has_brain_authoritative_dynamic_sizing_and_real_heat_check():
    assert 'BRAIN_CONFIDENCE_V7' in WORKER
    assert 'MAX_POSITION_FRACTION = 0.72' in WORKER
    assert 'MAX_PORTFOLIO_HEAT = 0.80' in WORKER
    assert 'riskBudgetPct' in WORKER
    assert 'desiredFraction' in WORKER
    assert 'openNotional(rt) + plan.notional > eq * MAX_PORTFOLIO_HEAT' in WORKER
    assert 'brain_quality' in WORKER
    assert 'actual_fraction' in WORKER
    assert 'risk_budget_pct' in WORKER


def test_worker_manages_lifecycle_without_browser_timers():
    for reason in ['HARD_5M_STRUCTURE_STOP', 'FLOW_INVALIDATION', 'HTF_INVALIDATION', 'PROFIT_TRAIL_V7', 'EXPERT_TARGET_V7', 'TIME_EXIT']:
        assert reason in WORKER
    assert 'LONG_HOLD_MS = 18 * 60_000' in WORKER
    assert 'SHORT_HOLD_MS = 22 * 60_000' in WORKER
    assert 'browserHeartbeatFresh' in WORKER
    assert 'WAIT_BROWSER_HANDOFF' in WORKER


def test_server_takes_append_only_ownership_and_fences_old_browser_writes():
    assert "brian_reject_browser_dip_write_after_server_takeover" in MIGRATION
    assert "browser snapshot rejected" in MIGRATION
    assert "browser event rejected" in MIGRATION
    assert "coalesce(new.metadata ->> 'server_v7', 'false') <> 'true'" in MIGRATION
    assert "brian-dip-shadow-worker-v7-1m" in MIGRATION
    assert "'* * * * *'" in MIGRATION
    assert "/functions/v1/brian-dip-shadow-worker" in MIGRATION


def test_v7_handoff_cannot_be_kept_alive_by_legacy_browser_heartbeat():
    assert 'brian_v7_neuter_browser_engine_heartbeat' in HANDOFF_MIGRATION
    assert "server_authoritative" in HANDOFF_MIGRATION
    assert "monster-coins-pro-dip-v4" in HANDOFF_MIGRATION
    assert "now() - interval '2 minutes'" in HANDOFF_MIGRATION
    assert 'brian_dip_v7_browser_heartbeat_fence' in HANDOFF_MIGRATION
    assert 'before insert or update of heartbeat_at, claimed_by' in HANDOFF_MIGRATION.lower()


def test_browser_is_view_only_and_server_handoff_loads_after_v5():
    assert 'v4Evaluate = function(){ return; }' in UI
    assert 'snapshot = async function(){ return; }' in UI
    assert "api('engine_check'" not in UI
    assert "const _v7Api = api" in UI
    assert "action==='engine_check'||action==='claim_engine'" in UI
    assert "SERVER_AUTHORITATIVE_VIEW_ONLY" in UI
    assert "server_authoritative:true" in UI
    assert "browser_execution:false" in UI
    assert "browser_view_only:true" in UI
    assert HTML.index('/dip-expert-v5-brain.js') < HTML.index('/dip-server-authoritative-v7.js')


def test_server_ui_javascript_parses_when_node_is_available():
    node = shutil.which('node')
    if node is None:
        return
    result = subprocess.run([node, '--check', str(ROOT / 'monster-coins-pro' / 'dip-server-authoritative-v7.js')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
