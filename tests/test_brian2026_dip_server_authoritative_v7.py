from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORKER = (ROOT / "supabase" / "functions" / "brian-dip-shadow-worker" / "index.ts").read_text(encoding="utf-8")
FORESIGHT = (ROOT / "supabase" / "functions" / "brian-dip-foresight" / "index.ts").read_text(encoding="utf-8")
V7_MIGRATION = (ROOT / "supabase" / "migrations" / "202609070071_brian_dip_server_authoritative_v7.sql").read_text(encoding="utf-8")
V8_MIGRATION = (ROOT / "supabase" / "migrations" / "202609070074_brian_dip_v8_chart_reader.sql").read_text(encoding="utf-8")
HANDOFF_MIGRATION = (ROOT / "supabase" / "migrations" / "202609070073_brian_dip_v7_handoff_lease_fence.sql").read_text(encoding="utf-8")
SERVER_UI = (ROOT / "monster-coins-pro" / "dip-server-authoritative-v7.js").read_text(encoding="utf-8")
THESIS_UI = (ROOT / "monster-coins-pro" / "dip-foresight-v7.js").read_text(encoding="utf-8")
HTML = (ROOT / "monster-coins-pro" / "dip.html").read_text(encoding="utf-8")


# V8.1 split the worker into pure, behavior-tested modules. Keep these wiring checks,
# while actual state transitions, costs and outcomes run in the Deno/Postgres suites.
SHARED = (ROOT / 'supabase/functions/_shared/dip_v8.ts').read_text()
MARKET = (ROOT / 'supabase/functions/_shared/dip_v8_market.ts').read_text()
WORKER += ''.join(p.read_text() for p in (ROOT / 'supabase/functions/brian-dip-shadow-worker').glob('*.ts') if not p.name.endswith('.test.ts')) + SHARED + MARKET
FORESIGHT += (ROOT / 'supabase/functions/brian-dip-foresight/resolver.ts').read_text() + SHARED

def test_v8_worker_is_cron_authorized_leased_eth_only_and_shadow_only():
    assert 'requireCronAuth' in WORKER
    assert 'withCollectorLease' in WORKER
    assert 'brian-dip-shadow-worker-v8' in WORKER
    assert 'brian-dip-chart-reader-v8' in WORKER
    assert 'const SYMBOL = "ETHUSDT"' in WORKER
    assert 'live_execution: false' in WORKER
    assert 'shadow_only: true' in WORKER
    assert '/api/v3/order' not in WORKER
    assert '/fapi/v1/order' not in WORKER


def test_v8_worker_has_real_structure_single_thesis_and_safe_calibrating_size():
    for token in ['classifyPivots', 'HH', 'HL', 'LH', 'LL', 'bos', 'choch', 'sweep', 'failedBreak', 'equalHigh', 'equalLow']:
        assert token in WORKER
    assert '"4h"' in MARKET
    assert 'SWEEP_RECLAIM' in WORKER
    assert 'FAILED_BREAK' in WORKER
    assert 'BOS_RETEST' in WORKER
    assert 'MAX_POSITION_FRACTION_CALIBRATING = 0.08' in SHARED
    assert 'MAX_HEAT = 0.20' in SHARED
    assert 'MIN_CAL_SAMPLES = 40' in WORKER
    assert 'TARGET_BELOW_COST' in WORKER
    assert 'RR_TOO_LOW' in WORKER
    assert 'actual_fraction' in WORKER


def test_v8_zombie_thesis_is_locked_until_new_structure_and_new_closed_5m():
    assert 'THESIS_LOCKED_NO_NEW_STRUCTURE' in WORKER
    assert 'rt.lastLock' in WORKER
    assert 'last5m_t' in WORKER
    assert 'combinedFp' in WORKER
    assert 'INVALIDATION_FIRST' in WORKER
    assert 'EXPIRED_NO_BARRIER' in WORKER
    assert 'TARGET_FIRST' in WORKER


def test_v8_measurement_is_target_before_invalidation_and_has_no_sine_wave():
    assert 'target-before-invalidation-v8' in FORESIGHT
    assert 'TARGET_FIRST' in FORESIGHT
    assert 'INVALIDATION_FIRST' in FORESIGHT
    assert 'AMBIGUOUS' in FORESIGHT
    assert 'EXPIRED_NO_BARRIER' in FORESIGHT
    assert 'Math.sin' not in FORESIGHT
    assert 'Math.sin' not in THESIS_UI
    assert 'MIN_CAL_SAMPLES' not in FORESIGHT or 'calibration_samples' in FORESIGHT


def test_v8_migration_preserves_server_fence_and_runs_v8_crons():
    assert 'brian_dip_theses' in V8_MIGRATION
    assert 'server_v8' in V8_MIGRATION
    assert 'target-before-invalidation-v8' in V8_MIGRATION
    assert 'legacy-v7-direction-8m' in V8_MIGRATION
    assert 'brian-dip-shadow-worker-v8-1m' in V8_MIGRATION
    assert 'brian-dip-foresight-v8-1m' in V8_MIGRATION
    assert '/functions/v1/brian-dip-shadow-worker' in V8_MIGRATION
    assert '/functions/v1/brian-dip-foresight' in V8_MIGRATION
    assert 'brian_reject_browser_dip_write_after_server_takeover' in V7_MIGRATION


def test_v7_handoff_stays_as_browser_view_only_safety_fence():
    assert 'brian_v7_neuter_browser_engine_heartbeat' in HANDOFF_MIGRATION
    assert 'v4Evaluate = function(){ return; }' in SERVER_UI
    assert 'snapshot = async function(){ return; }' in SERVER_UI
    assert "action==='engine_check'||action==='claim_engine'" in SERVER_UI
    assert 'SERVER_AUTHORITATIVE_VIEW_ONLY' in SERVER_UI
    assert HTML.index('/dip-expert-v5-brain.js') < HTML.index('/dip-server-authoritative-v7.js')


def test_v8_ui_is_eth_only_single_thesis_and_has_no_fake_future_candles():
    assert "const V8_FOCUS_UNIVERSE=['ETHUSDT']" in THESIS_UI
    assert "symbols:['ETHUSDT']" in THESIS_UI
    assert "engine_version:'brian-dip-chart-reader-v8'" in THESIS_UI
    assert 'BRIAN V8 · ETH CHART THESIS' in THESIS_UI
    assert 'CALIBRATING' in THESIS_UI
    assert 'sahte gelecek mum yok' in THESIS_UI.lower()
    assert 'future.forEach' not in THESIS_UI
    assert 'Math.sin' not in THESIS_UI


def test_v8_browser_javascript_parses_when_node_is_available():
    node = shutil.which('node')
    if node is None:
        return
    for path in [ROOT / 'monster-coins-pro' / 'dip-server-authoritative-v7.js', ROOT / 'monster-coins-pro' / 'dip-foresight-v7.js']:
        result = subprocess.run([node, '--check', str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
