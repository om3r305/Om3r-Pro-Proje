from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "monster-coins-pro" / "dip.html").read_text(encoding="utf-8")
GUARD = (ROOT / "monster-coins-pro" / "dip-expert-v4-runtime-guard.js").read_text(encoding="utf-8")
SERVER_UI = (ROOT / "monster-coins-pro" / "dip-server-authoritative-v7.js").read_text(encoding="utf-8")
SW = (ROOT / "monster-coins-pro" / "sw.js").read_text(encoding="utf-8")


def test_v83_live_page_loads_only_v83_runtime_and_liveview_in_order():
    runtime = HTML.index('/dip-v83.js')
    liveview = HTML.index('/dip-v83-liveview.js')
    assert runtime < liveview
    for legacy in ('/dip-expert-v4-hotfix.js','/dip-expert-v4-runtime-guard.js','/dip-expert-v5-brain.js','/dip-server-authoritative-v7.js'):
        assert legacy not in HTML


def test_market_loading_is_bounded_partial_and_overlay_fail_safe():
    assert 'AbortController' in GUARD
    assert 'V4_RUNTIME_FETCH_TIMEOUT_MS' in GUARD
    assert 'Promise.any' in GUARD
    assert 'Promise.allSettled' in GUARD
    assert 'finally' in GUARD
    assert "overlay.classList.remove('show')" in GUARD
    assert 'V4_RUNTIME_MIN_READY_MARKETS' in GUARD
    assert 'v4RuntimeSymbolReady' in GUARD


def test_legacy_browser_lease_recovery_remains_available_before_server_handoff():
    assert 'V4_RUNTIME_LEASE_GRACE_MS' in GUARD
    assert 'v4RuntimeNetworkish' in GUARD
    assert 'BRIAN_DIP_ENGINE_LEASE_ACTIVE' in GUARD


def test_v7_browser_is_view_only_after_all_v4_v5_wrappers():
    assert 'BRIAN_DIP_SERVER_AUTHORITATIVE_V7 = true' in SERVER_UI
    assert 'v4Evaluate = function(){ return; }' in SERVER_UI
    assert 'snapshot = async function(){ return; }' in SERVER_UI
    assert "api('engine_check'" not in SERVER_UI
    assert "browser_view_only:true" in SERVER_UI
    assert 'SERVER 24/7' in SERVER_UI


def test_pwa_cache_contains_runtime_guard_server_handoff_and_is_bumped():
    assert "monster-coins-pro-shell-v12" in SW
    assert "/dip-expert-v4-runtime-guard.js" in SW
    assert "/dip-server-authoritative-v7.js" in SW
