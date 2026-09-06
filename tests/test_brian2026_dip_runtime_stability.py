from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "monster-coins-pro" / "dip.html").read_text(encoding="utf-8")
GUARD = (ROOT / "monster-coins-pro" / "dip-expert-v4-runtime-guard.js").read_text(encoding="utf-8")
SW = (ROOT / "monster-coins-pro" / "sw.js").read_text(encoding="utf-8")


def test_runtime_guard_loads_after_v4_hotfix_before_v5_brain():
    hotfix = HTML.index('/dip-expert-v4-hotfix.js')
    guard = HTML.index('/dip-expert-v4-runtime-guard.js')
    v5 = HTML.index('/dip-expert-v5-brain.js')
    assert hotfix < guard < v5


def test_market_loading_is_bounded_partial_and_overlay_fail_safe():
    assert 'AbortController' in GUARD
    assert 'V4_RUNTIME_FETCH_TIMEOUT_MS' in GUARD
    assert 'Promise.any' in GUARD
    assert 'Promise.allSettled' in GUARD
    assert 'finally' in GUARD
    assert "overlay.classList.remove('show')" in GUARD
    assert 'V4_RUNTIME_MIN_READY_MARKETS' in GUARD
    assert 'v4RuntimeSymbolReady' in GUARD


def test_transient_lease_fault_keeps_runtime_recoverable_but_blocks_new_entries():
    assert 'V4_RUNTIME_LEASE_GRACE_MS' in GUARD
    assert 'v4RuntimeNetworkish' in GUARD
    assert 'feedFresh' in GUARD
    assert 'running=true' in GUARD
    assert 'v4CloudFault=true' in GUARD
    assert 'yeni girişler fail-closed' in GUARD
    assert 'BRIAN_DIP_ENGINE_LEASE_ACTIVE' in GUARD
    assert 'running=false' in GUARD


def test_pwa_cache_contains_runtime_guard_and_is_bumped():
    assert "monster-coins-pro-shell-v7" in SW
    assert "/dip-expert-v4-runtime-guard.js" in SW
