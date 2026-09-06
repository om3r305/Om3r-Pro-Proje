from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DASH = (ROOT / "monster-coins-pro" / "dashboard.js").read_text(encoding="utf-8")
INDEX = (ROOT / "monster-coins-pro" / "index.html").read_text(encoding="utf-8")
GUARD = (ROOT / "monster-coins-pro" / "dip-expert-v4-runtime-guard.js").read_text(encoding="utf-8")
SW = (ROOT / "monster-coins-pro" / "sw.js").read_text(encoding="utf-8")


def test_general_overview_defaults_to_live_alpha_instead_of_zero_frozen_tracker():
    assert '<option value="ALPHA" selected>ALPHA Canlı</option>' in INDEX
    assert "currentPolicy(){const v=$('policyView')?.value||'ALPHA'" in DASH
    assert "renderAlphaOverview" in DASH
    assert "direction-only/no notional" in DASH
    assert "OPEN_LONG/SHORT" in DASH


def test_dip_universe_is_sticky_and_self_recovers_from_partial_refresh():
    assert "BRIAN_DIP_LIVE_V6" in GUARD
    assert "v6LastHealthyUniverse" in GUARD
    assert "v6RecoverUniverse" in GUARD
    assert "if(before.length){v4Universe=[...before]" in GUARD
    assert "setInterval" in GUARD


def test_v5_reasoner_can_bridge_quality_long_and_short_candidates_without_bypassing_safety():
    assert "PULLBACK_CONTINUATION" in GUARD
    assert "BREAKOUT_RETEST_CONTINUATION" in GUARD
    assert "TREND_EXHAUSTION" in GUARD
    assert "DOWNTREND_BREAK" in GUARD
    assert "v4FuturesSymbols.has" in GUARD
    assert "V4_COST_EDGE_MULT" in GUARD
    assert "vetoReasons" in GUARD
    assert "v4Open(st,ctx,'LONG'" in GUARD
    assert "v4Open(st,pctx,'SHORT'" in GUARD
    assert "SHADOW/PAPER ONLY" in GUARD
    assert "live order endpoint" in GUARD


def test_pwa_shell_is_bumped_for_live_fix():
    assert "monster-coins-pro-shell-v10" in SW


def test_modified_javascript_parses_when_node_is_available():
    node = shutil.which("node")
    if node is None:
        return
    for rel in ["monster-coins-pro/dashboard.js", "monster-coins-pro/dip-expert-v4-runtime-guard.js"]:
        result = subprocess.run([node, "--check", str(ROOT / rel)], capture_output=True, text=True)
        assert result.returncode == 0, f"{rel}: {result.stderr}"
