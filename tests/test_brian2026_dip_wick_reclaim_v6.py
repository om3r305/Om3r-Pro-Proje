from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAIN = (ROOT / "monster-coins-pro" / "dip-expert-v5-brain.js").read_text(encoding="utf-8")
SW = (ROOT / "monster-coins-pro" / "sw.js").read_text(encoding="utf-8")


def test_v5_runtime_escape_bug_is_closed():
    assert "function v5Esc(s)" in BRAIN
    assert "v5Esc(d.setup||'WAIT')" in BRAIN


def test_intrabar_wick_reclaim_uses_live_bar_local_structure_and_atr():
    assert "BRIAN_DIP_WICK_RECLAIM_V6" in BRAIN
    assert "tail&&tail.closed===false" in BRAIN
    assert "closed.slice(-6)" in BRAIN
    assert "v4TapeSpot?.[sym]" in BRAIN
    assert "sweepNeedPct=v4Clamp(atrPct*.10,.012,.12)" in BRAIN
    assert "reclaimNeedPct=v4Clamp(atrPct*.035,.006,.06)" in BRAIN
    assert "WICK_SWEEP_WAIT_RECLAIM_LONG" in BRAIN
    assert "WICK_SWEEP_RECLAIM_LONG" in BRAIN


def test_pattern_classification_does_not_bypass_executor_gates():
    assert "return v6BaseSetup(sym,ctx,p,dir)" in BRAIN
    assert "LIQUIDITY_SWEEP_REVERSAL" in BRAIN
    assert "FAILED_BREAK_REVERSAL" in BRAIN
    # No real execution endpoint may be introduced in the isolated browser brain.
    assert "/api/v3/order" not in BRAIN
    assert "live_execution=true" not in BRAIN


def test_wick_candidates_are_append_only_telemetry_and_cache_refreshes():
    assert "WICK_RECLAIM_CANDIDATE" in BRAIN
    assert "monster-coins-pro-shell-v10" in SW
