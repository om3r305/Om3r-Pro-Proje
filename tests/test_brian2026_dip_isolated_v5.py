from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V5 = (ROOT / "monster-coins-pro" / "dip-expert-v5-brain.js").read_text(encoding="utf-8")
DIP = (ROOT / "monster-coins-pro" / "dip.html").read_text(encoding="utf-8")
MAIN = (ROOT / "monster-coins-pro" / "index.html").read_text(encoding="utf-8")
BACKEND = (ROOT / "supabase" / "functions" / "brian-dip-trader" / "index.ts").read_text(encoding="utf-8")
SW = (ROOT / "monster-coins-pro" / "sw.js").read_text(encoding="utf-8")


def test_v5_is_archived_and_not_wired_to_live_dashboards():
    # V8.4.1 Authority owns /dip now. V5 may remain cached as an archived asset,
    # but it must not be executed by either the live DIP page or the main dashboard.
    assert "/dip-expert-v5-brain.js" not in DIP
    assert "/dip-expert-v5-brain.js" not in MAIN
    assert "/dip-v84.js" in DIP
    assert "/dip-v841-control.js" in DIP
    assert "/dip-v83.js" not in DIP
    assert "/dip-v83-liveview.js" not in DIP
    assert "/dip-expert-v5-brain.js" in SW


def test_v5_has_hard_cashbox_and_main_runtime_isolation():
    assert "DIP_SHADOW_CASHBOX_V5" in V5
    assert "DIP_BRAIN_MAIN_RUNTIME_MUTATION = false" in V5
    assert "mainRuntimeMutation:false" in V5
    assert "API.includes('/brian-dip-trader')" in V5
    assert "brian-control-center" not in V5


def test_v5_clones_brian_expert_roles_without_shared_mutable_state():
    for marker in (
        "structure: 0.32",
        "trend: 0.24",
        "momentum: 0.18",
        "volume: 0.12",
        "mean_reversion: 0.14",
        "microstructure: 0.20",
    ):
        assert marker in V5
    for setup in (
        "LIQUIDITY_SWEEP_REVERSAL",
        "BREAKOUT_RETEST_CONTINUATION",
        "PULLBACK_CONTINUATION",
        "RANGE_REJECTION",
        "TREND_EXHAUSTION",
    ):
        assert setup in V5


def test_v5_learning_and_controller_executor_contract():
    for marker in (
        "v5ControllerDecision",
        "MISSED_OPPORTUNITY_AUDIT_V5",
        "MISSED_PROFIT",
        "SAVED_LOSS",
        "CORRECT_ABSTENTION",
        "mfePct",
        "maePct",
        "HARD_DRIFT",
        "OUT_OF_DISTRIBUTION",
        "NET_EDGE_TOO_SMALL",
        "HYBRID_GATED",
        "_v5NativeOpen=v4Open",
        "_v5NativeManage=v4Manage",
    ):
        assert marker in V5


def test_v5_multi_horizon_and_net_cost_evidence_exist():
    for marker in ("'15s'", "'1m'", "'5m'", "'15m'", "grossEdgeBps", "costBps", "netEdgeBps"):
        assert marker in V5


def test_dip_backend_remains_dedicated_shadow_store():
    # The archived V8.3 control endpoint remains DIP-only and SHADOW-only even
    # after V8.4.1 Authority takes ownership of the live /dip page.
    for table in (
        "brian_dip_session_events",
        "brian_dip_v8_runtime",
        "brian_dip_v8_ledger",
    ):
        assert table in BACKEND
    assert 'const EVIDENCE="AGGRESSIVE_DIP_SHADOW"' in BACKEND
    assert "shadow_only:true" in BACKEND
    assert "live_execution:false" in BACKEND


def test_main_dashboard_source_is_not_rewired_to_v5():
    assert "DIP_SHADOW_CASHBOX_V5" not in MAIN
    assert "dip-expert-v5-brain" not in MAIN
