from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STREAMS = ROOT / "monster-coins-pro" / "dip-expert-v4-streams.js"


def test_combined_depth_stream_keeps_symbol_context():
    src = STREAMS.read_text(encoding="utf-8")
    assert "@depth5@1000ms" in src
    assert "function v4DispatchWsPacket" in src
    assert "packet.stream.split('@')[0]" in src
    assert "m.s=streamSymbol.toUpperCase()" in src
    assert "v4DispatchWsPacket(e.data,'SPOT')" in src
    assert "v4DispatchWsPacket(e.data,'USDM_PERP')" in src


def test_depth_updates_reach_book_metrics_and_no_live_execution_endpoint():
    src = STREAMS.read_text(encoding="utf-8")
    assert "m.lastUpdateId!=null||m.e==='depthUpdate'" in src
    assert "v4BookUpdate" in src
    forbidden = ["/api/v3/order", "/fapi/v1/order", "newOrder", "order.place"]
    for token in forbidden:
        assert token not in src
