from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "monster-coins-pro" / "dip-expert-v4-engine.js"
STREAMS = ROOT / "monster-coins-pro" / "dip-expert-v4-streams.js"
DIP = ROOT / "monster-coins-pro" / "dip.js"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_chart_market_data_is_direct_binance_spot_1m_and_aggtrade():
    dip = text(DIP)
    streams = text(STREAMS)
    assert "wss://stream.binance.com:9443" in streams
    assert "@aggTrade" in streams
    assert "@kline_1m" in streams
    assert "Binance Spot · 1m" in dip


def test_displayed_live_price_is_not_overwritten_by_orderbook_midpoint():
    streams = text(STREAMS)
    assert "live[sym]=(Number(m.b)+Number(m.a))/2" not in streams
    assert "const mid=(Number(m.b)+Number(m.a))/2;v4Evaluate(sym,mid)" in streams
    assert "if(venue==='SPOT')tick(sym,Number(m.p))" in streams


def test_dip_reference_follows_real_binance_trade_low_while_armed():
    engine = text(ENGINE)
    assert "if(!(armLow>0)||p<armLow)st.v4.armLow=p" in engine
    assert "st.dip=Number(st.v4.armLow||p)" in engine
    assert "if(p<Number(st.v4.armLow||p)){st.v4.armLow=p;st.dip=p;}" in engine
    assert "else if(!st.pos){st.dip=null;}" in engine


def test_modified_javascript_parses_when_node_is_available():
    node = shutil.which("node")
    if node is None:
        return
    for path in [ENGINE, STREAMS]:
        result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, f"{path.name}: {result.stderr}"
