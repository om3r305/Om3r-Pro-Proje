# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, time, os, json
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
from collections import defaultdict

CACHE = {
    "last_ts": 0.0,
    "slot_perf": {},      # {"dip": {"pnl":..,"n":..,"wins":..,"pf":..,"wr":..}, ...}
    "sym_perf": {},       # {"BTCUSDT": {...}}
    "open_count": 0,
    "cash": None,
}

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
TRADES = LOGS / "trades.csv"
EVENTS = LOGS / "events.csv"
STATE  = ROOT / "runtime" / "state.json"

def _safe_float(x, d=0.0):
    try: return float(x)
    except: return d

def _read_csv_tail(path: Path, max_rows: int = 5000) -> List[List[str]]:
    if not path.exists(): return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        rows = f.readlines()[-max_rows:]
    return [r.strip().split(",") for r in rows if r.strip()]

def _read_trades() -> Tuple[List[str], List[Dict[str, str]]]:
    if not TRADES.exists(): return [], []
    with TRADES.open("r", encoding="utf-8", errors="ignore") as f:
        rd = csv.DictReader(f)
        rows = [r for r in rd]
    return (rd.fieldnames or []), rows

def _read_state_cash_open() -> Tuple[Optional[float], int]:
    try:
        if STATE.exists():
            st = json.loads(STATE.read_text(encoding="utf-8", errors="ignore"))
            cash = st.get("cash")
            open_positions = 0
            for sym, slots in (st.get("positions") or {}).items():
                for k, v in (slots or {}).items():
                    if v: open_positions += 1
            return _safe_float(cash, None), int(open_positions)
    except Exception:
        pass
    return None, 0

def _perf_from_trades(trades: List[Dict[str,str]]) -> Tuple[Dict[str,dict], Dict[str,dict]]:
    slot_stat = defaultdict(lambda: {"pnl":0.0,"n":0,"wins":0})
    sym_stat  = defaultdict(lambda: {"pnl":0.0,"n":0,"wins":0})
    for r in trades:
        try:
            side = (r.get("side") or "").upper()
            if side not in ("SELL","CLOSE"):  # realize anlarında PnL görünüyor genelde
                # bazı sistemlerde SELL satırında PnL yok, CLOSE satırına yazılıyor.
                pass
            slot = r.get("slot") or r.get("reason") or ""
            if slot and ":" in slot:  # "PLUG:..." sadeleştir
                slot = slot.split(":")[0].strip()
            sym  = r.get("sym") or r.get("symbol") or ""
            pnl  = _safe_float(r.get("pnl") or r.get("realized_pnl") or 0.0, 0.0)
            if pnl == 0.0 and (r.get("side","").upper() == "SELL"):
                # SELL satırında pnl yoksa atla, bazı csv’lerde CLOSE ek satır var
                continue
            # close satırları için basit kural: pnl>0 win say
            win = 1 if pnl > 0 else 0

            if slot:
                s = slot_stat[slot]
                s["pnl"] += pnl; s["n"] += 1; s["wins"] += win
            if sym:
                s = sym_stat[sym]
                s["pnl"] += pnl; s["n"] += 1; s["wins"] += win
        except Exception:
            continue

    # türev metrikler
    def _finish(d: Dict[str,dict]) -> Dict[str,dict]:
        out = {}
        for k, v in d.items():
            n = max(1, int(v["n"]))
            wr = v["wins"]/n
            # kaba PF tahmini: toplam pozitif pnl / |toplam negatif pnl|
            pos = 0.0; neg = 0.0
            # burada trade trade gezmek daha doğru ama aggregate yetmezse ~
            # hızlı tahmin için toplam pnl ve winrate'ı kullanıyoruz:
            # PF ~ wr / (1-wr) ama çok uçlarda stabilize et:
            pf = (wr / max(1e-6, 1.0-wr))
            out[k] = {"pnl": v["pnl"], "n": v["n"], "wins": v["wins"], "wr": wr, "pf": pf}
        return out

    return _finish(slot_stat), _finish(sym_stat)

def collect_and_cache_metrics(cfg: dict) -> None:
    """
    trades.csv’den slot/symbol performansını topla, state’den cash & open say, cache’e yaz.
    """
    _, trades = _read_trades()
    slot_perf, sym_perf = _perf_from_trades(trades)
    cash, open_cnt = _read_state_cash_open()

    CACHE["slot_perf"] = slot_perf
    CACHE["sym_perf"]  = sym_perf
    CACHE["open_count"] = open_cnt
    if cash is not None: CACHE["cash"] = cash
    CACHE["last_ts"] = time.time()

def get_slot_perf(slot: str) -> dict:
    return (CACHE.get("slot_perf") or {}).get(slot, {"pnl":0.0,"n":0,"wins":0,"wr":0.5,"pf":1.0})

def get_sym_perf(sym: str) -> dict:
    return (CACHE.get("sym_perf") or {}).get(sym, {"pnl":0.0,"n":0,"wins":0,"wr":0.5,"pf":1.0})

def snapshot() -> dict:
    return {
        "ts": CACHE.get("last_ts", 0.0),
        "slot_perf": CACHE.get("slot_perf", {}),
        "sym_perf":  CACHE.get("sym_perf", {}),
        "open": CACHE.get("open_count", 0),
        "cash": CACHE.get("cash", None),
    }
