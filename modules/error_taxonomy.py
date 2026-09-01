# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, json, time
from pathlib import Path
from typing import Dict, Any, List

TRADES = Path("logs/trades_full_log.csv")
OUT    = Path("logs/error_taxonomy.jsonl")

def _read() -> List[Dict[str,Any]]:
    if not TRADES.exists(): return []
    out=[]
    with TRADES.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd: out.append(r)
    return out

def _slot(r: Dict[str,Any]) -> str:
    for k in ("slot","label","reason","enter_reason"):
        v = (r.get(k) or "").lower()
        if "dip" in v: return "dip"
        if "pred" in v: return "pred"
        if "news" in v: return "news"
        if "ob"   in v: return "ob"
    return "any"

def classify_once() -> None:
    rows = _read()
    buckets: Dict[str,int] = {}
    for r in rows:
        side = (r.get("side") or r.get("event") or "").upper()
        if side not in ("SELL","CLOSE"): continue
        try: pnl = float(r.get("pnl") or 0.0)
        except Exception: pnl = 0.0
        if pnl >= 0: continue
        reg = (r.get("regime") or "UNK").upper()
        spread = float(r.get("spread_bps") or 0.0)
        s = _slot(r)
        cause = "unknown"
        if reg=="CHOP": cause = "range_noise"
        if spread>=10: cause = "wide_spread"
        # geniş basit kurallar – ileride zenginleşir
        key = f"{s}:{cause}"
        buckets[key] = buckets.get(key,0) + 1
    try:
        rec = {"ts": time.time(), "summary": buckets}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
