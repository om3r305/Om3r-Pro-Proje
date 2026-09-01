# metrics_utils.py
import csv, time
from typing import Dict, Any, List
from collections import defaultdict
from pathlib import Path

def load_trades(csv_path="logs/trades.csv", lookback_sec=2*3600) -> List[dict]:
    p = Path(csv_path)
    if not p.exists(): return []
    rows=[]
    cut = time.time() - lookback_sec
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ts = int(float(row.get("ts", "0")))
                if ts < cut: continue
                rows.append({
                    "ts": ts,
                    "event": row.get("event",""),
                    "slot": row.get("slot",""),
                    "symbol": row.get("symbol",""),
                    "pnl": float(row.get("pnl","0") or 0)
                })
            except:
                continue
    return rows

def compute_metrics(rows: List[dict]) -> Dict[str, Any]:
    closes = [r for r in rows if r["event"]=="close"]
    pnl = [float(r["pnl"]) for r in closes]
    cnt = len(closes)
    wins = sum(1 for x in pnl if x > 0)
    losses = sum(1 for x in pnl if x < 0)
    total_pnl = sum(pnl)
    gains = sum(x for x in pnl if x > 0)
    loss_abs = sum(-x for x in pnl if x < 0)
    pf = (gains/loss_abs) if loss_abs>0 else (999.0 if gains>0 else 0.0)
    wr = (wins/cnt) if cnt>0 else 0.0
    cum=0.0; peak=0.0; maxdd=0.0
    for x in pnl:
        cum+=x; peak=max(peak,cum); maxdd=max(maxdd, peak-cum)

    slot_pnl=defaultdict(float)
    coin_pnl=defaultdict(float)
    for r in closes:
        slot_pnl[r["slot"]] += float(r["pnl"])
        coin_pnl[r["symbol"]] += float(r["pnl"])
    return {
        "trades": cnt, "pnl": total_pnl, "pf": pf, "wr": wr,
        "maxdd": -maxdd, "slot_pnl": dict(slot_pnl), "coin_pnl": dict(coin_pnl)
    }
