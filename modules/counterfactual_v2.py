# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, json, time
from pathlib import Path
OUT=Path("logs/counterfactual.jsonl")
def enrich_from_trades(trades_csv="logs/trades_full_log.csv"):
    p=Path(trades_csv)
    if not p.exists(): return 0
    n=0
    with p.open("r",encoding="utf-8",errors="ignore") as f:
        rd=csv.DictReader(f)
        for r in rd:
            try:
                pnl=float(r.get("pnl") or r.get("PnL") or 0)
                slot=r.get("slot") or r.get("reason") or "?"
            except Exception:
                continue
            what_if = {"tp_x1.2": pnl*1.2, "sl_x0.8": (pnl*0.8 if pnl<0 else pnl)}
            rec={"ts":time.time(),"symbol":r.get("symbol"),"slot":slot,"pnl":pnl,"what_if":what_if}
            try:
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.open("a",encoding="utf-8").write(json.dumps(rec,ensure_ascii=False)+"\n"); n+=1
            except Exception: pass
    return n
