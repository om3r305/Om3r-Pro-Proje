# learn/weight_tuner.py
from __future__ import annotations
from pathlib import Path
import csv, io
from typing import Dict

def read_trades_csv(path="logs/trades.csv"):
    p = Path(path)
    if not p.exists(): return []
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd: rows.append(r)
    return rows

def simple_tune() -> Dict[str,float]:
    """
    Yer tutucu: son 200 işlemde, hangi slot daha iyi? w_pred/w_dip/w_ob/w_news çıkar.
    """
    rows = read_trades_csv()
    rows = rows[-200:]
    agg = {"pred":0.0,"dip":0.0,"ob":0.0,"news":0.0}
    cnt = {"pred":0,"dip":0,"ob":0,"news":0}
    for r in rows:
        slot = (r.get("slot") or "pred").lower()
        pnl  = float(r.get("pnl_usd", r.get("pnl", 0.0)) or 0.0)
        if slot in agg:
            agg[slot]+=pnl; cnt[slot]+=1
    w={}
    total = sum(max(0.0,agg[k]) for k in agg)
    for k in agg:
        v = max(0.0, agg[k])
        w[k] = (v/total) if total>0 else 0.25
    return w
