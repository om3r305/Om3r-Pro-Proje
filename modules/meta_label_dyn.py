# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, statistics
from pathlib import Path
LOG=Path("logs/brain_log.jsonl")
def suggest_veto(min_winrate:float=0.50, lo:float=0.45, hi:float=0.80)->float:
    # naive heuristic based on last 200 outcomes in trades_full_log.csv
    import csv
    p=Path("logs/trades_full_log.csv"); wins=0;tot=0
    if p.exists():
        with p.open("r",encoding="utf-8",errors="ignore") as f:
            rd=csv.DictReader(f)
            rows=list(rd)[-200:]
            for r in rows:
                try: pnl=float(r.get("pnl") or r.get("PnL") or 0.0)
                except: pnl=0.0
                tot+=1; wins+= (1 if pnl>=0 else 0)
    wr=(wins/tot) if tot else 0.0
    tgt = max(lo, min(hi, 0.55 if wr<min_winrate else 0.50))
    # log
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.open("a",encoding="utf-8").write(json.dumps({"ts":time.time(),"type":"meta_label_dyn","wr":wr,"veto":tgt})+"\n")
    except Exception: pass
    return tgt
