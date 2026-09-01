# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, json
from pathlib import Path
OUT=Path("logs/telemetry_kpi.txt")
def _wr_pf(trades:Path):
    wins=0; tot=0; g=0.0; l=0.0
    if not trades.exists(): return 0.0,0.0
    with trades.open("r",encoding="utf-8",errors="ignore") as f:
        rd=csv.DictReader(f)
        for r in rd:
            tot+=1
            try:p=float(r.get("pnl") or r.get("PnL") or 0)
            except: p=0.0
            if p>=0:g+=p;wins+=1
            else:l+=-p
    wr=(wins/tot) if tot else 0.0; pf=(g/l) if l>0 else (99.0 if g>0 else 0.0)
    return wr,pf
def write_snapshot(trades_csv="logs/trades_full_log.csv", state_path="runtime/state.json"):
    wr,pf=_wr_pf(Path(trades_csv))
    cash=maxdd="-"
    try:
        st=json.loads(Path(state_path).read_text(encoding="utf-8")); cash=st.get("cash","-"); maxdd=st.get("maxdd","-")
    except Exception: pass
    line=f"WR:{wr:.2%} | PF:{pf:.2f} | Cash:{cash} | MaxDD:{maxdd}"
    try:
        OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(line,encoding="utf-8")
    except Exception: pass
    return line
