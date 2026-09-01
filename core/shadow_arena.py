# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, random
from pathlib import Path
LEADER=Path("logs/shadow_leaderboard.jsonl")
def simulate_configs(k:int=3):
    # placeholder: score random; in real impl hook into backtest/live-sim
    now=time.time()
    items=[]
    for i in range(k):
        cfg={"name":f"shadow_{i}","tp":round(random.uniform(0.03,0.12),3),"sl":round(random.uniform(0.03,0.10),3)}
        score=random.uniform(0.8,1.3)
        items.append({"ts":now,"cfg":cfg,"score":score})
    try:
        LEADER.parent.mkdir(parents=True, exist_ok=True)
        with LEADER.open("a",encoding="utf-8") as f:
            for it in items: f.write(json.dumps(it)+"\n")
    except Exception: pass
    return items
