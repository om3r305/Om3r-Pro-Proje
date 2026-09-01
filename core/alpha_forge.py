# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, random
from pathlib import Path
CHAMP=Path("model/evo_champions.jsonl")
def evolve_step():
    cand={"tp_pct":round(random.uniform(0.04,0.10),3),"sl_pct":round(random.uniform(0.03,0.08),3)}
    score=random.uniform(0.9,1.4)
    rec={"ts":time.time(),"candidate":cand,"score":score}
    try:
        CHAMP.parent.mkdir(parents=True, exist_ok=True)
        CHAMP.open("a",encoding="utf-8").write(json.dumps(rec)+"\n")
    except Exception: pass
    return rec
