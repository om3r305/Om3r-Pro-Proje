# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from pathlib import Path
OVR=Path("runtime/runtime_overrides.jsonl")
def kill_symbol(symbol:str, minutes:int=30):
    try:
        rec={"ts":time.time(),"set":{f"cooldown.{symbol}": minutes}}
        OVR.parent.mkdir(parents=True, exist_ok=True)
        OVR.open("a",encoding="utf-8").write(json.dumps(rec)+"\n")
    except Exception: pass
