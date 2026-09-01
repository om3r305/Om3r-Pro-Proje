# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from pathlib import Path
EV=Path("logs/events.jsonl"); OVR=Path("runtime/runtime_overrides.jsonl")
def _w(kind, **kw):
    try:
        EV.parent.mkdir(parents=True, exist_ok=True)
        EV.open("a",encoding="utf-8").write(json.dumps({"ts":time.time(),"type":kind,"payload":kw})+"\n")
    except Exception: pass
def soft_hard_check(wr:float, pf:float, wr_soft=0.49, wr_hard=0.46, pf_soft=1.08, pf_hard=1.02):
    level="ok"
    if wr<=wr_hard or pf<=pf_hard: level="hard"
    elif wr<=wr_soft or pf<=pf_soft: level="soft"
    return level
def bump_veto(level:str):
    bump=0.0
    if level=="soft": bump=0.02
    if level=="hard": bump=0.04
    if bump>0:
        try:
            rec={"ts":time.time(),"set":{"brain.veto_conf_min":"+%.2f"%bump}}
            OVR.parent.mkdir(parents=True, exist_ok=True); OVR.open("a",encoding="utf-8").write(json.dumps(rec)+"\n")
            _w("drift.bump_veto", level=level, bump=bump)
        except Exception: pass
