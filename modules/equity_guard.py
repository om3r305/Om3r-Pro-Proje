# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from pathlib import Path
EV=Path("logs/events.jsonl"); ST=Path("runtime/state.json"); OVR=Path("runtime/runtime_overrides.jsonl")
def _w_event(kind, **kw):
    try:
        EV.parent.mkdir(parents=True, exist_ok=True)
        EV.open("a",encoding="utf-8").write(json.dumps({"ts":time.time(),"type":kind,"payload":kw})+"\n")
    except Exception: pass
def _load_state():
    try:
        if ST.exists(): return json.loads(ST.read_text(encoding="utf-8"))
    except Exception: pass
    return {}
def guard(max_hourly_dd:float=-3.0, entry_scale_soft:float=0.85):
    st=_load_state(); dd=float(st.get("hourly_dd",0.0) or 0.0)
    if dd<=max_hourly_dd:
        try:
            rec={"ts":time.time(),"set":{"drift.entry_scale":entry_scale_soft}}
            OVR.parent.mkdir(parents=True, exist_ok=True); OVR.open("a",encoding="utf-8").write(json.dumps(rec)+"\n")
            _w_event("equity_guard.trigger", dd=dd, entry_scale=entry_scale_soft)
        except Exception: pass
