# -*- coding: utf-8 -*-
from __future__ import annotations
import json, math, time, csv
from pathlib import Path
from typing import Dict, Any, Tuple, List

STATE = Path("model/bandit_state.json")
TRADES = Path("logs/trades_full_log.csv")

POLICIES = [
    {"name":"tight_fast",   "tp_mult":0.95, "sl_mult":1.02, "time_sl_min": 8},
    {"name":"balanced",     "tp_mult":1.05, "sl_mult":0.98, "time_sl_min":15},
    {"name":"wide_trend",   "tp_mult":1.12, "sl_mult":0.95, "time_sl_min":25},
    {"name":"trail_break",  "tp_mult":1.08, "sl_mult":0.97, "time_sl_min":18, "trail":True},
]

def _load() -> Dict[str,Any]:
    if STATE.exists():
        try: return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception: pass
    return {"arms": {p["name"]: {"n":0,"r":0.0} for p in POLICIES}, "last_update":0.0}

def _save(d: Dict[str,Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def _ucb(mean: float, n: int, total: int, c: float=1.1) -> float:
    if n == 0: return 1e9
    return mean + c * math.sqrt(max(0.0, math.log(max(1,total))/n))

def choose(cfg: Dict[str,Any] | None, slot: str, regime: str) -> Dict[str,Any]:
    """
    Basit UCB1: yakın geçmişte iyi getireni seç.
    Slot/Rejim özel arm'lar ileride desteklenebilir; şimdi global.
    """
    st = _load()
    arms = st["arms"]
    total = sum(a["n"] for a in arms.values()) or 1
    best = None; score = -1.0
    for p in POLICIES:
        a = arms[p["name"]]
        s = _ucb((a["r"]/max(1,a["n"])), a["n"], total)
        if s > score:
            score = s; best = p
    return dict(best)

def reward(policy_name: str, pnl: float) -> None:
    st = _load()
    a = st["arms"].setdefault(policy_name, {"n":0,"r":0.0})
    a["n"] += 1
    a["r"] += float(max(-5.0, min(5.0, pnl)))  # aşırı uçları kırp
    st["last_update"] = time.time()
    _save(st)
