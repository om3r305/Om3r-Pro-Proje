# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any

DEFAULTS: Dict[str, Any] = {
    "learning.autopatch.lo": 0.45,
    "learning.autopatch.hi": 0.80,
    "brain.conf_blend.driver_min": 0.35,
    "brain.conf_blend.governor_min": 0.40,
    "brain.veto_conf_min": 0.55,
    "drift.ewma_alpha": 0.20,
    "drift.thresholds.wr_warn": 0.48,
    "drift.thresholds.pf_warn": 1.05,
    "drift.thresholds.maxdd_warn": -5.0,
    "drift.min_trades": 50,
    "drift.lookback_trades": 300,
    "drift.min_tick_sec": 60,
    "drift.actions.cooldown_min": 10,
    "drift.cooldown_min_hard": 20,
}

def get(cfg: Dict[str, Any] | None, path: str, fallback: Any = None) -> Any:
    cur = cfg or {}
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return DEFAULTS.get(path, fallback)
        cur = cur[k]
    return cur

def clamp(v: float, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except Exception:
        x = lo
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, x))
