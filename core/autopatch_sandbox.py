# -*- coding: utf-8 -*-
from __future__ import annotations
import time, json, random
from pathlib import Path
from typing import Dict, Any

try:
    from Proje1.core.guardrails import get as cfg_get, clamp
except Exception:
    def cfg_get(d, path, default=None):
        cur = d or {}
        for k in path.split('.'):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur
    def clamp(v, lo, hi):
        try: v = float(v)
        except Exception: v = lo
        return max(lo, min(hi, v))

try:
    from Proje1.core.brain_hook import brain_overrides
except Exception:
    def brain_overrides(*a, **k): pass

JOURNAL = Path("logs/autopatch_sandbox.jsonl")
JOURNAL.parent.mkdir(parents=True, exist_ok=True)

def _append(kind: str, payload: dict) -> None:
    try:
        with JOURNAL.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "kind": kind, **payload}, ensure_ascii=False) + "\n")
    except Exception:
        pass

def sandbox_try(cfg: Dict[str,Any]) -> None:
    if not bool(cfg_get(cfg, "learning.autopatch.enabled", True)):
        return

    lo = float(cfg_get(cfg, "learning.autopatch.lo", 0.45))
    hi = float(cfg_get(cfg, "learning.autopatch.hi", 0.80))
    cur = float(cfg_get(cfg, "brain.veto_conf_min", 0.55))
    jitter = (random.random()-0.5) * 0.01  # +-0.5pp
    new_veto = clamp(cur + jitter, lo, hi)

    if abs(new_veto - cur) >= 0.002:
        brain_overrides({"brain.veto_conf_min": round(new_veto, 3)}, cfg)
        _append("veto_jitter", {"old": cur, "new": new_veto})

__all__ = ["sandbox_try"]
