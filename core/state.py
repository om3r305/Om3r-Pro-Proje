# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from pathlib import Path

STATE_PATH = Path("runtime_state.json")

def save_state(cash: float, syms: dict):
    try:
        data = {
            "ts": time.time(),
            "cash": float(cash),
            "positions": {
                s: {k: v for k, v in st.pos.items() if v is not None}
                for s, st in syms.items()
            }
        }
        STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def load_state(default_cash: float) -> tuple[float, dict]:
    if not STATE_PATH.exists(): return default_cash, {}
    try:
        j = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return float(j.get("cash", default_cash)), j.get("positions", {})
    except Exception:
        return default_cash, {}
