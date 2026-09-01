# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any

CF = Path("logs/counterfactual.jsonl")

def log_candidate(symbol: str, slot: str, price: float, reason: str, ctx: Dict[str,Any]) -> None:
    rec = {"ts": time.time(), "symbol": symbol, "slot": slot, "price": float(price),
           "reason": reason, "ctx": ctx}
    try:
        CF.parent.mkdir(parents=True, exist_ok=True)
        with CF.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    except Exception:
        pass
