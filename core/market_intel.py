# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
OUT = LOGS / "market_intel.jsonl"

def maybe_note(symbol: str, ext_scores: Dict[str,Any], price: float) -> None:
    """
    Önemli dış sinyal anlarını kayda al.
    Koşul: news_shock >= 0.9 veya macro_risk > 1.2
    """
    try:
        ns = float(ext_scores.get("news_shock", 0.0))
        mr = float(ext_scores.get("macro_risk", 1.0))
        flow = ext_scores.get("flow", "neutral")
        if ns >= 0.9 or mr > 1.2:
            rec = {
                "ts": time.time(),
                "symbol": symbol,
                "price": float(price),
                "news_shock": ns,
                "macro_risk": mr,
                "flow": flow,
            }
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
