# research/sources.py
from __future__ import annotations
from typing import List, Dict, Any
import time, random

# Şimdilik uyduruk kaynak; yarın gerçek API'lere bağlanırız.
def fetch_mock_news() -> List[Dict[str,Any]]:
    base = [
        ("BTCUSDT","ETF inflows rise, positive outlook"),
        ("ETHUSDT","L2 activity spikes, gas down"),
        ("SOLUSDT","Validator set expands"),
    ]
    out=[]
    for sym, txt in base:
        out.append({
            "ts": int(time.time()),
            "source": "news",
            "symbols": [sym],
            "impact": random.uniform(-0.2,0.6),
            "confidence": random.uniform(0.3,0.8),
            "horizon_min": 60,
            "summary": txt,
            "link": None
        })
    return out
