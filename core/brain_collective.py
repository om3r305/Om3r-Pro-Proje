# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Tuple

MEM_FILE = Path("logs/collective_memory.jsonl")  # dış dünya işaretleri buraya düşebilir

class CollectiveMemory:
    def __init__(self, cfg: Dict):
        c = (cfg.get("brain") or {}).get("collective", {}) or {}
        self.enabled = bool(c.get("enabled", True))
        self.lookback_sec = int(c.get("lookback_sec", 3600*6))  # son 6 saat
        self.max_boost = float(c.get("max_boost", 0.15))
        self._last_ts = 0.0
        self._cache: Dict[str, float] = {}

    def _refresh(self) -> None:
        now = time.time()
        if now - self._last_ts < 20:  # 20 sn
            return
        self._last_ts = now
        cache: Dict[str, float] = {}
        if MEM_FILE.exists():
            with MEM_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts = float(rec.get("ts", now))
                    if now - ts > self.lookback_sec:
                        continue
                    sym = str(rec.get("symbol", "")).upper()
                    score = float(rec.get("score", 0.0))  # +:pozitif, -:negatif
                    cache[sym] = cache.get(sym, 0.0) + score
        # normalize to [-1,1]
        if cache:
            mx = max(abs(v) for v in cache.values()) or 1.0
            for k in list(cache.keys()):
                cache[k] = max(-1.0, min(1.0, cache[k]/mx))
        self._cache = cache

    def boost(self, symbol: str) -> Tuple[float, str]:
        if not self.enabled:
            return 0.0, "off"
        self._refresh()
        s = self._cache.get(symbol.upper(), 0.0)
        return (s * self.max_boost, f"cmem:{s:+.2f}")
