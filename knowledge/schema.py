# knowledge/schema.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, TypedDict

class Finding(TypedDict, total=False):
    ts: int
    source: str         # "news|onchain|paper|social"
    symbols: List[str]
    impact: float       # -1..+1
    confidence: float   # 0..1
    horizon_min: int
    summary: str
    link: Optional[str]
