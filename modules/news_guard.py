# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Tuple

# opsiyonel dış skor
try:
    from Proje1.core.external_data import get_scores
except Exception:
    def get_scores(*a, **k): return {"news_shock":0.0, "macro_risk":1.0, "flow":"neutral"}

def news_block(symbol: str, cfg: Dict[str,Any] | None) -> Tuple[bool, Dict[str,float]]:
    mc = (cfg or {}).get("modules", {}).get("news_guard", {})
    shock_cut = float(mc.get("shock_cut", 0.80))
    macro_cut = float(mc.get("macro_cut", 1.35))
    try:
        s = get_scores(symbol, cfg) or {}
        shock = float(s.get("news_shock", 0.0))
        macro = float(s.get("macro_risk", 1.0))
        veto = (shock >= shock_cut) or (macro >= macro_cut)
        return veto, {"shock":shock, "macro":macro}
    except Exception:
        return False, {"shock":0.0, "macro":1.0}
