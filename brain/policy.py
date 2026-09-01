# brain/policy.py
from __future__ import annotations
from typing import Dict

REGIME_MULT = {
    "TREND": {"pred": 1.3, "dip": 1.0, "ob": 1.0, "news": 1.1},
    "MEAN":  {"pred": 0.8, "dip": 1.2, "ob": 1.0, "news": 0.9},
    "CHOP":  {"pred": 0.7, "dip": 1.1, "ob": 1.0, "news": 1.0},
    "UNKNOWN":{"pred": 1.0, "dip": 1.0, "ob": 1.0, "news": 1.0},
}

BASE_W = {"pred":0.4, "dip":0.3, "ob":0.2, "news":0.1}

def alpha_mix(alphas: Dict[str,float], regime: str = "UNKNOWN") -> float:
    """
    alphas: {"pred":score, "dip":score, "ob":score, "news":score}  # 0..1
    return: combined score (0..1)
    """
    reg = REGIME_MULT.get((regime or "UNKNOWN").upper(), REGIME_MULT["UNKNOWN"])
    s = 0.0; wsum=0.0
    for k, base_w in BASE_W.items():
        a = float(alphas.get(k, 0.0))
        w = base_w * reg.get(k, 1.0)
        s += a * w
        wsum += w
    return max(0.0, min(1.0, s / (wsum or 1.0)))
