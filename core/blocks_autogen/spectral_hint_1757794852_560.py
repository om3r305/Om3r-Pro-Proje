# AUTOGEN BLOCK (L60+) — spectral_hint_1757794852_560
# created_ts: 1757794852.634787
# NOTE: return dict keys may include: mid, up, dn, hi, lo, r, k, o, atr, value

from typing import Dict, Any
def _ema(st, n):
    return getattr(st, "ema", lambda k: st.last_px)(int(n))
def _vol(st):
    return max(0.001, float(getattr(st, "vol_norm", 0.2)))
def run(st, params: Dict[str,Any]) -> Dict[str,Any]:
    p = int(params.get("p", 7))
    v = _vol(st)
    e1 = _ema(st, p)
    e2 = _ema(st, max(2, p//2))
    val = (e2 - e1) * v
    return {"value": val}
