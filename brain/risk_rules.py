# brain/risk_rules.py
from __future__ import annotations
from typing import Dict, Any

def can_open_by_risk(cfg: Dict[str,Any], symbol: str, slot: str, proposal: Dict[str,Any], market: Dict[str,Any]):
    """
    Çok temel: spread/funding/vol/qty sanity gibi.
    Geliştirilebilir. Şimdilik iki hızlı veto:
      - spread_bps > 25 -> reddet
      - qty <= 0 -> reddet
    """
    spread = float(market.get("spread_bps", 0.0))
    if spread > 25.0:
        return False, f"spread_bps>{spread:.1f}"
    qty = float(proposal.get("qty") or 0.0)
    if qty <= 0:
        return False, "qty<=0"
    return True, "ok"
