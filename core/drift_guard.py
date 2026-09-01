# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, time
from pathlib import Path
from typing import Dict, Tuple
from Proje1.core.brain_hook import brain_overrides
try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, parse_mode="HTML", **k): pass

TRADES_CSV = Path("logs/trades_full_log.csv")

def _recent_wr_pf(hours: int = 6) -> Tuple[float, float, int]:
    if not TRADES_CSV.exists(): return 0.0, 0.0, 0
    now = time.time(); wins=0; n=0; gpos=0.0; gneg=0.0
    try:
        for r in csv.DictReader(TRADES_CSV.open("r", encoding="utf-8")):
            ts = float(r.get("ts", 0.0) or 0.0)
            if ts and now - ts > hours*3600: continue
            pnl = float(r.get("pnl", 0.0)); n += 1
            if pnl > 0: wins += 1; gpos += pnl
            else:       gneg += abs(pnl)
    except Exception:
        return 0.0, 0.0, 0
    wr = (wins / n) if n else 0.0
    pf = (gpos / gneg) if gneg > 1e-9 else (gpos > 0 and 99.0 or 0.0)
    return wr, pf, n

def drift_check_and_actions(cfg: Dict) -> None:
    """WR/PF bozulduysa küçük kaçış: veto_conf_min ↑ veya portföyü defansif ayarla."""
    guard = ((cfg or {}).get("autopilot") or {}).get("targets", {})
    wr_min = float(guard.get("wr_min", 0.52))
    pf_min = float(guard.get("pf_min", 1.20))

    wr, pf, n = _recent_wr_pf(hours=6)
    if n < 6: return

    if wr < wr_min or pf < pf_min:
        # defansif küçük ayar
        cur = float(((cfg or {}).get("brain") or {}).get("veto_conf_min", 0.55))
        step = 0.01
        newv = max(0.45, min(0.80, cur + step))
        patch = {"brain.veto_conf_min": round(newv, 3)}
        # ağırlıkları biraz dengeli mod’a çek
        patch.update({
            "portfolio.dip": 0.33, "portfolio.pred": 0.27, "portfolio.news": 0.20, "portfolio.ob": 0.20
        })
        brain_overrides(patch, cfg)
        try:
            tg_send(f"🧯 Drift guard: wr={wr:.2f} pf={pf:.2f} n={n} → küçük defansif ayar (veto+portföy).",
                    parse_mode="HTML")
        except Exception: pass
