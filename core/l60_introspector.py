# -*- coding: utf-8 -*-
from __future__ import annotations
"""
L60 — l60_introspector
- Kaybedilen işlemleri/logları analiz eder, kök neden (root-cause) özetleri çıkarır.
- Tek başına import edildiğinde mevcut loglar yoksa no-op döner.
"""
import csv, json, statistics, time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

TRADES_FULL = Path("logs/trades_full_log.csv")
REPORT_JSON = Path("logs/l60_introspect_report.json")

def _safe_float(x, d=0.0):
    try: return float(x)
    except Exception: return d

def _read_trades() -> List[Dict[str,Any]]:
    if not TRADES_FULL.exists(): return []
    out = []
    with TRADES_FULL.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd: out.append(r)
    return out

def analyze_failures(cfg: Dict[str,Any] | None = None) -> Dict[str,Any]:
    """
    Basit kök-neden analizi:
      - Rejim bazlı PnL
      - Slot bazlı PnL
      - En sık kapanış nedeni
      - TP/SL mesafeleri (yakın/uzak)
    Dönüş: policy/suggestion + destekleyici metrikler
    """
    rows = _read_trades()
    if not rows:
        rep = {"ts": time.time(), "ok": False, "reason": "no_logs"}
        REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
        REPORT_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        return rep

    # toparla
    regime_pnl: Dict[str, float] = {}
    slot_pnl: Dict[str, float] = {}
    reasons: Dict[str, int] = {}
    closed = 0
    for r in rows:
        evt = r.get("event","")
        if evt != "close": continue
        closed += 1
        pnl = _safe_float(r.get("pnl", 0.0))
        reg = (r.get("regime") or "UNKNOWN") or "UNKNOWN"
        slot = (r.get("slot") or "pred")
        regime_pnl[reg] = regime_pnl.get(reg, 0.0) + pnl
        slot_pnl[slot] = slot_pnl.get(slot, 0.0) + pnl
        reasons[r.get("reason","unknown")] = reasons.get(r.get("reason","unknown"),0)+1

    worst_reg = None
    if regime_pnl:
        worst_reg = sorted(regime_pnl.items(), key=lambda x: x[1])[0][0]
    worst_slot = None
    if slot_pnl:
        worst_slot = sorted(slot_pnl.items(), key=lambda x: x[1])[0][0]
    top_reason = None
    if reasons:
        top_reason = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[0][0]

    # Öneriler (basit kural tabanı)
    suggestions: List[str] = []
    policy: Dict[str,Any] = {"actions": []}
    if worst_reg == "CHOP":
        suggestions.append("CHOP rejiminde dip ağırlığı azalt; threshold yükselt.")
        policy["actions"].append({"type":"tune_block_weight", "block":"regime_trend_bonus", "delta":-0.15})
        policy["actions"].append({"type":"raise_threshold", "slot":"pred", "delta":0.03})
    if worst_slot in ("pred","ob"):
        suggestions.append(f"{worst_slot} slotunda giriş eşiğini +0.02 yükselt.")
        policy["actions"].append({"type":"raise_threshold", "slot": worst_slot, "delta":0.02})
    if (top_reason or "").startswith("bearish_exit"):
        suggestions.append("Bearish çıkış tetikleniyor: momentum/accel bloklarına daha çok ağırlık ver.")
        policy["actions"].append({"type":"tune_block_weight", "block":"price_accel", "delta":+0.10})

    rep = {
        "ts": time.time(),
        "ok": True,
        "summary": {
            "regime_pnl": regime_pnl,
            "slot_pnl": slot_pnl,
            "top_close_reason": top_reason,
            "closed_count": closed
        },
        "policy": policy,
        "suggestions": suggestions
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep
