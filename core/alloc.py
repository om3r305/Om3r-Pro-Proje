# -*- coding: utf-8 -*-
from __future__ import annotations
# core/alloc.py — slot bütçe, boyutlama ve frekans guard
import time
from typing import Dict, Tuple, List, Any

def _extract_spent(v: Any) -> float:
    """Pozisyon değeri float ya da dict olabilir. dict ise 'spent|cash|usd' içinden çeker."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in ("spent", "cash", "usd", "amount"):
            if k in v:
                try:
                    return float(v.get(k) or 0.0)
                except Exception:
                    return 0.0
    return 0.0

def slot_cash(syms: Dict[str, Any], cash: float, alloc: Dict[str, float], slot: str) -> Tuple[float, float]:
    """
    Belirli slot için ayrılmış toplam bütçe (total_slot) ve kullanılabilir kısmı (free) döner.
    - alloc: {"dip":0.40, "pred":0.30, ...}
    - syms:  { "BTUSDT": SymbolEngine, ... }  -> st.pos[slot] float veya dict olabilir.
    """
    frac = 0.0
    if isinstance(alloc, dict):
        try:
            frac = float(alloc.get(slot, 0.0) or 0.0)
        except Exception:
            frac = 0.0
    total_slot = max(0.0, float(cash) * frac)

    used = 0.0
    for st in syms.values():
        pos = getattr(st, "pos", None)
        if isinstance(pos, dict):
            used += _extract_spent(pos.get(slot))

    free = max(0.0, total_slot - used)
    return total_slot, free

def size_with_conf(entry_frac: Dict[str, float],
                   sizing: Dict[str, float],
                   slot: str,
                   free: float,
                   conf: float) -> float:
    """
    Giriş boyutu: free * entry_frac[slot] * multiplier(conf)
    multiplier(conf) = min_mult + (max_mult - min_mult) * clamp(conf,0..1)
    """
    try:
        entry = float(entry_frac.get(slot, 0.3) or 0.0)
    except Exception:
        entry = 0.3

    try:
        min_mult = float(sizing.get("min_mult", 0.5))
        max_mult = float(sizing.get("max_mult", 2.0))
    except Exception:
        min_mult, max_mult = 0.5, 2.0

    c = max(0.0, min(1.0, float(conf or 0.0)))
    mult = min_mult + (max_mult - min_mult) * c

    spend = max(0.0, float(free) * entry * mult)
    return spend

def can_trade_now(min_gap: int,
                  max_tph: int,
                  _cool_until: float,
                  st_last_trade_ts: float = 0.0,
                  symbol: str = None,
                  slot: str = None,
                  trade_hist: List[Tuple[float, str]] = None) -> bool:
    """
    Frekans/kural kontrolü (geriye dönük uyumlu).
    - min_gap: aynı sembol için iki trade arası minimum saniye
    - max_tph: saatlik toplam işlem limiti (tüm semboller)
    - _cool_until: risk cooldown sonu (epoch saniye)
    - st_last_trade_ts: sembol özel son trade zamanı (opsiyonel)
    - trade_hist: [(ts, sym)]  — opsiyonel
    """
    now = time.time()
    trade_hist = trade_hist or []

    try:
        _cool_until = float(_cool_until or 0.0)
    except Exception:
        _cool_until = 0.0

    try:
        last_ts = float(st_last_trade_ts or 0.0)
    except Exception:
        last_ts = 0.0

    try:
        min_gap = int(min_gap)
    except Exception:
        min_gap = 0

    try:
        max_tph = int(max_tph)
    except Exception:
        max_tph = 10**9  # pratikte sınırsız

    # Global cooldown
    if _cool_until and now < _cool_until:
        return False

    # Sembol özel aralık
    if last_ts and (now - last_ts) < min_gap:
        return False

    # Saatlik hacim limiti
    one_hour_ago = now - 3600.0
    recent = [t for (t, s) in trade_hist if t >= one_hour_ago]
    if len(recent) >= max_tph:
        return False

    return True



