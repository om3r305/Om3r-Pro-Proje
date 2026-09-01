# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any

try:
    from Proje1.core.risk import can_trade_now as _risk_can_trade_now  # (cfg, st=None, dip=None, **kw)
except Exception:
    _risk_can_trade_now = None

try:
    from Proje1.core.alloc import can_trade_now as _alloc_can_trade_now  # (min_gap, max_tph, _cool_until, st_last_trade_ts=0.0, ...)
except Exception:
    _alloc_can_trade_now = None

def can_trade_now(symbol=None, st=None, market=None) -> bool:
    """
    Ortak guard. symbol/st/market versek de vermesek de True/False döner.
    İçeride risk ve tahsis kontrollerini sırayla dener.
    """
    cfg = getattr(st, "cfg", None) or getattr(market, "cfg", None) or {}

    # Risk katmanı (esnek imza)
    if _risk_can_trade_now is not None:
        try:
            if not _risk_can_trade_now(cfg, st=st, dip=getattr(st, "slot", "pred")):
                return False
        except Exception:
            pass

    # Tahsis/frekans katmanı — burada genelde bot kendi paramlarını çağırır.
    # Eğer uygulamada bu çağrı bot içinde yapılıyorsa, burası True kalabilir.
    if _alloc_can_trade_now is not None:
        try:
            # Varsayılanları gevşek bırakıyoruz; caller spesifik verebilir.
            return bool(_alloc_can_trade_now(
                min_gap=20, max_tph=60,
                _cool_until=getattr(st, "risk_cool_until", 0.0),
                st_last_trade_ts=getattr(st, "last_trade_ts", 0.0),
                symbol=symbol, slot=getattr(st, "slot", "pred"),
                trade_hist=getattr(st, "trade_hist", []) or []
            ))
        except Exception:
            pass

    return True
