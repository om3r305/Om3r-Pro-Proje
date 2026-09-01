# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Optional
import time
import requests

BINANCE_BASE = "https://api.binance.com"

# Binance destekli spot kline interval listesi
_VALID_INTERVALS = {
    "1s","1m","3m","5m","15m","30m",
    "1h","2h","4h","6h","8h","12h",
    "1d","3d","1w","1M"
}

# ---------- helpers ----------

def _norm_symbol(symbol: str) -> str:
    """ETH/USDT -> ETHUSDT ; already ETHUSDT -> ETHUSDT"""
    s = symbol.strip().upper()
    if "/" in s:
        base, quote = s.split("/", 1)
        s = f"{base}{quote}"
    return s

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _now_ms() -> int:
    return int(time.time() * 1000)

# ---------- public API ----------

def get_candles(
    symbol: str,
    tf: str = "1m",
    lookback: int = 50,
    *,
    include_partial: bool = False,
    **kwargs
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Mum verisi döndürür (oldest -> newest sırada).
    Dönüş: [(ts, open, high, low, close, volume), ...]
    Hata olursa boş liste döner.

    Parametre alias'ları (geri uyum):
      - timeframe / interval  -> tf
      - limit                 -> lookback
    """
    # ---- alias uyarlama ----
    if "timeframe" in kwargs and kwargs["timeframe"]:
        tf = kwargs["timeframe"]
    if "interval" in kwargs and kwargs["interval"]:
        tf = kwargs["interval"]
    if "limit" in kwargs and kwargs["limit"]:
        lookback = kwargs["limit"]

    interval = tf if tf in _VALID_INTERVALS else "1m"
    # Binance /klines max 1000
    limit = max(1, min(int(lookback), 1000))
    sym = _norm_symbol(symbol)

    try:
        url = f"{BINANCE_BASE}/api/v3/klines"
        params = {"symbol": sym, "interval": interval, "limit": limit + 2}  # +2: dedup/partial payı
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        rows = r.json()  # [ [openTime, open, high, low, close, volume, closeTime, ...], ... ]
        if not isinstance(rows, list):
            return []

        # map -> tuples
        kl = []
        for row in rows:
            if not (isinstance(row, list) and len(row) >= 7):
                continue
            open_time = int(row[0])
            close_time = int(row[6])
            o = _safe_float(row[1])
            h = _safe_float(row[2])
            l = _safe_float(row[3])
            c = _safe_float(row[4])
            v = _safe_float(row[5])

            # partial (kapanmamış) son mum: now < close_time ise ve include_partial=False -> at
            if not include_partial and (_now_ms() < close_time):
                # bu satırı eklemiyoruz (kapanmamış mum)
                continue

            kl.append((open_time, o, h, l, c, v))

        if not kl:
            return []

        # oldest -> newest sırala
        kl.sort(key=lambda x: x[0])

        # duplicate ts temizliği (son değeri koru)
        uniq = {}
        for t in kl:
            uniq[t[0]] = t
        kl = [uniq[k] for k in sorted(uniq.keys())]

        # son limit adedi
        if len(kl) > limit:
            kl = kl[-limit:]

        return kl

    except requests.RequestException:
        return []
    except Exception:
        return []


def get_price(symbol: str) -> Optional[float]:
    """
    Son fiyatı döndürür. Önce /ticker/price, sonra /ticker/bookTicker fallback.
    Hata olursa None döner.
    """
    sym = _norm_symbol(symbol)
    try:
        # primary: /ticker/price
        r = requests.get(f"{BINANCE_BASE}/api/v3/ticker/price", params={"symbol": sym}, timeout=6)
        if r.ok:
            js = r.json()
            px = js.get("price")
            if px is not None:
                return float(px)

        # fallback: /ticker/bookTicker (bid/ask ortalaması)
        rb = requests.get(f"{BINANCE_BASE}/api/v3/ticker/bookTicker", params={"symbol": sym}, timeout=6)
        if rb.ok:
            jb = rb.json()
            bid = jb.get("bidPrice"); ask = jb.get("askPrice")
            if bid and ask:
                return (float(bid) + float(ask)) / 2.0

        return None
    except Exception:
        return None
