# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Any, Dict
import math
import statistics as stats

try:
    # Proje içi import
    from Proje1.core.market import get_candles   # -> List[List[ts, o, h, l, c, v]]
except Exception:
    # En azından import patlamasın; kullanıcı kendi market.get_candles'ını sağlar.
    def get_candles(symbol: str, interval: str = "15m", limit: int = 120):
        return []

OHLC = Tuple[int, float, float, float, float, float]


# ----------------- yardımcılar -----------------
def _safelog(msg: str) -> None:
    # İstersen kapat: pass
    print(msg)


def normalize_candles(raw: Iterable[Any], max_count: Optional[int] = None) -> List[OHLC]:
    """
    Çeşitli formatlarda gelebilecek mumları güvenli şekilde normalize eder.
    Kabul: [ts, o, h, l, c, v] (en az 6 eleman)
    Hatalı item'lar atlanır. Hiç geçerli yoksa [] döner (hata fırlatmaz).
    """
    out: List[OHLC] = []
    skipped = 0
    idx = -1

    for c in raw or []:
        idx += 1
        try:
            if isinstance(c, (list, tuple)) and len(c) >= 6:
                ts, o, h, l, cl, v = c[:6]
                ts = int(ts)
                o = float(o); h = float(h); l = float(l); cl = float(cl); v = float(v)
                # basit tutarlılık kontrolü
                if not (l <= o <= h and l <= cl <= h and (h - l) >= 0.0):
                    skipped += 1
                    _safelog(f"[DEBUG] Skipped candle[{idx}] = malformed range")
                    continue
                out.append((ts, o, h, l, cl, v))
            else:
                skipped += 1
                _safelog(f"[DEBUG] Skipped candle[{idx}] = unsupported type")
        except Exception:
            skipped += 1
            _safelog(f"[DEBUG] Skipped candle[{idx}] = parse error")

    if not out:
        _safelog(f"[WARN] normalize_candles: no valid candles (skipped={skipped})")
        return []

    if max_count:
        out = out[-int(max_count):]
    return out


def _ema(values: List[float], length: int) -> List[float]:
    if length <= 1 or len(values) == 0:
        return values[:]
    k = 2.0 / (length + 1.0)
    ema_vals = [values[0]]
    for x in values[1:]:
        ema_vals.append(ema_vals[-1] + k * (x - ema_vals[-1]))
    return ema_vals


def _atr(ohlc: List[OHLC], length: int) -> float:
    if len(ohlc) < 2:
        return 0.0
    trs = []
    for i in range(1, len(ohlc)):
        _, _, high, low, close, _ = ohlc[i]
        _, _, prev_high, prev_low, prev_close, _ = ohlc[i - 1]
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        trs.append(tr)
    if len(trs) < 1:
        return 0.0
    if length > len(trs):
        length = len(trs)
    return stats.mean(trs[-length:])


# ----------------- ana bias hesap -----------------
def candle_bias(symbol: str, cfg: Dict[str, Any]) -> Tuple[str, float, str]:
    """
    Bias hesaplar: ('bullish' | 'bearish' | 'neutral'), confidence [0..1], açıklayıcı tag.
    cfg:
      - interval: "15m"
      - lookback: 120
      - fast_len: 7
      - len: 14        (slow)
      - ema_len: 9     (çıkış için opsiyonel)
    """
    interval = str(cfg.get("interval", "15m"))
    lookback = int(cfg.get("lookback", 120))
    slow_len = int(cfg.get("len", 14))
    fast_len = int(cfg.get("fast_len", 7))

    raw = get_candles(symbol, interval=interval, limit=max(lookback, slow_len + 5))
    ohlc = normalize_candles(raw, max_count=max(lookback, slow_len + 5))

    if len(ohlc) < max(10, slow_len + 2):
        # veri yetersiz → neutral
        return "neutral", 0.0, "no_data"

    closes = [c[4] for c in ohlc]

    ema_fast = _ema(closes, fast_len)
    ema_slow = _ema(closes, slow_len)

    # En son mumları kullan
    last_c = closes[-1]
    last_fast = ema_fast[-1]
    last_slow = ema_slow[-1]

    # çapraz kontrolü
    bias = "neutral"
    if last_fast > last_slow and last_c > last_fast:
        bias = "bullish"
    elif last_fast < last_slow and last_c < last_fast:
        bias = "bearish"

    # confidence: EMA ayrışması & volatilite ile normalize
    sep = abs(last_fast - last_slow)
    atr_val = _atr(ohlc, length=slow_len)
    conf = 0.0
    if atr_val > 0:
        conf = max(0.0, min(1.0, (sep / atr_val) * 0.8))
    # Son kapanışın EMA'ya mesafesinden ufak katkı
    if last_fast > 0:
        conf += max(0.0, min(0.2, abs(last_c - last_fast) / (atr_val + 1e-12) * 0.1))
    conf = max(0.0, min(1.0, conf))

    tag_parts = [f"fast_len={fast_len}", f"len={slow_len}"]
    tag = ", ".join(tag_parts)

    return bias, conf, tag


# --------------- karar yardımcıları ---------------
def allow_long(cfg: Dict[str, Any], symbol: str) -> Tuple[bool, Tuple[str, float, str], float]:
    """
    Long’a izin verilip verilmeyeceğini belirler.
    Dönen:
      allow (bool),
      (bias, conf, tag),
      bonus_conf (bullish ise ek güven katkısı)
    """
    strict = bool(cfg.get("strict", True))
    bias, conf, tag = candle_bias(symbol, cfg)

    allow = (not strict) or (bias == "bullish")
    bonus_conf = (min(1.0, 0.1 + 0.8 * conf) if bias == "bullish" else 0.0)

    return allow, (bias, conf, tag), bonus_conf


import json

def should_bearish_exit(cfg, symbol: str, ema9: float, cbias: str, cconf: float, px: float) -> bool:
    """
    Ayı (bearish) çıkışı kontrol et.
    """
    need_conf = cfg.get("candles", {}).get("exit_conf", 0.75)
    if cbias != "bearish" or cconf < need_conf:
        return False
    if ema9 and px < ema9:
        return True
    return False


    cbias, cconf, ctag = candle_bias(symbol, cfg)
    need_conf = float(cfg.get("exit_conf", 0.75))

    # EMA9 varsa ve fiyat EMA9’un altında ve bias güçlü “bearish” ise çıkar
    if cbias == "bearish" and (cconf >= need_conf) and (ema9 is not None) and (px < ema9):
        return True
    return False
