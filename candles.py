# candles.py
# ------------------------------------------------------------
# Esnek mum (candle) yardımcıları + temel bias ölçümü
# - Birçok veri kaynağı formatını OHLC'ye dönüştürür
# - Geçersiz kayıtları atlar, hiç veri kalmazsa açıklayıcı hata verir
# - Basit trend/bias & EMA bilgisi üretir
#
# Dış API:
#   normalize_candles(raw, max_count=None) -> List[OHLC]
#   candle_bias(candles, cfg: dict, extras: dict | None = None)
#       -> (bias: float, conf: dict, tag: dict)
#
# Fast trader buradaki candle_bias'i üçlü dönüş ile çağırır:
#   cbias, cconf, ctag = candle_bias(series, cfg.get("candles", {}))
# ------------------------------------------------------------

from __future__ import annotations

from math import isnan
from decimal import Decimal
from typing import Iterable, Any, List, Tuple, Dict, Optional

OHLC = Tuple[float, float, float, float]  # (open, high, low, close)


# --------------------------- düşük seviyeli yardımcılar ----------------------

def _to_float(x: Any) -> float:
    if x is None:
        raise ValueError("None")
    if isinstance(x, (int, float)):
        v = float(x)
    elif isinstance(x, Decimal):
        v = float(x)
    elif isinstance(x, str):
        s = x.strip().replace(",", ".")
        if s == "":
            raise ValueError("empty-str")
        v = float(s)
    else:
        raise ValueError(f"bad-type:{type(x).__name__}")
    if isnan(v):
        raise ValueError("NaN")
    return v


def _keys_lc(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in d.items()}


def _to_ohlc(c: Any) -> OHLC:
    """
    Çok formatlı tekil candle -> OHLC çevirici.

    Desteklenenler:
      - [ts, open, high, low, close, *rest]
      - [open, high, low, close]
      - {"o":..,"h":..,"l":..,"c":..} / {"open":..,"high":..,"low":..,"close":..}
      - "open,high,low,close"
    """
    # list/tuple
    if isinstance(c, (list, tuple)):
        if len(c) >= 5:
            o, h, l, cl = c[1], c[2], c[3], c[4]
        elif len(c) >= 4:
            o, h, l, cl = c[0], c[1], c[2], c[3]
        else:
            raise TypeError(f"Unsupported candle len={len(c)}")
        return (_to_float(o), _to_float(h), _to_float(l), _to_float(cl))

    # dict
    if isinstance(c, dict):
        d = _keys_lc(c)

        def pick(*names):
            for n in names:
                if n in d:
                    return d[n]
            raise KeyError("/".join(names))

        o = pick("o", "open", "op")
        h = pick("h", "high", "hi")
        l = pick("l", "low", "lo")
        cl = pick("c", "close", "closing", "cl")
        return (_to_float(o), _to_float(h), _to_float(l), _to_float(cl))

    # "o,h,l,c"
    if isinstance(c, str):
        parts = [p.strip() for p in c.replace(";", ",").split(",")]
        if len(parts) >= 4:
            o, h, l, cl = parts[0], parts[1], parts[2], parts[3]
            return (_to_float(o), _to_float(h), _to_float(l), _to_float(cl))
        raise TypeError("Unsupported candle string")

    raise TypeError(f"Unsupported candle format: {type(c).__name__}")


# ----------------------------- halka açık fonksiyonlar ----------------------

def normalize_candles(raw: Iterable[Any], max_count: Optional[int] = None) -> List[OHLC]:
    data = list(raw)
    if max_count:
        data = data[-max_count:]

    out: List[OHLC] = []
    skipped = 0
    for i, c in enumerate(data):
        try:
            out.append(_to_ohlc(c))
        except Exception as e:
            skipped += 1
            if i < 3:  # sadece ilk 3 hatalı örneği bas
                print(f"[DEBUG] Skipped candle[{i}] -> {c!r}, reason: {e}")
            continue

    if not out:
        raise TypeError(f"Too few valid candles (valid=0, skipped={skipped})")
    return out


# ------------------------------ mini indikatörler ---------------------------

def _ema(values: List[float], n: int) -> List[float]:
    if n <= 1:
        return values[:]
    if not values:
        return []
    k = 2.0 / (n + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1.0 - k))
    return out


def _atr(ohlc: List[OHLC], n: int) -> List[float]:
    if not ohlc:
        return []
    trs: List[float] = []
    prev_close = ohlc[0][3]
    for (_, h, l, c) in ohlc:
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    return _ema(trs, n)


def _last_cross(a: List[float], b: List[float]) -> Optional[str]:
    """
    a ile b arasında son kesişim yönü:
      return 'up' / 'down' / None
    """
    if len(a) < 2 or len(b) < 2:
        return None
    for i in range(len(a) - 1, 0, -1):
        da = a[i] - b[i]
        db = a[i - 1] - b[i - 1]
        if da == 0:
            continue
        if da > 0 and db < 0:
            return "up"
        if da < 0 and db > 0:
            return "down"
    return None


# ------------------------------ ana bias fonksiyonu -------------------------

def candle_bias(
    candles: Iterable[Any],
    cfg: Dict[str, Any],
    extras: Optional[Dict[str, Any]] = None,
):
    """
    Basit trend/bias ölçümü + açıklayıcı konfig.

    Returns:
        bias       : float  (yaklaşık -1 .. +1 arası)
        conf       : dict   (ema/atr/vol gibi yardımcı bilgiler)
        tag        : dict   (trend etiketi ve kısa açıklama)
    """
    extras = extras or {}

    bias_len: int = int(cfg.get("bias_len", 20))
    ema_len: int = int(cfg.get("ema_len", 50))
    lookback: int = int(cfg.get("lookback", 200))
    atr_len: int = int(cfg.get("atr_len", 14))

    need = max(ema_len, lookback, bias_len, atr_len) + 5
    ohlc: List[OHLC] = normalize_candles(candles, max_count=need)

    # close serisi
    closes = [c[3] for c in ohlc]
    opens = [c[0] for c in ohlc]

    # momentum benzeri bias (son bias_len mumun ortalama gövdesi)
    bodies = [c - o for (o, _, _, c) in ohlc][-bias_len:]
    rng = [max(h - l, 1e-12) for (_, h, l, _) in ohlc][-bias_len:]
    body_ratio = [b / r for b, r in zip(bodies, rng)]
    if body_ratio:
        raw_bias = sum(body_ratio) / len(body_ratio)
    else:
        raw_bias = 0.0

    # EMA ve kesişim bilgisi
    ema_fast_len = max(5, ema_len // 2)
    ema_fast = _ema(closes, ema_fast_len)
    ema_slow = _ema(closes, ema_len)
    above = closes[-1] >= ema_slow[-1] if ema_slow else False
    cross = _last_cross(ema_fast, ema_slow)

    # ATR & basit volatilite
    atr = _atr(ohlc, atr_len)
    vol = atr[-1] if atr else 0.0

    # Son değerlerin güvenilirliği için küçük sıkıştırma
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    bias = clamp(raw_bias, -1.0, 1.0)
    trend = "UP" if bias >= 0 else "DOWN"

    conf: Dict[str, Any] = {
        "ema": {
            "len": ema_len,
            "fast_len": ema_fast_len,
            "above": above,
            "last_cross": cross,
        },
        "atr": {"len": atr_len, "value": vol},
        "samples": len(ohlc),
        "params": {
            "bias_len": bias_len,
            "lookback": lookback,
        },
    }

    note_parts = []
    if cross == "up":
        note_parts.append("bullish cross")
    elif cross == "down":
        note_parts.append("bearish cross")
    if above:
        note_parts.append("price ≥ EMA")
    else:
        note_parts.append("price < EMA")

    tag = {
        "trend": trend,
        "note": ", ".join(note_parts),
    }

    return bias, conf, tag
