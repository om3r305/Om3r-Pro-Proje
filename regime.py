# regime.py — piyasa rejimi tespitçisi
import requests, statistics

def _kl(symbol, interval="1m", limit=60):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": limit},
                     timeout=4)
    k = r.json()
    return [float(x[4]) for x in k]  # close fiyatları

def _slope(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    x = range(n)
    mx = (n - 1) / 2
    num = sum((i - mx) * (v - sum(xs) / n) for i, v in zip(x, xs))
    den = sum((i - mx) ** 2 for i in x) + 1e-12
    return num / den

def _vol_ratio(c):
    if len(c) < 25:
        return 1.0
    s5  = statistics.pstdev(c[-5:])  or 1e-9
    s20 = statistics.pstdev(c[-20:]) or 1e-9
    return s5 / s20

def ema(seq, n):
    a = 2 / (n + 1)
    out = []
    for v in seq:
        if not out:
            out.append(v)
        else:
            out.append(out[-1] + a * (v - out[-1]))
    return out

class RegimeDetector:
    def __init__(self, symbol: str):
        self.s = symbol

    def snapshot(self):
        c1 = _kl(self.s, "1m", 60)
        c5 = _kl(self.s, "5m", 60)
        c15 = _kl(self.s, "15m", 60)
        if not c1 or not c5 or not c15:
            return {"regime": "UNKNOWN", "ema9": None}
        sl1 = _slope(c1[-30:]); sl5 = _slope(c5[-30:]); sl15 = _slope(c15[-30:])
        vr  = _vol_ratio(c1)
        e9  = ema(c1, 9)[-1]
        score_trend = (sl1 > 0) + (sl5 > 0) + (sl15 > 0) - ((sl1 < 0) + (sl5 < 0) + (sl15 < 0))
        if abs(score_trend) >= 2 and vr >= 0.8:
            regime = "TREND"
        elif vr <= 0.6:
            regime = "MEAN"
        else:
            regime = "CHOP"
        return {"regime": regime, "ema9": e9, "vol_ratio": vr, "slope": (sl1, sl5, sl15)}
