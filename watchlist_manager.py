# watchlist_manager.py — Adaptive Watchlist (dinamik en hareketli coinler)
import time, requests
from typing import List, Dict, Tuple

def _kl_change_pct(symbol: str, interval="1m", bars=3) -> float:
    """son barlar arası kısa vadeli değişim (%)"""
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": symbol, "interval": interval, "limit": bars},
                         timeout=4)
        k = r.json()
        if len(k) < 2: return 0.0
        c1 = float(k[-2][4]); c2 = float(k[-1][4])
        if c1 == 0: return 0.0
        return (c2 / c1 - 1.0) * 100.0
    except Exception:
        return 0.0

def _range_pct(symbol: str, interval="1m", bars=10) -> float:
    """kısa dönem range/close (%) — oynaklık proxysi"""
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": symbol, "interval": interval, "limit": bars},
                         timeout=4)
        k = r.json()
        highs = [float(x[2]) for x in k]
        lows  = [float(x[3]) for x in k]
        closes= [float(x[4]) for x in k]
        if not closes: return 0.0
        rng = (max(highs) - min(lows)) / (sum(closes)/len(closes) + 1e-12)
        return rng * 100.0
    except Exception:
        return 0.0

class WatchlistManager:
    """
    Allowed universe → filtrele (likidite, minimum fiyat)
    Her refresh'te skorla: w_qv*quoteVolume + w_range*range% + w_change*1m_change%
    En yüksek TOP_N'i aktif liste olarak döndür.
    """
    def __init__(self, cfg: Dict):
        sc = cfg.get("scanner", {})
        self.static_list: List[str] = sc.get("watchlist", [])
        ad = sc.get("adaptive", {})
        self.enabled = bool(ad.get("enabled", True))
        self.refresh_min = int(ad.get("refresh_min", 10))
        self.top_n = int(ad.get("top_n", max(len(self.static_list), 6) or 6))
        self.min_qv = float(ad.get("min_quote_volume_usdt", 10_000_000))
        self.min_price = float(ad.get("min_price", 0.02))
        self.exclude = set(ad.get("exclude", []))
        scw = ad.get("score", {"w_qv": 0.5, "w_range": 0.3, "w_change": 0.2})
        self.w_qv = float(scw.get("w_qv", 0.5))
        self.w_range = float(scw.get("w_range", 0.3))
        self.w_change = float(scw.get("w_change", 0.2))
        self._last = 0.0
        self._active: List[str] = self.static_list[:]

    def _universe_usdt(self) -> List[str]:
        try:
            r = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=6)
            j = r.json()["symbols"]
            out = []
            for s in j:
                if s.get("status") != "TRADING": continue
                if s.get("quoteAsset") != "USDT": continue
                if s.get("isSpotTradingAllowed") is False: continue
                sym = s.get("symbol")
                if sym.endswith("BUSD"): continue
                out.append(sym)
            return out
        except Exception:
            # fallback
            return self.static_list[:]

    def _score(self, sym: str) -> float:
        try:
            t = requests.get("https://api.binance.com/api/v3/ticker/24hr",
                             params={"symbol": sym}, timeout=4).json()
            qv = float(t.get("quoteVolume", 0.0))
            last = float(t.get("lastPrice", 0.0) or 0.0)
            if qv < self.min_qv or last < self.min_price: return 0.0
            r = _range_pct(sym, "1m", 10)      # %
            c = _kl_change_pct(sym, "1m", 3)   # %
            # normalize kabaca
            qvn = min(1.0, qv / (200_000_000.0))   # 200M+ USDT → 1.0
            rn  = min(1.0, r / 1.0)                # %1 range → 1.0
            cn  = max(0.0, min(1.0, (c + 1.0) / 3.0))  # +2% → ~1.0 ; -1% → 0
            return self.w_qv*qvn + self.w_range*rn + self.w_change*cn
        except Exception:
            return 0.0

    def update(self):
        if not self.enabled:
            self._active = self.static_list[:]
            return
        now = time.time()
        if now - self._last < self.refresh_min * 60:
            return
        self._last = now
        candidates = self._universe_usdt()
        candidates = [c for c in candidates if c not in self.exclude]
        scored = [(sym, self._score(sym)) for sym in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        self._active = [s for s,_ in scored[:self.top_n]] or self.static_list[:]

    def active(self) -> List[str]:
        return self._active[:]
