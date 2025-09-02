# news_hunter.py — Haber yakalayıcı
# - Binance duyuru/Listing API'leri (best-effort)
# - CoinMarketCal (opsiyonel API key)
# - Key yoksa heuristik: 1m hacim+fiyat spike haberi taklidi

import time, requests
from typing import Tuple, Optional, Dict

class NewsHunter:
    def __init__(self, cfg: Dict):
        self.enabled = bool(cfg.get("enabled", True))
        self.cooldown = int(cfg.get("cooldown_sec", 300))
        self.last_ts = 0.0
        self.cmc_key = cfg.get("cmc_api_key") or None
        self.cmc_base = "https://developers.coinmarketcal.com/v1/events"
        self.min_qv = float(cfg.get("min_quote_volume_usdt", 10_000_000))
        self.min_1m = float(cfg.get("one_min_change_pct", 1.5))

    def _too_soon(self) -> bool:
        return (time.time() - self.last_ts) < self.cooldown

    # ---- Binance CMS (Listing/Announcements) — best effort, değişebilir ----
    def _binance_listing_ping(self, symbol: str) -> Optional[str]:
        try:
            # Genel CMS feed — kategori filtreleri değişebilir; best effort
            r = requests.get(
                "https://www.binance.com/bapi/composite/v1/public/cms/article/list",
                params={"catalogId":"48","pageSize":20,"pageNo":1}, timeout=4
            )
            j = r.json()
            # başlıklarda parite/simge yakala
            sym = symbol.replace("USDT","")
            for it in j.get("data",{}).get("articles",[]):
                ttl = (it.get("title","") or "").upper()
                if "LISTING" in ttl and sym in ttl:
                    return it.get("title","Binance Listing")
        except Exception:
            return None
        return None

    # ---- CoinMarketCal (opsiyonel anahtar) ----
    def _cmc_events(self, symbol: str) -> Optional[str]:
        if not self.cmc_key: return None
        try:
            headers={"x-api-key": self.cmc_key}
            sym = symbol.replace("USDT","")
            r = requests.get(self.cmc_base, headers=headers, params={"symbols": sym, "max": 5}, timeout=5)
            arr = r.json().get("data",[])
            for e in arr:
                t = (e.get("title","") or "") + " " + (e.get("source","") or "")
                if "LIST" in t.upper() or "MAINNET" in t.upper() or "AIRDROP" in t.upper():
                    return e.get("title","CMC event")
        except Exception:
            return None
        return None

    # ---- Heuristik: 1m spike (haber benzeri ani hareket) ----
    def _heuristic_spike(self, symbol: str) -> Optional[str]:
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params={"symbol": symbol, "interval":"1m", "limit": 3}, timeout=4)
            ks = r.json()
            c1 = float(ks[-2][4]); c2 = float(ks[-1][4])
            chg = (c2/c1 - 1.0) * 100.0
            r2 = requests.get("https://api.binance.com/api/v3/ticker/24hr",
                              params={"symbol": symbol}, timeout=4)
            qv = float(r2.json().get("quoteVolume", 0))
            if chg >= self.min_1m and qv >= self.min_qv:
                return f"1m spike {chg:.2f}% / qv≥{self.min_qv:.0f}"
        except Exception:
            return None
        return None

    def maybe_signal(self, symbol: str) -> Tuple[bool,str]:
        if not self.enabled or self._too_soon():
            return False, ""
        why = (self._binance_listing_ping(symbol)
               or self._cmc_events(symbol)
               or self._heuristic_spike(symbol))
        if why:
            self.last_ts = time.time()
            return True, why
        return False, ""
