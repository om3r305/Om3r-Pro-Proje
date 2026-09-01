# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, requests
from typing import Tuple, Optional, Dict, Any

# --- TG & SelfHeal fallback (sessiz) -----------------------------------------
try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, parse_mode: str = "HTML", **k):  # type: ignore
        return

try:
    from Proje1.core.brain_selfheal import selfheal_api_error, report_exception
except Exception:
    def selfheal_api_error(what: str = "api"):  # type: ignore
        return
    def report_exception(context: str, e: BaseException):  # type: ignore
        try:
            tg_send(f"⚠️ <b>{context}</b>: <code>{e}</code>", parse_mode="HTML")
        except Exception:
            pass

# --- Sabitler ----------------------------------------------------------------
BINANCE_API = "https://api.binance.com"
# Eski CMS endpoint bozulduğu için varsayılan kapalı tutacağız
BINANCE_CMS_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/list"

CMCAL_BASE = "https://developers.coinmarketcal.com/v1/events"          # farklı servis!
CMC_QUOTE  = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"

DEFAULT_TIMEOUT = (4.0, 6.0)  # (connect, read)

class NewsHunter:
    """
    Haber/spike tetikleyici (gürültü korumalı):
      - (ops, default OFF) Binance CMS 'listing' başlığı
      - (ops) CoinMarketCal etkinlikleri (CMCAL_API_KEY varsa)
      - 1m close-close spike + 24h quote volume eşiği (her zaman)
      - (ops) CoinMarketCap quote/volume teyidi (CMC_API_KEY varsa)
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.enabled   = bool(cfg.get("enabled", True))
        self.cooldown  = int(cfg.get("cooldown_sec", 300))
        self.last_ts   = 0.0

        # Eşikler
        self.min_qv    = float(cfg.get("min_quote_volume_usdt", 10_000_000))
        self.min_1m    = float(cfg.get("one_min_change_pct", 1.5))

        # Kaynak bayrakları (override için config keys)
        src = cfg.get("sources") or {}
        self.use_binance_cms = bool(src.get("binance_cms", False))   # <- default False
        self.use_cmcal       = True                                  # cmcal key varsa çalışır
        self.use_cmc_quote   = True                                  # cmc key varsa çalışır

        # Anahtarlar (.env veya cfg)
        self.cmcal_key = os.getenv("CMCAL_API_KEY") or cfg.get("cmcal_api_key")
        self.cmc_key   = os.getenv("CMC_API_KEY")   or cfg.get("cmc_api_key")

        # Sessize alma/backoff (spam önlemek için)
        self._mute_until: Dict[str, float] = {}  # {"binance_cms": ts, "cmcal": ts}
        self._mute_hours = int(cfg.get("mute_hours_on_error", 12))

        # HTTP oturum
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": "Brian-NewsHunter/1.1 (bot)",
            "Accept": "application/json, */*;q=0.1",
        })

    # ---------------- utilities ----------------
    def _too_soon(self) -> bool:
        return (time.time() - self.last_ts) < self.cooldown

    def _muted(self, key: str) -> bool:
        return time.time() < self._mute_until.get(key, 0.0)

    def _mute(self, key: str, hours: Optional[int] = None) -> None:
        self._mute_until[key] = time.time() + 3600.0 * float(hours if hours is not None else self._mute_hours)

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None,
             headers: Optional[Dict[str, str]] = None, timeout=DEFAULT_TIMEOUT) -> Optional[requests.Response]:
        try:
            r = self.s.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            selfheal_api_error("news_api")
            # Hata mesajlarını tek seferlik bildir + backoff
            if code in (401, 403):
                tg_send(f"🟨 Kaynak engelli (HTTP {code}): {url}", parse_mode="HTML")
            elif code == 404:
                tg_send(f"⛔ News API 404: {url}", parse_mode="HTML")
            else:
                tg_send(f"⛔ News API hata {code or '?'}: {url}", parse_mode="HTML")
            return None
        except Exception as e:
            selfheal_api_error("news_api")
            report_exception("news_hunter.get", e)
            return None

    # ---------------- sources ----------------
    def _binance_listing(self, symbol: str) -> Optional[str]:
        """Binance CMS (varsayılan kapalı). 404 yağarsa kaynağı sessize alır."""
        if not self.use_binance_cms or self._muted("binance_cms"):
            return None
        r = self._get(BINANCE_CMS_URL, params={"catalogId": "48", "pageSize": 20, "pageNo": 1})
        if not r:
            # 404 veya başka bir hata geldiyse kaynağı mute et
            self._mute("binance_cms")
            return None
        try:
            sym = symbol.replace("USDT", "")
            j = r.json() or {}
            for it in (j.get("data", {}) or {}).get("articles", []) or []:
                ttl = ((it.get("title") or "") + " " + (it.get("code") or "")).upper()
                if "LISTING" in ttl and sym in ttl:
                    return it.get("title", "Binance Listing")
        except Exception as e:
            report_exception("news_hunter.binance_parse", e)
        return None

    def _cmcal_events(self, symbol: str) -> Optional[str]:
        """CoinMarketCal — yalnız CMCal anahtarı varsa."""
        if not self.cmcal_key or self._muted("cmcal"):
            return None
        headers = {"x-api-key": self.cmcal_key}
        sym = symbol.replace("USDT", "")
        r = self._get(CMCAL_BASE, headers=headers, params={"symbols": sym, "max": 5}, timeout=(4.0, 8.0))
        if not r:
            self._mute("cmcal")
            return None
        try:
            arr = (r.json().get("data") or [])[:5]
            for e in arr:
                t = f"{(e.get('title') or '')} {(e.get('source') or '')}".upper()
                if any(k in t for k in ("LIST", "MAINNET", "AIRDROP", "PARTNERSHIP", "LAUNCH")):
                    return e.get("title", "CMCal event")
        except Exception as e:
            report_exception("news_hunter.cmcal_parse", e)
        return None

    def _heuristic_spike(self, symbol: str) -> Optional[str]:
        """1m close-to-close değişim + 24h quote volume eşiği."""
        r = self._get(f"{BINANCE_API}/api/v3/klines",
                      params={"symbol": symbol, "interval": "1m", "limit": 3})
        if not r:
            return None
        try:
            ks = r.json() or []
            if len(ks) < 2:
                return None
            c1 = float(ks[-2][4]); c2 = float(ks[-1][4])
            chg = (c2 / c1 - 1.0) * 100.0
        except Exception as e:
            report_exception("news_hunter.kline_parse", e)
            return None

        r2 = self._get(f"{BINANCE_API}/api/v3/ticker/24hr", params={"symbol": symbol})
        if not r2:
            return None
        try:
            qv = float((r2.json() or {}).get("quoteVolume", 0.0) or 0.0)
            if chg >= self.min_1m and qv >= self.min_qv:
                return f"⚡ 1m spike {chg:.2f}% • 24h qv≥{self.min_qv:,.0f}"
        except Exception as e:
            report_exception("news_hunter.ticker_parse", e)
        return None

    def _cmc_quote_confirm(self, symbol: str) -> Optional[str]:
        """Opsiyonel CMC (CoinMarketCap) teyidi — FREE planla hafif kullan."""
        if not self.cmc_key:
            return None
        try:
            sym = symbol.replace("USDT", "")
            r = self._get(CMC_QUOTE,
                          params={"symbol": sym, "convert": "USD"},
                          headers={"X-CMC_PRO_API_KEY": self.cmc_key},
                          timeout=(4.0, 8.0))
            if not r:
                return None
            # Bu aşamada sinyal üretmiyoruz; gerekiyorsa burada ek filtreler koyabilirsin.
            return None
        except Exception as e:
            report_exception("news_hunter.cmc_quote", e)
            return None

    # ---------------- public API ----------------
    def maybe_signal(self, symbol: str) -> Tuple[bool, str]:
        if not self.enabled or self._too_soon():
            return False, ""

        # 1) (ops) Binance CMS
        why = self._binance_listing(symbol)
        if why:
            self.last_ts = time.time()
            tg_send(f"📰 <b>News</b> • {symbol} • {why}", parse_mode="HTML")
            return True, why

        # 2) (ops) CMCal events
        why = self._cmcal_events(symbol)
        if why:
            self.last_ts = time.time()
            tg_send(f"🗓️ <b>Event</b> • {symbol} • {why}", parse_mode="HTML")
            return True, why

        # 3) Spike
        why = self._heuristic_spike(symbol)
        if why:
            self.last_ts = time.time()
            tg_send(f"⚡ <b>Spike</b> • {symbol} • {why}", parse_mode="HTML")
            return True, why

        # 4) (ops) CMC quote confirm (sinyal üretmez, sadece arka plan teyit)
        self._cmc_quote_confirm(symbol)
        return False, ""
