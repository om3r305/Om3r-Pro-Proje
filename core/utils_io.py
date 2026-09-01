# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import json
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Iterable

# requests + retry adapter
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    HTTPAdapter = object  # type: ignore
    Retry = object  # type: ignore

__all__ = [
    "ensure_tg", "tg_ready", "tg_send", "tg_send_long",
    "log_event", "log_trade", "write_event_cash",
    "ev_ok", "reason_tag", "tg_log",
]

# =========================
#  küçük yardımcılar
# =========================
def _ts() -> int:
    return int(time.time())

def _iso(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

def _safe_mkdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    try:
        _safe_mkdir(path)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        # log yazarken asla uygulamayı düşürme
        pass


# --- Telegram günlük log (OUT/IN/ERR) ---
def _cfg_get(d, path, default=None):
    cur = d or {}
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def tg_log(kind: str, text: str, extra: dict | None = None, cfg: dict | None = None) -> None:
    """
    Günlük düz metin + JSONL telegram logu.
    Config anahtarları:
      logging.telegram_daily.enabled: bool
      logging.telegram_daily.dir: str (default logs/telegram_out)
      logging.telegram_daily.max_line_len: int (default 4000)
      logging.telegram_daily.mirror_jsonl: bool (default True)
    """
    try:
        if not _cfg_get(cfg, "logging.telegram_daily.enabled", False):
            return
        base = _cfg_get(cfg, "logging.telegram_daily.dir", "logs/telegram_out")
        Path(base).mkdir(parents=True, exist_ok=True)

        ts = time.time()
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        plain_path = Path(base) / f"{day}.log"
        jsonl_path = Path(base) / f"{day}.jsonl"

        maxlen = int(_cfg_get(cfg, "logging.telegram_daily.max_line_len", 4000) or 4000)
        safe_text = (text or "")
        if len(safe_text) > maxlen:
            safe_text = safe_text[:maxlen] + "…[trunc]"

        # düz metin
        hhmmss = time.strftime("%H:%M:%S", time.localtime(ts))
        line = f"[{hhmmss}] {kind.upper()} {safe_text}\n"
        with plain_path.open("a", encoding="utf-8") as f:
            f.write(line)

        # jsonl aynası (öğrenme için)
        if _cfg_get(cfg, "logging.telegram_daily.mirror_jsonl", True):
            rec = {"ts": ts, "kind": kind, "text": safe_text}
            if extra: rec["extra"] = extra
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # log yazımı başarısız olursa sessiz geç (ana akışı bozma)
        pass


# =========================
#  TELEGRAM (retry/backoff)
# =========================
_TG_LOCK = threading.Lock()
_TG: Dict[str, Any] = {
    "ready": False,
    "token": None,
    "chat": None,
    "enabled": True,
    "session": None,
    "last_sent": 0.0,
    "min_interval": 0.6,  # saniye (throttle)
    "cfg": None,          # ensure_tg(cfg) ile aktarılır → tg_send burada kullanır
}

def _load_dotenv_into_environ() -> None:
    """Basit .env yükleyici (setli env değişkenlerini ezmez)."""
    try_paths = [
        Path(".env"),
        Path("config/.env"),
        Path("Proje1/.env"),
        Path("proje1/.env"),
    ]
    for p in try_paths:
        try:
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip(); v = v.strip()
                    os.environ.setdefault(k, v)
        except Exception:
            pass

def _new_session() -> Optional["requests.Session"]:
    if requests is None:
        return None
    s = requests.Session()
    try:
        # urllib3 v1/v2 uyumlu Retry ayarı
        retry_kwargs = dict(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        try:
            # v2: allowed_methods
            retry = Retry(allowed_methods=frozenset({"POST", "GET"}), **retry_kwargs)  # type: ignore
        except TypeError:
            # v1: method_whitelist
            retry = Retry(method_whitelist=frozenset({"POST", "GET"}), **retry_kwargs)  # type: ignore
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    except Exception:
        pass
    return s

def _parse_bool(x: Any, default: bool = True) -> bool:
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default

def tg_ready() -> bool:
    with _TG_LOCK:
        return bool(_TG.get("ready"))

def ensure_tg(cfg: dict | None) -> None:
    """
    Telegram ayarlarını config + .env + ortamdan birleştirir.
    Hata olsa bile sessizce tamamlar.
    """
    _load_dotenv_into_environ()
    c = cfg or {}
    tg = (c.get("telegram") or {}) if isinstance(c, dict) else {}

    token = (
        tg.get("token")
        or os.environ.get("TG_TOKEN")
        or os.environ.get("TELEGRAM_TOKEN")
        or ""
    )
    chat_raw = (
        tg.get("chat_id")
        or os.environ.get("TG_CHAT_ID")
        or os.environ.get("TELEGRAM_CHAT_ID")
        or ""
    )
    # chat id'yi güvenli stringe çevir (negatif int olabilir)
    chat = str(chat_raw).strip() if str(chat_raw).strip() else None

    enabled = _parse_bool(
        tg.get("enabled") if isinstance(tg, dict) and "enabled" in tg else os.environ.get("TELEGRAM_ENABLED"),
        default=True,
    )

    with _TG_LOCK:
        _TG["token"] = (token.strip() or None)
        _TG["chat"] = chat
        _TG["enabled"] = bool(enabled)
        _TG["session"] = _new_session() if enabled else None
        _TG["ready"] = bool(_TG["enabled"] and _TG["token"] and _TG["chat"])
        _TG["cfg"] = c  # ← tg_send içinde günlükleme configine erişmek için

def _tg_post(session: "requests.Session", url: str, payload: dict) -> None:
    try:
        r = session.post(url, json=payload, timeout=6)
        # Başarısız olursa da patlatma; sadece logla
        if hasattr(r, "status_code") and int(r.status_code) >= 400:
            _append_jsonl(Path("logs/telegram.jsonl"), {
                "ts": _ts(), "iso": _iso(), "type": "send_http_error",
                "code": int(r.status_code),
                "text": (getattr(r, "text", "") or "")[:300]
            })
    except Exception as e:
        _append_jsonl(Path("logs/telegram.jsonl"), {
            "ts": _ts(), "iso": _iso(), "type": "send_error",
            "err": f"{e}", "trace": traceback.format_exc(limit=2)
        })

def tg_send(text: str, parse_mode: str = "HTML") -> None:
    """
    Telegram'a mesaj gönder. Ayar yoksa/no-op; hata olsa bile patlatmaz.
    Ayrıca, logging.telegram_daily.enabled true ise günlük dosyasına OUT olarak yazar.
    """
    try:
        with _TG_LOCK:
            ready = bool(_TG.get("ready"))
            token = _TG.get("token")
            chat = _TG.get("chat")
            session = _TG.get("session")
            min_interval = float(_TG.get("min_interval", 0.6))
            last_sent = float(_TG.get("last_sent", 0.0))
            cfg = _TG.get("cfg")

        # Günlük OUT aynası (göndermeye çalışmadan önce bile kayıt altına al)
        try:
            tg_log("OUT", text, {"parse_mode": parse_mode}, cfg=cfg)
        except Exception:
            pass

        if not ready or requests is None or session is None:
            return

        # throttle
        now = time.time()
        wait = last_sent + min_interval - now
        if wait > 0:
            time.sleep(wait)

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat,
            "text": text,
            "disable_web_page_preview": True,
            "parse_mode": parse_mode,
        }
        try:
            _tg_post(session, url, payload)
        finally:
            with _TG_LOCK:
                _TG["last_sent"] = time.time()

    except Exception as e:
        _append_jsonl(Path("logs/telegram.jsonl"), {
            "ts": _ts(), "iso": _iso(), "type": "send_error",
            "err": f"{e}", "trace": traceback.format_exc(limit=2)
        })

def _chunk_text(s: str, n: int) -> Iterable[str]:
    # Telegram limiti 4096; emniyet payı ile böl
    n = max(512, int(n))
    for i in range(0, len(s), n):
        yield s[i:i+n]

def tg_send_long(text: str, parse_mode: str = "HTML", chunk: int = 3900) -> None:
    """
    Uzun mesajları güvenle böler ve sırayla yollar.
    """
    if not isinstance(text, str):
        text = str(text)
    for part in _chunk_text(text, chunk):
        tg_send(part, parse_mode=parse_mode)

# =========================
#  LOG yardımcıları
# =========================
def log_event(event: str, **kwargs) -> None:
    _append_jsonl(Path("logs/events.jsonl"), {
        "ts": _ts(), "iso": _iso(), "type": str(event), "payload": kwargs
    })

def log_trade(slot: str, sym: str, pnl: float, extra: dict | None = None) -> None:
    row: Dict[str, Any] = {
        "ts": _ts(), "iso": _iso(),
        "slot": str(slot), "symbol": str(sym), "pnl": float(pnl),
    }
    if extra:
        row.update(extra)
    _append_jsonl(Path("logs/trades.jsonl"), row)

def write_event_cash(cash: float, open_positions: int = 0) -> None:
    _append_jsonl(Path("logs/cash_stream.jsonl"), {
        "ts": _ts(), "iso": _iso(),
        "cash": float(cash), "open_positions": int(open_positions),
    })


# =========================
#  EV (expected value) filtresi
# =========================
def ev_ok(price: float, tp_abs: float, sl_abs: float,
          conf: float, fee_pct: float, slip_pct: float,
          ev_min: float = 0.0) -> Tuple[bool, float]:
    """
    Basit EV: p*getiri – (1-p)*kayıp, round-trip maliyeti dahil.
    tp_abs/sl_abs: mutlak fark (price + tp_abs / price + sl_abs)
    """
    try:
        roundtrip_cost = price * (fee_pct + slip_pct) * 2.0
        gain = max(0.0, tp_abs) - roundtrip_cost
        loss = abs(min(0.0, sl_abs)) + roundtrip_cost
        p = max(0.0, min(1.0, float(conf)))
        ev = p * gain - (1.0 - p) * loss
        return (ev >= float(ev_min)), ev
    except Exception:
        # filtre güvenli tarafta kalsın
        return True, 0.0


# =========================
#  Etiket yardımcıları
# =========================
_TAGS = {
    "DIP": "dip", "PRED": "pred", "NEWS": "news",
    "OB": "ob", "ORDERBOOK": "ob",
    "TP": "tp", "SL": "sl",
    "TIME_SL": "time", "BREAKEVEN/MINPROFIT": "be",
}

def reason_tag(label: Optional[str]) -> str:
    if not label:
        return "-"
    s = str(label).strip().upper()
    if s in _TAGS:
        return _TAGS[s]
    # kısa/temiz bir fallback üret
    s = "".join(ch for ch in s if (ch.isalnum() or ch in ("_", "-")))
    return s[:16].lower() if s else "-"
