# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Optional

# --- Telegram (sessiz fallback)
try:
    from Proje1.core.utils_io import tg_send
except Exception:  # pragma: no cover
    def tg_send(*a, parse_mode: str = "HTML", **k) -> None:  # type: ignore
        return

# --- Güvenli import yardımcıları (paket bağlamı GARANTİ) --------------------
import importlib

def safe_import_knowledge(name: str):
    """
    Modülleri HER ZAMAN fullname ile yükler: knowledge.<name>
    Yol/çıplak isimle yüklemeyi kesinlikle yapmaz.
    """
    fullname = f"knowledge.{name}"
    try:
        mod = importlib.import_module(fullname)
        return mod
    except Exception as e:
        report_exception(f"safe_import:{fullname}", e)
        raise

def diagnose_imports(names: List[str]) -> Dict[str, str]:
    """
    Diagnose için güvenli import testi. Her isim knowledge.<name> olarak denenir.
    """
    out: Dict[str, str] = {}
    for n in names:
        fullname = f"knowledge.{n}"
        try:
            importlib.import_module(fullname)
            out[n] = "ok"
        except Exception as e:
            out[n] = f"err:{e.__class__.__name__}"
            try:
                tg_send(f"🛠️ <b>diagnose</b>: <code>{n}</code> → <code>{e}</code>", parse_mode="HTML")
            except Exception:
                pass
    return out

# ========== Yardımcı ==========
def _utc_ts() -> int:
    return int(time.time())

def _ts_iso(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

def _safe_mkdir(path_like: Path) -> None:
    try:
        Path(path_like).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _write_json(path: Path, obj: Any) -> None:
    try:
        _safe_mkdir(path)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    try:
        _safe_mkdir(path)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _append_csv(path: Path, row: Dict[str, Any], headers: List[str]) -> None:
    try:
        _safe_mkdir(path)
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow({h: row.get(h, "") for h in headers})
    except Exception:
        pass

# ===================================================
#  SelfHeal: sağlık izleyici + otomatik koruyucu (L9)
# ===================================================
class SelfHeal:
    _instance_lock = threading.Lock()
    _instance: Optional["SelfHeal"] = None

    @classmethod
    def get(cls) -> Optional["SelfHeal"]:
        with cls._instance_lock:
            return cls._instance

    @classmethod
    def create_or_get(cls, cfg: dict) -> "SelfHeal":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = SelfHeal(cfg)
            return cls._instance

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        hc = (self.cfg.get("health") or {})
        self.enabled: bool = bool(hc.get("enabled", True))
        self.min_hb_per_min: int = int(hc.get("min_heartbeats_per_min", 30))
        self.max_api_err_5m: int = int(hc.get("max_api_errors_5m", 10))
        self.action: str = str(hc.get("action", "cooldown"))
        self.cooldown_min: int = int(hc.get("cooldown_min", 10))
        self.write_csv: bool = bool(hc.get("write_csv", True))

        # yollar
        self.journal_path = Path("logs/selfheal_journal.jsonl")
        self.csv_path = Path("logs/selfheal_metrics.csv")
        self.state_path = Path("runtime/selfheal_state.json")

        # runtime
        st = _read_json(self.state_path, {})
        self.cool_until: float = float(st.get("cool_until", 0.0))
        self._hb_ts: List[float] = []
        self._api_err_ts: List[float] = []
        self._last_sample_flush: float = 0.0

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # --------------- dış API ---------------
    def heartbeat(self) -> None:
        now = time.time()
        self._hb_ts.append(now)
        cut = now - 120.0
        if len(self._hb_ts) > 256:
            self._hb_ts = [t for t in self._hb_ts if t >= cut]
        else:
            i = 0
            for t in self._hb_ts:
                if t >= cut:
                    break
                i += 1
            if i:
                del self._hb_ts[:i]

    def record_api_error(self, what: str = "api") -> None:
        now = time.time()
        self._api_err_ts.append(now)
        cut = now - 360.0
        self._api_err_ts = [t for t in self._api_err_ts if t >= cut]
        _append_jsonl(self.journal_path, {
            "ts": _utc_ts(), "iso": _ts_iso(), "type": "api_error", "what": what
        })

    def is_cooldown(self) -> bool:
        return time.time() < self.cool_until

    def remaining_cooldown_sec(self) -> int:
        rem = int(self.cool_until - time.time());  return max(0, rem)

    def force_cooldown(self, minutes: Optional[int] = None, reason: str = "manual") -> None:
        mins = int(minutes or self.cooldown_min)
        self.cool_until = time.time() + max(60, mins * 60)
        self._save_state()
        self._announce(f"🧯 <b>SelfHeal</b>: manual cooldown → {mins} dk ({reason}).")

    # --------------- rapor/eylem ---------------
    def _save_state(self) -> None:
        _write_json(self.state_path, {"cool_until": self.cool_until})

    def _announce(self, text: str) -> None:
        try: tg_send(text, parse_mode="HTML")
        except Exception: pass

    def _apply_action(self, reason: str, metrics: Dict[str, Any]) -> None:
        if self.action == "cooldown":
            self.cool_until = time.time() + max(60, self.cooldown_min * 60)
            self._save_state()
            self._announce(f"🟥 <b>SelfHeal</b>: {reason} → cooldown {self.cooldown_min} dk.")
        else:
            self._announce(f"🟧 <b>SelfHeal</b>: {reason} → action={self.action} (log).")
        _append_jsonl(self.journal_path, {
            "ts": _utc_ts(), "iso": _ts_iso(), "type": "action",
            "reason": reason, "metrics": metrics
        })

    # --------------- izleme döngüsü ---------------
    def _metrics_snapshot(self) -> Dict[str, Any]:
        now = time.time()
        hb_60s = [t for t in self._hb_ts if (now - t) <= 60.0]
        err_5m = [t for t in self._api_err_ts if (now - t) <= 300.0]
        return {
            "ts": _utc_ts(), "iso": _ts_iso(),
            "hb_per_min": len(hb_60s),
            "api_err_5m": len(err_5m),
            "cooldown": int(self.remaining_cooldown_sec()),
        }

    def _flush_metrics_csv(self, snap: Dict[str, Any]) -> None:
        if not self.write_csv: return
        headers = ["ts", "iso", "hb_per_min", "api_err_5m", "cooldown"]
        _append_csv(self.csv_path, snap, headers)

    def _loop(self) -> None:
        self._announce("🩺 <b>SelfHeal</b>: watcher başladı (L9).")
        while not self._stop.is_set():
            try:
                if not self.enabled:
                    time.sleep(1.0);  continue

                snap = self._metrics_snapshot()
                now = time.time()

                if now - self._last_sample_flush > 30.0:
                    self._flush_metrics_csv(snap)
                    _append_jsonl(self.journal_path, {"type": "metrics", **snap})
                    self._last_sample_flush = now

                reason: Optional[str] = None
                if snap["hb_per_min"] < self.min_hb_per_min:
                    reason = f"low_heartbeat ({snap['hb_per_min']}/{self.min_hb_per_min})"
                if snap["api_err_5m"] > self.max_api_err_5m:
                    reason = (reason + " + " if reason else "") + f"api_errors_5m({snap['api_err_5m']}>{self.max_api_err_5m})"

                if self.is_cooldown():
                    rem = self.remaining_cooldown_sec()
                    if rem % 60 == 0:
                        _append_jsonl(self.journal_path, {"type": "cooldown_tick", **snap})
                    time.sleep(1.0);  continue

                if reason:
                    self._apply_action(reason, snap)

            except Exception as e:
                _append_jsonl(self.journal_path, {
                    "ts": _utc_ts(), "iso": _ts_iso(),
                    "type": "watcher_error",
                    "err": f"{e}", "trace": traceback.format_exc(limit=3)
                })
                try: tg_send(f"⚠️ <b>SelfHeal loop err</b>: <code>{e}</code>", parse_mode="HTML")
                except Exception: pass
                time.sleep(1.0)

            time.sleep(1.0)

    # --------------- yaşam döngüsü ---------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive(): return
        self._stop.clear()
        t = threading.Thread(target=self._loop, name="selfheal_watcher", daemon=True)
        t.start();  self._thread = t

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            try: self._thread.join(timeout=2.0)
            except Exception: pass

    # --------------- koruma/dekoratör ---------------
    @contextmanager
    def guard(self, section: str) -> ContextManager[None]:
        try:
            yield
        except Exception as e:
            report_exception(section, e);  raise

    def guard_decorator(self, section: str) -> Callable:
        def _wrap(func: Callable) -> Callable:
            def _inner(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    report_exception(section, e);  raise
            return _inner
        return _wrap

# ======================================
#  DIŞ ARAYÜZ: ensure_* ve report_*
# ======================================
_watcher_started = False

def ensure_selfheal_watcher(cfg: Optional[dict] = None) -> None:
    global _watcher_started
    try:
        sh = SelfHeal.create_or_get(cfg or {})
        if not _watcher_started:
            sh.start();  _watcher_started = True
    except Exception as e:
        try: tg_send(f"⚠️ <b>SelfHeal init fail</b>: <code>{e}</code>", parse_mode="HTML")
        except Exception: pass

def report_exception(context: str, e: BaseException) -> None:
    sh = SelfHeal.get() or SelfHeal.create_or_get({})
    row = {
        "ts": _utc_ts(), "iso": _ts_iso(), "type": "exception",
        "context": str(context), "err": f"{e}", "trace": traceback.format_exc()
    }
    _append_jsonl(sh.journal_path, row)

    err_s = f"{e}".lower()
    if any(k in err_s for k in ("http", "api", "request", "network", "timeout", "rate limit")):
        sh.record_api_error(what=context)

    msg = f"🚑 <b>SelfHeal</b> exception @{context}:\n<code>{str(e)[:220]}</code>"
    try: tg_send(msg, parse_mode="HTML")
    except Exception: pass

    if any(k in err_s for k in ("database locked", "out of memory", "cannot import", "keyerror")):
        try: sh.force_cooldown(minutes=max(2, int(sh.cooldown_min / 2)), reason=f"auto_critical:{context}")
        except Exception: pass

# =====================================================
#  Global yardımcı
# =====================================================
def selfheal_heartbeat() -> None:
    sh = SelfHeal.get()
    if sh: sh.heartbeat()

def selfheal_api_error(what: str = "api") -> None:
    sh = SelfHeal.get()
    if sh: sh.record_api_error(what=what)
