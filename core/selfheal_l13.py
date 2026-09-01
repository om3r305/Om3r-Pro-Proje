# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, re, traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Sessiz fallback'lar
try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, **k): pass

try:
    from Proje1.core.brain_hook import brain_overrides, log_brain
except Exception:
    def brain_overrides(*a, **k): pass
    def log_brain(kind, data): pass

# Yol/assetler
RUNTIME = Path("runtime")
FLAGS   = RUNTIME / "flags"
BACKUPS = Path(".patch_backups")           # file_watcher auto_patch buraya yedek koyuyor
JOURNAL = Path("logs/selfheal_l13.jsonl")

for p in (RUNTIME, FLAGS, BACKUPS, JOURNAL.parent):
    p.mkdir(parents=True, exist_ok=True)

# Dahili durum (tekrar eden hata imzası, cooldown vb.)
_ST: Dict[str, Any] = {
    "last_err_sig": None,
    "last_err_ts": 0.0,
    "repeat_count": 0,
    "cool_until": 0.0,
}

# ---- yardımcılar ----
def _cfg(d: Dict[str,Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _write_journal(obj: Dict[str,Any]) -> None:
    rec = {"ts": time.time(), **obj}
    try:
        with JOURNAL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _fingerprint(exc: BaseException) -> str:
    # Kısa parmak izi: Exception sınıfı + ilk satır mesajı
    name = exc.__class__.__name__
    msg  = str(exc).strip().splitlines()[:1]
    return f"{name}:{msg[0] if msg else ''}"[:180]

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _restore_latest_backup(target: Path) -> Optional[Path]:
    """
    Hata bir dosyayla ilişkilendirilebiliyorsa (.py), .patch_backups altındaki
    en yeni kopyayı bulup geri yükler (best-effort).
    """
    try:
        if not target.exists():
            return None
        cand = sorted(BACKUPS.glob(f"{target.name}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cand:
            return None
        latest = cand[0]
        data = latest.read_bytes()
        target.write_bytes(data)
        return latest
    except Exception:
        return None

# ---- eşleştirme kuralları (pattern -> aksiyon etiketi) ----
_PATTERNS = [
    (re.compile(r"division by zero", re.I),                   "risk_bump"),
    (re.compile(r"rate limit|429|Too Many Requests", re.I),   "throttle"),
    (re.compile(r"KeyError|AttributeError", re.I),            "rollback"),
    (re.compile(r"JSONDecodeError|invalid json", re.I),       "throttle"),
    (re.compile(r"Timeout|ReadTimeout|Connection", re.I),     "throttle"),
]

def _decide_action(exc: BaseException) -> str:
    s = f"{exc.__class__.__name__}: {str(exc)}"
    for pat, tag in _PATTERNS:
        if pat.search(s):
            return tag
    return "generic"

# ---- kamu fonksiyonları ----
def ensure_autorepair(cfg: Dict[str,Any]) -> None:
    """
    L13’i aktif etmek için init’te bir kez çağır.
    """
    _write_journal({"kind":"boot", "msg":"L13 autorepair online"})

def l13_heartbeat() -> None:
    """
    Döngü içinde hafif heartbeat. Cooldown durumunu temizler.
    """
    if _ST["cool_until"] and time.time() > _ST["cool_until"]:
        _ST["cool_until"] = 0.0
        _write_journal({"kind":"cooldown_end"})

def l13_on_exception(where: str, exc: BaseException, cfg: Dict[str,Any]) -> None:
    """
    Bot döngüsündeki except bloğunda çağır. Tekrarlı hatalarda
    otomatik risk sıkma / evo tetik / rollback dener.
    """
    sig = _fingerprint(exc)
    now = time.time()

    if _ST["last_err_sig"] == sig and now - float(_ST["last_err_ts"] or 0) < 180:
        _ST["repeat_count"] = int(_ST.get("repeat_count", 0)) + 1
    else:
        _ST["repeat_count"] = 1

    _ST["last_err_sig"] = sig
    _ST["last_err_ts"]  = now

    action = _decide_action(exc)
    rep = {
        "kind": "exception",
        "where": where,
        "sig": sig,
        "repeat": _ST["repeat_count"],
        "action": action,
    }
    _write_journal(rep)
    log_brain("selfheal_exc", rep)

    # Çok sık tekrar ediyorsa -> aksiyon
    repeat_soft = int(_cfg(cfg, "selfheal.repeat_soft", 2))
    repeat_hard = int(_cfg(cfg, "selfheal.repeat_hard", 4))

    if _ST["repeat_count"] >= repeat_soft:
        _apply_soft_measures(cfg, action)

    if _ST["repeat_count"] >= repeat_hard:
        _apply_hard_measures(cfg, action)

def _apply_soft_measures(cfg: Dict[str,Any], action: str) -> None:
    """
    Küçük ayarlar: veto +0.01…0.03, entry’leri x0.95, evo force flag.
    """
    lo = float(_cfg(cfg, "learning.autopatch.lo", 0.45) or 0.45)
    hi = float(_cfg(cfg, "learning.autopatch.hi", 0.80) or 0.80)
    cur= float(_cfg(cfg, "brain.veto_conf_min", 0.55)  or 0.55)

    bump = 0.02 if action in ("generic","risk_bump") else 0.01
    new_veto = _clamp(cur + bump, lo, hi)

    # entry_frac’leri %5 daralt
    ef = dict(_cfg(cfg, "entry_frac", {})) or {"dip":0.40,"pred":0.40,"news":0.60,"ob":0.50}
    for k in list(ef.keys()):
        ef[k] = round(_clamp(float(ef[k]) * 0.95, 0.20, 0.80), 2)

    overrides = {"brain.veto_conf_min": round(new_veto,3)}
    overrides.update({f"entry_frac.{k}": v for k,v in ef.items()})
    try: brain_overrides(overrides, cfg)
    except Exception: pass

    # Evo’yu dürt
    try:
        (FLAGS / "force_evo.json").write_text(
            json.dumps({"ts": time.time(), "why": "selfheal_soft"}), encoding="utf-8"
        )
    except Exception: pass

    msg = f"🟠 SelfHeal (soft): action={action} → veto→{new_veto:.3f}, entry x0.95 • evo=force"
    _write_journal({"kind":"soft", "overrides":overrides})
    log_brain("selfheal_soft", {"action":action, "overrides":overrides})
    try: tg_send(msg)
    except Exception: pass

def _apply_hard_measures(cfg: Dict[str,Any], action: str) -> None:
    """
    Sert hamle: veto +0.04, entry x0.85, olabiliyorsa rollback denemesi.
    """
    lo = float(_cfg(cfg, "learning.autopatch.lo", 0.45) or 0.45)
    hi = float(_cfg(cfg, "learning.autopatch.hi", 0.80) or 0.80)
    cur= float(_cfg(cfg, "brain.veto_conf_min", 0.55)  or 0.55)

    bump = 0.04 if action in ("generic","risk_bump") else 0.02
    new_veto = _clamp(cur + bump, lo, hi)

    # entry_frac %15 daralt
    ef = dict(_cfg(cfg, "entry_frac", {})) or {"dip":0.40,"pred":0.40,"news":0.60,"ob":0.50}
    for k in list(ef.keys()):
        ef[k] = round(_clamp(float(ef[k]) * 0.85, 0.20, 0.80), 2)

    overrides = {"brain.veto_conf_min": round(new_veto,3)}
    overrides.update({f"entry_frac.{k}": v for k,v in ef.items()})
    try: brain_overrides(overrides, cfg)
    except Exception: pass

    # Eğer son değişen dosya bilgisi varsa (file_watcher yazmış olabilir):
    # best-effort rollback: son dokunulan .py dosyasını yedekten geri getir
    restored: Optional[str] = None
    try:
        last_file_flag = FLAGS / "last_touched.json"
        if last_file_flag.exists():
            info = json.loads(last_file_flag.read_text(encoding="utf-8"))
            p = Path(info.get("path",""))
            if p.suffix == ".py" and p.exists():
                r = _restore_latest_backup(p)
                if r: restored = f"{p.name} ← {r.name}"
    except Exception:
        pass

    # Kısa cooldown
    cool_min = int(_cfg(cfg, "selfheal.cooldown_min", 10))
    _ST["cool_until"] = time.time() + max(60, cool_min*60)

    # Evo’yu tekrar dürt
    try:
        (FLAGS / "force_evo.json").write_text(
            json.dumps({"ts": time.time(), "why": "selfheal_hard"}), encoding="utf-8"
        )
    except Exception: pass

    msg = f"🔴 SelfHeal (HARD): action={action} → veto→{new_veto:.3f}, entry x0.85, rollback={restored or 'n/a'} • cooldown {cool_min}m"
    _write_journal({"kind":"hard", "overrides":overrides, "rollback": restored})
    log_brain("selfheal_hard", {"action":action, "overrides":overrides, "rollback":restored})
    try: tg_send(msg)
    except Exception: pass
