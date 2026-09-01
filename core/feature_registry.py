# -*- coding: utf-8 -*-
from __future__ import annotations
import importlib, inspect, time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Any, List, Tuple, Sequence, Union

ImportPath = Union[str, Sequence[str]]

@dataclass
class Feature:
    name: str                    # "dip", "pred", "ob", "candles_guard", "ev_filter", "autopilot"...
    kind: str                    # "signal" | "risk" | "io" | "engine" | "meta" | "ui"
    config_key: Optional[str]    # cfg kök anahtar (örn. "dip", "predictor", "risk", "telegram")
    import_path: Optional[ImportPath]  # modül yolu (None ise sadece config ile kontrol)
    affects: List[str] = field(default_factory=list)  # "entry","exit","alloc","report","dashboard","config"
    enabled: Optional[bool] = None     # cfg'ye göre aktif olması beklenen
    available: Optional[bool] = None   # modül import edilebiliyor mu?
    reason: str = ""                   # pasifse neden?
    file: Optional[str] = None         # modül dosya yolu
    used_count: int = 0                # runtime kullanım sayacı
    last_used_ts: float = 0.0          # son kullanım zamanı

_REG: Dict[str, Feature] = {}

# ---------------- import yardımcıları ----------------
def _candidates_for(path: str) -> List[str]:
    """
    Verilen import path için denenecek adaylar.
    Örn: "core.dip_tracker" -> ["core.dip_tracker", "Proje1.core.dip_tracker"]
    """
    cands = [path]
    if not path.startswith("Proje1."):
        cands.append(f"Proje1.{path}")
    return cands

def _try_import_one(path: str) -> Tuple[bool, Optional[str]]:
    try:
        mod = importlib.import_module(path)
        src = inspect.getsourcefile(mod) or inspect.getfile(mod)
        return True, src
    except Exception:
        return False, None

def _try_import(paths: Optional[ImportPath]) -> Tuple[bool, Optional[str]]:
    if not paths:
        return True, None
    # tek string ise tuple'a çevir
    seq: Sequence[str] = (paths,) if isinstance(paths, str) else tuple(paths)
    for p in seq:
        for cand in _candidates_for(p):
            ok, src = _try_import_one(cand)
            if ok:
                return True, src
    return False, None

# ---------------- kayıt & özet ----------------
def register_feature(name: str, *, kind: str, config_key: Optional[str],
                     import_path: Optional[ImportPath], affects: List[str]) -> None:
    if name in _REG:
        return
    _REG[name] = Feature(
        name=name, kind=kind, config_key=config_key,
        import_path=import_path, affects=list(affects or [])
    )

def bootstrap_from_cfg(cfg: Dict[str, Any]) -> None:
    """cfg’ye ve import durumuna göre enabled/available/why doldur."""
    for f in _REG.values():
        # enabled: config_key yoksa default True; varsa dict ise enabled anahtarına; değilse truthy
        if f.config_key is None:
            f.enabled = True
        else:
            node = cfg.get(f.config_key, None)
            if isinstance(node, dict):
                f.enabled = bool(node.get("enabled", True))
            else:
                f.enabled = bool(node)
        avail, src = _try_import(f.import_path)
        f.available = avail
        f.file = src
        if not f.enabled:
            f.reason = f"disabled via cfg.{f.config_key}"
        elif not f.available:
            f.reason = f"module import failed: {f.import_path}"
        else:
            f.reason = "ok"

def mark_used(name: str) -> None:
    f = _REG.get(name)
    if not f:
        return
    f.used_count += 1
    f.last_used_ts = time.time()

def snapshot() -> Dict[str, Any]:
    """Dashboard/CLI için JSON serializable özet."""
    out = []
    for f in sorted(_REG.values(), key=lambda x: (x.kind, x.name)):
        out.append({
            "name": f.name,
            "kind": f.kind,
            "affects": f.affects,
            "enabled": f.enabled,
            "available": f.available,
            "status": ("active" if (f.enabled and f.available) else "inactive"),
            "reason": f.reason,
            "config_key": f.config_key,
            "import_path": f.import_path,
            "file": f.file,
            "used_count": f.used_count,
            "last_used_ts": int(f.last_used_ts) if f.last_used_ts else 0,
        })
    return {"generated_at": int(time.time()), "features": out}

def snapshot_text() -> str:
    """Terminal için okunaklı tablo benzeri metin."""
    data = snapshot()["features"]
    hdr = f"{'Status':8} {'Name':14} {'Kind':9} {'Enabled':8} {'Avail':6} {'Used':5}  {'Config':12}  {'Why/Where'}"
    lines = [hdr, "-" * len(hdr)]
    for d in data:
        status = "✅ACTIVE" if d["status"] == "active" else "⛔INACTV"
        lines.append(
            f"{status:8} {d['name'][:14]:14} {d['kind'][:9]:9} "
            f"{str(d['enabled']):8} {str(d['available']):6} {str(d['used_count']):5}  "
            f"{(d['config_key'] or '-')[:12]:12}  {d['reason'] or d['file'] or ''}"
        )
    return "\n".join(lines)

# ---- Varsayılan kayıtlar ----
def register_defaults():
    # Sinyaller
    register_feature("dip",   kind="signal", config_key="dip",
                     import_path=("core.dip_tracker", "Proje1.core.dip_tracker"),
                     affects=["entry"])
    register_feature("pred",  kind="signal", config_key="predictor",
                     import_path=("core.pump_predictor", "Proje1.core.pump_predictor"),
                     affects=["entry"])
    register_feature("news",  kind="signal", config_key="news_mode",
                     import_path=None,  # NewsHunter dışarıdan enjekte ediliyor
                     affects=["entry","exit"])
    register_feature("ob",    kind="signal", config_key="orderbook",
                     import_path=None,  # orderbook sinyali SymbolEngine içinde
                     affects=["entry"])

    # Guard / Risk
    register_feature("candles_guard", kind="risk", config_key="candles",
                     import_path=("core.candles_guard", "Proje1.core.candles_guard"),
                     affects=["entry","exit"])
    register_feature("ev_filter",     kind="risk", config_key=None,
                     import_path=None,
                     affects=["entry"])
    register_feature("daily_risk",    kind="risk", config_key="risk",
                     import_path=("core.risk", "Proje1.core.risk"),
                     affects=["global"])

    # IO / Bildirim / UI
    register_feature("telegram", kind="io",  config_key="telegram",
                     import_path=("core.telegram_utils", "Proje1.core.telegram_utils"),
                     affects=["report"])
    register_feature("reporter", kind="io",  config_key="report",
                     import_path=("core.reporting", "Proje1.core.reporting"),
                     affects=["report"])
    register_feature("dashboard",kind="ui",  config_key=None,
                     import_path=("dashboard", "Proje1.dashboard"),
                     affects=["ui"])

    # Meta / Engine
    register_feature("watchlist", kind="engine", config_key=None,
                     import_path=("core.watchlist_manager", "Proje1.core.watchlist_manager"),
                     affects=["entry"])
    register_feature("state",     kind="engine", config_key=None,
                     import_path=("core.state", "Proje1.core.state"),
                     affects=["persistence"])

    # Ekosistem / opsiyoneller
    register_feature("autopilot",     kind="meta", config_key="autopilot",
                     import_path=("core.autopilot", "Proje1.core.autopilot"),
                     affects=["config"])
    register_feature("adaptive",      kind="meta", config_key="adaptive",
                     import_path=("core.adaptive", "Proje1.core.adaptive"),
                     affects=["config"])
    register_feature("ie_predictor",  kind="meta", config_key="iepredictor",
                     import_path=("core.ie_predictor", "Proje1.core.ie_predictor"),
                     affects=["entry"])
    register_feature("meta_brain",    kind="meta", config_key="meta_brain",
                     import_path=("meta.brain_engine", "Proje1.meta.brain_engine"),
                     affects=["config","ui"])
    register_feature("security_mgr",  kind="meta", config_key="security",
                     import_path=("core.security", "Proje1.core.security"),
                     affects=["io"])
    register_feature("backtest",      kind="meta", config_key=None,
                     import_path=("core.backtest", "Proje1.core.backtest"),
                     affects=["offline"])
