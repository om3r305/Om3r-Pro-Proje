# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Tuple, List, Dict

try:
    from telegram_utils import tg_send
except Exception:
    def tg_send(*a,parse_mode="HTML",**k): pass

LOG_DIR = Path("logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
SCHEMA_LOG = LOG_DIR / "schema_fixes.jsonl"

def _wjsonl(path: Path, obj: dict) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

# --- Kurallar ---
# path tuple: (root_key, sub_key, ...)
# type: beklenen_type(ler)
# coerce: özel dönüştürücü fonksiyon adı (opsiyonel)
# min/max: aralık sınırları (opsiyonel)
# default: değer yoksa kullanılacak (opsiyonel)
RULES: Dict[Tuple[str, ...], Dict[str, Any]] = {
    # FW hatanı birebir karşılayan kural (dict → int)
    ("file_watcher", "quality_rules", "min_lines"): {
        "type": (int, float),
        "coerce": "coerce_min_lines",
        "min": 10,
        "max": 2000,
        "default": 80
    },
    # Güvenli sayı alanları örnekleri
    ("brain", "veto_conf_min"): {"type": (int, float), "min": 0.0, "max": 1.0, "default": 0.55},
    ("predictor", "enter_prob"): {"type": (int, float), "min": 0.40, "max": 0.95, "default": 0.70},
}

def coerce_min_lines(v: Any) -> int:
    # dict -> int: {"py":80} gibi gelirse py veya ilk sayı değeri alınır
    if isinstance(v, dict):
        for k, val in v.items():
            try:
                return int(val)
            except Exception:
                continue
        return 80
    try:
        return int(v)
    except Exception:
        return 80

# ---------- yardımcılar ----------
def _get_ref(cfg: dict, path: Tuple[str, ...]):
    cur = cfg
    for p in path[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return None, None
        cur = cur[p]
    return cur, path[-1] if path else None

def _backup_file(p: Path) -> None:
    try:
        bdir = Path(".patch_backups"); bdir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        (bdir / f"{p.name}.{ts}.bak").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

def _apply_rule(cfg: dict, path: Tuple[str, ...], spec: Dict[str, Any], fixes: List[dict]) -> None:
    parent, last = _get_ref(cfg, path)
    if parent is None or last is None:
        return
    cur = parent.get(last, spec.get("default"))

    # boşsa default
    if cur is None and "default" in spec:
        parent[last] = spec["default"]
        fixes.append({"path": path, "old": None, "new": parent[last], "why": "missing->default"})
        return

    # coerce
    if "coerce" in spec:
        fn = globals().get(spec["coerce"])
        if callable(fn):
            newv = fn(cur)
            if newv != cur:
                parent[last] = newv
                fixes.append({"path": path, "old": cur, "new": newv, "why": "coerce" })
                cur = newv

    # type kontrol
    if "type" in spec and not isinstance(cur, spec["type"]):
        try:
            # basit cast dene
            if int in spec["type"]:
                newv = int(cur)
            elif float in spec["type"]:
                newv = float(cur)
            else:
                raise ValueError()
            fixes.append({"path": path, "old": cur, "new": newv, "why": "type_cast"})
            parent[last] = newv
            cur = newv
        except Exception:
            # tip uygunsuz ve cast edilemedi → default
            if "default" in spec:
                fixes.append({"path": path, "old": cur, "new": spec["default"], "why": "type_default"})
                parent[last] = spec["default"]
                cur = parent[last]

    # range clamp
    mn, mx = spec.get("min"), spec.get("max")
    if isinstance(cur, (int, float)) and (mn is not None or mx is not None):
        newv = cur
        if mn is not None and newv < mn: newv = mn
        if mx is not None and newv > mx: newv = mx
        if newv != cur:
            parent[last] = newv
            fixes.append({"path": path, "old": cur, "new": newv, "why": "range_clip"})

def validate_and_fix(config_path: str | Path, extra_rules: Dict[Tuple[str, ...], Dict[str, Any]] | None = None) -> dict:
    """config json’ı oku → RULES + extra_rules uygula → değiştiyse yaz ve logla. Geri dönen: (cfg)"""
    p = Path(config_path)
    if not p.exists():
        return {}
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

    rules = dict(RULES)
    if extra_rules:
        rules.update(extra_rules)

    fixes: List[dict] = []
    for path, spec in rules.items():
        try:
            _apply_rule(cfg, path, spec, fixes)
        except Exception:
            continue

    if fixes:
        _backup_file(p)
        p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        for fx in fixes:
            fx["ts"] = time.time()
            _wjsonl(SCHEMA_LOG, fx)
        try:
            tg_send(f"🩺 L9/Schema: {len(fixes)} alan düzeltildi.", parse_mode="HTML")
        except Exception:
            pass
    return cfg
