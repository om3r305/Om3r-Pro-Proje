# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, random, importlib.util, re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Basit log güvenliği (opsiyonel)
try:
    from Proje1.core.logger_utils import log_event
except Exception:
    def log_event(kind, **kw): pass

BLOCKS_DIR = Path("Proje1/core/blocks_autogen")

REGISTRY_PATH = Path("model/blocks_registry.json")

DEFAULT_REGISTRY = {"blocks": {}}

# ---------------------
# IO Helpers
# ---------------------
def _read_json(p: Path, default):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return json.loads(json.dumps(default))

def _write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def save_registry(reg: Dict[str,Any]):
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    _write_json(tmp, reg)
    tmp.replace(REGISTRY_PATH)

def load_registry() -> Dict[str,Any]:
    return _read_json(REGISTRY_PATH, DEFAULT_REGISTRY)

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)[:80]

# ---------------------
# Import Cache
# ---------------------
_MODULE_CACHE: dict[str, object] = {}

def _dyn_import(path: Path):
    key = str(path.resolve())
    mod = _MODULE_CACHE.get(key)
    if mod:
        return mod
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    _MODULE_CACHE[key] = mod
    return mod

# ---------------------
# BLOK ÇALIŞTIRICI API
# ---------------------
def run_block(st, name: str, params: Dict[str,Any]) -> Dict[str,Any]:
    """
    Dinamik bloğu yükleyip çalıştırır.
    DÖNÜŞ: dict (mid/up/dn/hi/lo/r/k/o/atr/value ... hangisini üretirse)
    """
    reg = load_registry()
    meta = (reg.get("blocks") or {}).get(name)
    if not meta:
        raise RuntimeError(f"unknown block '{name}'")
    path = BLOCKS_DIR / meta["file"]
    mod = _dyn_import(path)
    if not hasattr(mod, "run"):
        raise RuntimeError(f"block '{name}' has no run()")
    try:
        out = mod.run(st, params or {})
        if not isinstance(out, dict):
            return {"value": float(out)}
        exp_keys = {"mid","up","dn","hi","lo","r","k","o","atr","value"}
        if not (set(out.keys()) & exp_keys):
            log_event("block_run_unexpected_keys", name=name, keys=list(out.keys()))
        return out
    except Exception as e:
        log_event("block_run_fail", name=name, err=str(e))
        raise

# ---------------------
# BLOK ÜRETİM API
# ---------------------
def _emit_block_module(name: str, code_body: str) -> Path:
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    (BLOCKS_DIR / "__init__.py").touch(exist_ok=True)
    header = f"""# AUTOGEN BLOCK (L60+) — {name}
# created_ts: {time.time()}
# NOTE: return dict keys may include: mid, up, dn, hi, lo, r, k, o, atr, value

"""
    full = header + code_body.strip() + "\n"
    p = BLOCKS_DIR / f"{name}.py"
    p.write_text(full, encoding="utf-8")
    return p

def register_block(name: str, spec: Dict[str,Any], path: Path):
    reg = load_registry()
    reg["blocks"][name] = {
        "file": path.name,
        "created_ts": time.time(),
        "spec": spec or {}
    }
    save_registry(reg)

def ensure_block(name: str, spec: Dict[str,Any], code_body: str) -> Path:
    """
    Yoksa üretir, varsa dokunmaz.
    """
    name = _sanitize(name)
    reg = load_registry()
    if name in (reg.get("blocks") or {}):
        return BLOCKS_DIR / reg["blocks"][name]["file"]
    p = _emit_block_module(name, code_body)
    try:
        _dyn_import(p)
    except Exception as e:
        bad = p.with_suffix(".bad.py")
        p.replace(bad)
        log_event("block_emit_bad", name=name, err=str(e), file=str(bad))
        raise
    register_block(name, spec, p)
    return p

# ---------------------
# İCAT – Heuristik
# ---------------------
def _pz(f: float, lo=-3.0, hi=3.0):
    return max(lo, min(hi, f))

def invent_block(diag: Dict[str,Any]|None) -> Tuple[str, Dict[str,Any], str]:
    reg = ((diag or {}).get("hot") or {}).get("reg", "UNKNOWN")
    suffix = f"{int(time.time())}_{random.randint(100,999)}"

    if reg == "CHOP":
        name = f"range_compressor_{suffix}"
        spec = {"shape": ["mid","up","dn"], "idea": "BB+range squeeze with adaptive width"}
        code = f"""
from typing import Dict, Any
def _ema(st, n): return getattr(st, "ema", lambda k: st.last_px)(int(n))
def _vol(st): return max(0.001, float(getattr(st, "vol_norm", 0.2)))
def run(st, params: Dict[str,Any]) -> Dict[str,Any]:
    p = int(params.get("period", 20))
    dev = float(params.get("dev", 2.0))
    mid = _ema(st, p)
    width = dev * _vol(st) * float(getattr(st, "last_px", 0.0))
    return {{"mid": mid, "up": mid + width, "dn": mid - width}}
"""
        return name, spec, code

    if reg == "TREND":
        name = f"momentum_flux_{suffix}"
        spec = {"shape":["value","k"], "idea":"kama-like drift + price momentum mix"}
        code = f"""
from typing import Dict, Any
def _ema(st, n): return getattr(st, "ema", lambda k: st.last_px)(int(n))
def _rsi(st, n): return getattr(st, "rsi", lambda k: 50)(int(n))
def run(st, params: Dict[str,Any]) -> Dict[str,Any]:
    fast = int(params.get("fast", 5)); slow = int(params.get("slow", 30))
    k = _ema(st, fast)*0.3 + _ema(st, slow)*0.7
    r = _rsi(st, 14)
    val = (float(getattr(st,"last_px",0.0)) - k) * (r-50)/50.0
    return {{"value": val, "k": k, "r": r}}
"""
        return name, spec, code

    if reg == "MEAN":
        name = f"obv_pressure_{suffix}"
        spec = {"shape":["o","mid"], "idea":"OBV + short MA as pressure gauge"}
        code = f"""
from typing import Dict, Any
def _obv(st):
    fn = getattr(st, "obv", None)
    return float(fn()) if callable(fn) else 0.0
def _ema(st, n): return getattr(st, "ema", lambda k: st.last_px)(int(n))
def run(st, params: Dict[str,Any]) -> Dict[str,Any]:
    ma = int(params.get("ma", 10))
    o = _obv(st); mid = _ema(st, ma)
    return {{"o": o, "mid": mid}}
"""
        return name, spec, code

    # UNKNOWN
    name = f"spectral_hint_{suffix}"
    spec = {"shape":["value"], "idea":"vol_norm scaled derivative-ish hint"}
    code = f"""
from typing import Dict, Any
def _ema(st, n): return getattr(st, "ema", lambda k: st.last_px)(int(n))
def _vol(st): return max(0.001, float(getattr(st, "vol_norm", 0.2)))
def run(st, params: Dict[str,Any]) -> Dict[str,Any]:
    p = int(params.get("p", 7))
    v = _vol(st)
    e1 = _ema(st, p); e2 = _ema(st, max(2, p//2))
    val = (e2 - e1) * v
    return {{"value": val}}
"""
    return name, spec, code

# ---------------------
# DIAG → BLOK ÖNERİSİ
# ---------------------
def propose_block_from_diag(diag: Dict[str,Any]|None) -> Tuple[str, Dict[str,Any], str]:
    return invent_block(diag)
