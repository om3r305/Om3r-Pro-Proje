# -*- coding: utf-8 -*-
from __future__ import annotations
import importlib, pkgutil, traceback, types, inspect, importlib.util, hashlib, time, fnmatch
from typing import Dict, Any, List, Tuple
from pathlib import Path

# --- Opsiyonel bağımlılıklar ---
try:
    from . import market as market_mod
except Exception:
    market_mod = None

try:
    # Senin gönderdiğin API
    from Proje1.core.strategy_api import StrategyPlugin  # type: ignore
except Exception:
    class StrategyPlugin:  # fallback
        pass

try:
    from Proje1.core.can_trade_now import can_trade_now as _can_trade_now
except Exception:
    _can_trade_now = None

try:
    from Proje1.core.brain_learn import adjust_confidence as _adjust_conf
except Exception:
    _adjust_conf = None

try:
    from Proje1.core.logger_utils import log_event
except Exception:
    def log_event(*args, **kwargs):  # no-op
        pass

# --- Varsayılan yollar / kalıplar ---
PKG = "strategies"
IDEAS_DIR = Path("ideas/strategies")        # *.live.py
DEFAULT_WRITE_DIRS = ["live/strategies", "model/strategies", "core/strategies"]
LIVE_GLOB = "*.live.py"
PY_GLOB = "*.py"

VALID_SLOTS = {"pred", "dip", "news", "ob", "aux"}  # aux eklendi

def _mod_name_for_file(p: Path) -> str:
    h = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:10]
    stem = p.stem.replace("-", "_").replace(".", "_")
    return f"dyn_{stem}_{h}"

def _match_any(path: Path, patterns: List[str]) -> bool:
    s = str(path).replace("\\", "/")
    return any(fnmatch.fnmatch(s, pat) for pat in patterns)

class StrategyLoader:
    """
    - strategies/*.py, ideas/strategies/*.live.py ve L60 yazım dizinleri (cfg.l60.design.guardrails.write_dirs)
    - Hot-reload: mtime değişince yeniden yükler
    - Plugin (StrategyPlugin) ve basit class(update/signal) destekler
    - Modül meta: STRAT_INFO / __meta__  => slot, requires
    - Slot setine 'aux' eklendi (Bot bilinmeyeni zaten 'pred'e düşürüyor)
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self._classes: List[type] = []
        self._instances: Dict[str, Dict[str, Any]] = {}
        self._file_index: Dict[Path, float] = {}
        self._module_by_file: Dict[Path, Any] = {}
        self._class_origin: Dict[type, Path] = {}
        self._last_full_scan = 0.0
        self._scan_interval_sec = 10.0

        self._deny_globs: List[str] = []
        self._allow_write_dirs: List[Path] = []
        self._search_roots: List[Tuple[Path, str]] = []

        self._configure_search_roots()
        self._full_scan(force=True)

    # -------------------- config --------------------
    def _configure_search_roots(self) -> None:
        pkg_root = self._ensure_pkg()
        self._search_roots.append((pkg_root, PY_GLOB))
        self._search_roots.append((IDEAS_DIR, LIVE_GLOB))

        l60 = (self.cfg.get("l60") or {})
        design = (l60.get("design") or {})
        guard = (design.get("guardrails") or {})
        write_dirs = guard.get("write_dirs") or DEFAULT_WRITE_DIRS
        self._deny_globs = list(guard.get("deny_globs") or [])
        for d in write_dirs:
            self._allow_write_dirs.append(Path(d))
            self._search_roots.append((Path(d), PY_GLOB))

    def _ensure_pkg(self) -> Path:
        try:
            pkg = importlib.import_module(PKG)
            return Path(pkg.__file__).parent
        except Exception:
            p = Path(PKG)
            p.mkdir(parents=True, exist_ok=True)
            (p / "__init__.py").write_text("# strategies package\n", encoding="utf-8")
            return p

    # -------------------- sınıf tespit --------------------
    def _is_plugin_class(self, cls: type) -> bool:
        try:
            return issubclass(cls, StrategyPlugin) and cls is not StrategyPlugin
        except Exception:
            return False

    def _is_basic_class(self, cls: type) -> bool:
        return hasattr(cls, "update") and hasattr(cls, "signal") and isinstance(getattr(cls, "signal"), types.FunctionType)

    # -------------------- dosya/import --------------------
    def _iter_candidate_files(self) -> List[Path]:
        files: List[Path] = []
        for root, glob_pat in self._search_roots:
            if not root.exists():
                continue
            for py in sorted(root.rglob(glob_pat)):
                if py.suffix != ".py":
                    continue
                if self._deny_globs and _match_any(py, self._deny_globs):
                    continue
                files.append(py)
        return files

    def _safe_exec_module(self, path: Path):
        try:
            mod_name = _mod_name_for_file(path)
            spec = importlib.util.spec_from_file_location(mod_name, str(path))
            if not spec or not spec.loader:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod
        except Exception:
            traceback.print_exc()
            return None

    def _collect_classes_from_module(self, mod, origin: Path) -> List[type]:
        picked: List[type] = []
        try:
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and (self._is_plugin_class(obj) or self._is_basic_class(obj)):
                    picked.append(obj)
                    self._class_origin[obj] = origin
        except Exception:
            traceback.print_exc()
        return picked

    def _full_scan(self, force: bool = False) -> None:
        now = time.time()
        if (not force) and (now - self._last_full_scan < self._scan_interval_sec):
            return
        self._last_full_scan = now

        changed = False
        seen = set()
        for f in self._iter_candidate_files():
            seen.add(f)
            mtime = f.stat().st_mtime
            if f not in self._file_index or self._file_index[f] != mtime:
                mod = self._safe_exec_module(f)
                if mod:
                    self._module_by_file[f] = mod
                    self._file_index[f] = mtime
                    changed = True

        for f in list(self._file_index.keys()):
            if f not in seen:
                self._file_index.pop(f, None)
                self._module_by_file.pop(f, None)
                changed = True

        if changed:
            self._classes.clear()
            self._class_origin = {}
            for f, mod in self._module_by_file.items():
                self._classes.extend(self._collect_classes_from_module(mod, f))

            if self._instances:
                keep = {getattr(cls, "NAME", getattr(cls, "name", cls.__name__)) for cls in self._classes}
                for sym, d in list(self._instances.items()):
                    for k in list(d.keys()):
                        if k not in keep:
                            d.pop(k, None)
            try:
                log_event("strat_scan", changed=True, files=len(self._file_index), classes=len(self._classes))
            except Exception:
                pass

    # -------------------- meta yardımcıları --------------------
    def _default_slot_for_module(self, mod) -> str:
        try:
            info = getattr(mod, "STRAT_INFO", None) or getattr(mod, "__meta__", None)
            if isinstance(info, dict):
                slot = str(info.get("slot", "")).lower().strip()
                if slot in VALID_SLOTS:
                    return slot
        except Exception:
            pass
        return "pred"

    def _symbol_allowed_by_meta(self, symbol: str, cls: type) -> bool:
        try:
            origin = self._class_origin.get(cls)
            mod = self._module_by_file.get(origin)
            info = getattr(mod, "STRAT_INFO", None) or getattr(mod, "__meta__", None)
            if isinstance(info, dict) and info.get("requires"):
                req = info.get("requires")
                if isinstance(req, (list, tuple, set)) and symbol not in req:
                    return False
        except Exception:
            pass
        return True

    def _slot_for_class(self, cls: type) -> str:
        try:
            origin = self._class_origin.get(cls)
            mod = self._module_by_file.get(origin)
            slot = self._default_slot_for_module(mod)
            s2 = getattr(cls, "SLOT", None)
            if isinstance(s2, str) and s2.lower() in VALID_SLOTS:
                return s2.lower()
            return slot
        except Exception:
            return "pred"

    def _get_instance(self, symbol: str, cls: type):
        name = getattr(cls, "NAME", getattr(cls, "name", cls.__name__))
        per_sym = self._instances.setdefault(symbol, {})
        if name not in per_sym:
            try:
                inst_cfg = (self.cfg.get("strategies") or {}).get(name.lower(), {})
                inst = cls(inst_cfg)
            except Exception:
                inst = cls({})
            per_sym[name] = inst
        return per_sym[name]

    # -------------------- public --------------------
    def available(self) -> List[str]:
        self._full_scan()
        return [getattr(cls, "NAME", getattr(cls, "name", cls.__name__)) for cls in self._classes]

    def run_for_symbol(self, symbol: str, st) -> List[Dict[str, Any]]:
        self._full_scan()  # hot-reload
        out: List[Dict[str, Any]] = []
        price = float(getattr(st, "last_px", 0.0) or 0.0)

        if _can_trade_now is not None:
            try:
                sig = inspect.signature(_can_trade_now)
                ok = _can_trade_now() if len(sig.parameters) == 0 else _can_trade_now(symbol, st, market_mod)
                if not ok:
                    return out
            except Exception:
                traceback.print_exc()

        for cls in self._classes:
            if not self._symbol_allowed_by_meta(symbol, cls):
                continue
            try:
                # --- Plugin sınıfı ---
                if self._is_plugin_class(cls):
                    inst = self._get_instance(symbol, cls)
                    if hasattr(inst, "warmup_symbols"):
                        want = inst.warmup_symbols()
                        if want is not None and symbol not in want:
                            continue
                    sig = inst.on_symbol(symbol, st, market_mod)
                    if not sig or not getattr(sig, "fired", False):
                        continue

                    conf = float(getattr(sig, "confidence", 0.0))
                    reason = str(getattr(sig, "reason", "plugin"))
                    slot = getattr(sig, "slot", None)
                    if not slot:
                        slot = self._slot_for_class(cls)
                    slot = str(slot).lower().strip()
                    if slot not in VALID_SLOTS:
                        slot = "pred"  # güvenli degrade

                    if _adjust_conf is not None:
                        try:
                            cls_name = getattr(cls, "NAME", getattr(cls, "name", cls.__name__))
                            conf = _adjust_conf(self.cfg, symbol, cls_name, reason, conf)
                        except Exception:
                            pass

                    out.append({"slot": slot, "fired": True, "confidence": conf, "reason": reason})
                    continue

                # --- Basit sınıf ---
                inst = self._get_instance(symbol, cls)
                if price > 0:
                    try:
                        inst.update(price)
                    except Exception:
                        pass

                sig_attr = getattr(inst, "signal", None)
                if sig_attr is None:
                    continue
                try:
                    resp = sig_attr() if callable(sig_attr) else sig_attr
                except TypeError:
                    resp = sig_attr
                except Exception:
                    traceback.print_exc()
                    continue

                if not isinstance(resp, dict):
                    continue
                fired = bool(resp.get("buy") or resp.get("fired", False))
                if not fired:
                    continue

                slot = str(resp.get("slot", "")).lower().strip()
                if slot not in VALID_SLOTS:
                    slot = self._slot_for_class(cls)

                conf = float(resp.get("confidence", 0.55 if resp.get("buy") else 0.0))
                reason = str(resp.get("info", resp.get("reason", getattr(cls, "NAME", cls.__name__))))

                if _adjust_conf is not None:
                    try:
                        cls_name = getattr(cls, "NAME", getattr(cls, "name", cls.__name__))
                        conf = _adjust_conf(self.cfg, symbol, cls_name, reason, conf)
                    except Exception:
                        pass

                out.append({"slot": slot, "fired": True, "confidence": conf, "reason": reason})
            except Exception:
                traceback.print_exc()

        return out
