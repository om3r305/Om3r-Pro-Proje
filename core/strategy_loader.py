# -*- coding: utf-8 -*-
from __future__ import annotations
import importlib, pkgutil, traceback, types, inspect, importlib.util, hashlib, time, sys
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import fnmatch

# --- Opsiyonel bağımlılıklar ---
try:
    from . import market as market_mod
except Exception:
    market_mod = None

try:
    from Proje1.core.strategy_api import StrategyPlugin  # opsiyonel
except Exception:
    class StrategyPlugin:  # yoksa dummy
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
    def log_event(*args, **kwargs):  # no-op fallback
        pass

# --- Varsayılan yollar / kalıplar ---
PKG = "Proje1.strategies"
IDEAS_DIR = Path("Proje1/strategies")        # L10 canlı fikir klasörü (yalnız *.live.py)
DEFAULT_WRITE_DIRS = [
    "Proje1/strategies"                      # L60 handoff hedefi
]
LIVE_GLOB = "*.live.py"
PY_GLOB = "*.py"

# -------------------------------------------------------------------
# Yardımcılar
# -------------------------------------------------------------------
def _mod_name_for_file(p: Path) -> str:
    """Paket dışı fallback için çakışmasız modül adı üret."""
    h = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:10]
    stem = p.stem.replace("-", "_").replace(".", "_")
    return f"dyn_{stem}_{h}"

def _match_any(path: Path, patterns: List[str]) -> bool:
    s = str(path).replace("\\", "/")
    for pat in patterns:
        if fnmatch.fnmatch(s, pat):
            return True
    return False

def _has_init(path: Path) -> bool:
    return (path / "__init__.py").exists()

def _norm_parts(base: Path, p: Path) -> List[str]:
    rel = str(p.resolve().relative_to(base.resolve())).replace("\\", "/")
    parts = rel.split("/")
    return parts

# -------------------------------------------------------------------
# StrategyLoader (L14–L60 uyumlu)
# -------------------------------------------------------------------
class StrategyLoader:
    """
    - Klasik paket: strategies/*.py (paket-ismiyle import)
    - L10/L60 canlı üretimler: Proje1/strategies altı (*.py / *.live.py)
    - Hot-reload: mtime değişince yeniden import/reload
    - İki tip strateji: Basic (update/signal) & Plugin (StrategyPlugin)
    - STRAT_INFO / __meta__ desteği
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}

        # Keşfedilmiş sınıflar ve instance cache
        self._classes: List[type] = []
        self._instances: Dict[str, Dict[str, Any]] = {}  # per-symbol -> {class_name: instance}

        # Dosya izleme (hot-reload)
        self._file_index: Dict[Path, float] = {}         # path -> last_mtime
        self._module_by_file: Dict[Path, Any] = {}       # path -> module obj
        self._module_name_by_file: Dict[Path, str] = {}  # path -> dotted or dyn name
        self._class_origin: Dict[type, Path] = {}        # class -> origin file
        self._last_full_scan = 0.0
        self._scan_interval_sec = 10.0  # sık ama hafif

        # Guardrails (L14/L60)
        self._deny_globs: List[str] = []
        self._allow_write_dirs: List[Path] = []

        # Hangi dizinler taranacak?
        self._search_roots: List[Tuple[Path, str]] = []  # (root, glob)
        self._pkg_root: Path = self._ensure_pkg()        # Proje1/strategies paket kökü
        self._configure_search_roots()

        # İlk tarama
        self._full_scan(force=True)

    # -------------------- Konfigürasyon --------------------
    def _configure_search_roots(self) -> None:
        # 1) strategies paketi (her zaman)
        self._search_roots.append((self._pkg_root, PY_GLOB))

        # 2) L10: ideas/strategies/*.live.py  (bizde zaten Proje1/strategies altında)
        self._search_roots.append((IDEAS_DIR, LIVE_GLOB))

        # 3) L60: write_dirs (cfg’den oku) + varsayılanlar
        l60 = (self.cfg.get("l60") or {})
        design = (l60.get("design") or {})
        guard = (design.get("guardrails") or {})
        write_dirs = guard.get("write_dirs") or []
        if not write_dirs:
            write_dirs = DEFAULT_WRITE_DIRS

        self._deny_globs = list(guard.get("deny_globs") or [])
        for d in write_dirs:
            self._allow_write_dirs.append(Path(d))

        # Arama köklerine ekle (sadece var olanları tara)
        for d in write_dirs:
            self._search_roots.append((Path(d), PY_GLOB))

    def _ensure_pkg(self) -> Path:
        """`Proje1.strategies` paketinin varlığını garanti eder ve dizinini döndürür."""
        try:
            pkg = importlib.import_module(PKG)
            return Path(pkg.__file__).parent
        except Exception:
            p = Path("Proje1/strategies")
            p.mkdir(parents=True, exist_ok=True)
            (p / "__init__.py").write_text("# strategies package\n", encoding="utf-8")
            # importlanabilir hale getir
            importlib.invalidate_caches()
            importlib.import_module(PKG)
            return p

    # -------------------- Sınıf tür kontrolleri --------------------
    def _is_plugin_class(self, cls: type) -> bool:
        try:
            return issubclass(cls, StrategyPlugin) and cls is not StrategyPlugin
        except Exception:
            return False

    def _is_basic_class(self, cls: type) -> bool:
        return hasattr(cls, "update") and hasattr(cls, "signal") and isinstance(getattr(cls, "signal"), types.FunctionType)

    # -------------------- Dosya tarama / import --------------------
    def _iter_candidate_files(self) -> List[Path]:
        files: List[Path] = []
        seen: set[Path] = set()
        for root, glob_pat in self._search_roots:
            if not root.exists():
                continue
            for py in sorted(root.rglob(glob_pat)):
                if py in seen:
                    continue
                # deny globs
                if self._deny_globs and _match_any(py, self._deny_globs):
                    continue
                if py.suffix != ".py":
                    continue
                seen.add(py)
                files.append(py)
        return files

    def _dotted_from_path(self, path: Path) -> Optional[str]:
        """
        Mümkünse paket adı üret.
        Sadece Proje1/strategies kökünün altı için: Proje1.strategies.<alt.yol>
        """
        try:
            if path.is_relative_to(self._pkg_root):
                parts = _norm_parts(self._pkg_root, path)  # e.g. ["foo", "bar.live.py"]
                if parts and parts[-1] == "__init__.py":
                    parts = parts[:-1]
                else:
                    parts[-1] = path.stem
                if parts:
                    return "Proje1.strategies." + ".".join(parts)
                return "Proje1.strategies"
        except Exception:
            pass
        return None

    def _exec_as_dyn(self, path: Path) -> Tuple[Optional[types.ModuleType], Optional[str]]:
        """
        Paket adı türetilemezse dosyayı dinamik adla yükler.
        __package__'ı 'Proje1.strategies' olarak set ederiz ki relative importlar kırılmasın.
        """
        try:
            mod_name = _mod_name_for_file(path)
            spec = importlib.util.spec_from_file_location(mod_name, str(path))
            if not spec or not spec.loader:
                return None, None
            mod = importlib.util.module_from_spec(spec)
            # relative importlar için bağlam
            mod.__package__ = PKG
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod, mod_name
        except Exception:
            traceback.print_exc()
            return None, None

    def _load_module_for_file(self, path: Path, mtime: float):
        """
        Öncelik: paket-ismi → importlib.import_module / reload
        Fallback: dinamik isim → exec
        """
        dotted = self._dotted_from_path(path)
        if dotted:
            try:
                importlib.invalidate_caches()
                if dotted in sys.modules:
                    # reload
                    mod = importlib.reload(sys.modules[dotted])
                else:
                    mod = importlib.import_module(dotted)
                self._module_by_file[path] = mod
                self._module_name_by_file[path] = dotted
                self._file_index[path] = mtime
                return
            except Exception:
                traceback.print_exc()
                # paket importu başaramadı → fallback'e düş
        # fallback
        mod, dyn_name = self._exec_as_dyn(path)
        if mod:
            self._module_by_file[path] = mod
            self._module_name_by_file[path] = dyn_name or ""
            self._file_index[path] = mtime

    def _collect_classes_from_module(self, mod, origin: Path) -> List[type]:
        picked: List[type] = []
        try:
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type):
                    if self._is_plugin_class(obj) or self._is_basic_class(obj):
                        picked.append(obj)
                        self._class_origin[obj] = origin
        except Exception:
            traceback.print_exc()
        return picked

    def _full_scan(self, force: bool = False) -> None:
        """Tüm arama köklerini tara, değişen dosyaları yükle ve sınıfları yenile."""
        now = time.time()
        if (not force) and (now - self._last_full_scan < self._scan_interval_sec):
            return
        self._last_full_scan = now

        changed = False
        seen_files = set()
        for f in self._iter_candidate_files():
            seen_files.add(f)
            try:
                mtime = f.stat().st_mtime
            except FileNotFoundError:
                continue
            if f not in self._file_index or self._file_index[f] != mtime:
                self._load_module_for_file(f, mtime)
                changed = True

        # Silinenleri temizle
        for f in list(self._file_index.keys()):
            if f not in seen_files:
                self._file_index.pop(f, None)
                self._module_by_file.pop(f, None)
                self._module_name_by_file.pop(f, None)
                changed = True

        if changed:
            # sınıf listesini yeniden kur
            self._classes.clear()
            self._class_origin = {}
            for f, mod in self._module_by_file.items():
                cls_list = self._collect_classes_from_module(mod, f)
                self._classes.extend(cls_list)

            # instance cache’i kısmi temizle (yalnız değişen dosyalara ait sınıfları)
            if self._instances:
                purge_names = set()
                for cls in self._classes:
                    purge_names.add(getattr(cls, "NAME", getattr(cls, "name", cls.__name__)))
                for sym, d in list(self._instances.items()):
                    for k in list(d.keys()):
                        if k not in purge_names:
                            d.pop(k, None)

            try:
                log_event("strat_scan", changed=True, files=len(self._file_index), classes=len(self._classes))
            except Exception:
                pass

    # -------------------- Public API --------------------
    def available(self) -> List[str]:
        self._full_scan()
        return [getattr(cls, "NAME", getattr(cls, "name", cls.__name__)) for cls in self._classes]

    def _default_slot_for_module(self, mod) -> str:
        """Modül düzeyinde STRAT_INFO/__meta__ varsa slot’a öncelik ver."""
        try:
            info = getattr(mod, "STRAT_INFO", None) or getattr(mod, "__meta__", None)
            if isinstance(info, dict):
                slot = str(info.get("slot", "")).lower().strip()
                if slot in ("pred", "dip", "news", "ob"):
                    return slot
        except Exception:
            pass
        return "pred"

    def _get_instance(self, symbol: str, cls: type):
        name = getattr(cls, "NAME", getattr(cls, "name", cls.__name__))
        per_sym = self._instances.setdefault(symbol, {})
        if name not in per_sym:
            try:
                # class’a özel config: strategies bölümündeki alt-anahtarla eşleşirse ver
                inst_cfg = (self.cfg.get("strategies") or {}).get(name.lower(), {})
                inst = cls(inst_cfg)
            except Exception:
                inst = cls({})
            per_sym[name] = inst
        return per_sym[name]

    def _symbol_allowed_by_meta(self, symbol: str, cls: type) -> bool:
        """Sınıf modülündeki STRAT_INFO/__meta__.requires varsa sembolü filtrele."""
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
        """Öncelik: modül meta slot, sonra sınıf SLOT/NAME, yoksa 'pred'."""
        try:
            origin = self._class_origin.get(cls)
            mod = self._module_by_file.get(origin)
            slot = self._default_slot_for_module(mod)
            s2 = getattr(cls, "SLOT", None)
            if isinstance(s2, str) and s2.lower() in ("pred", "dip", "news", "ob"):
                return s2.lower()
            return slot
        except Exception:
            return "pred"

    def run_for_symbol(self, symbol: str, st) -> List[Dict[str, Any]]:
        """
        Her sembol için stratejileri çalıştır; normalize sinyaller döndür.
        - Hot-reload (mtime tabanlı, 10 sn)
        - STRAT_INFO / __meta__ ile slot & requires
        - can_trade_now global guard
        - adjust_confidence ile meta-öğrenme
        """
        # Hot-reload
        self._full_scan()

        out: List[Dict[str, Any]] = []
        price = float(getattr(st, "last_px", 0.0) or 0.0)

        # Global trade guard (varsa)
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
                # 1) Plugin sınıfı
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

                    if _adjust_conf is not None:
                        try:
                            cls_name = getattr(cls, "NAME", getattr(cls, "name", cls.__name__))
                            conf = _adjust_conf(self.cfg, symbol, cls_name, reason, conf)
                        except Exception:
                            pass

                    out.append({"slot": slot, "fired": True, "confidence": conf, "reason": reason})
                    continue

                # 2) Basit sınıf
                inst = self._get_instance(symbol, cls)
                if price > 0:
                    try:
                        inst.update(float(price))
                    except Exception:
                        pass

                sig_attr = getattr(inst, "signal", None)
                if sig_attr is None:
                    continue
                try:
                    resp = sig_attr() if callable(sig_attr) else sig_attr
                except TypeError:
                    resp = sig_attr  # property ise
                except Exception:
                    traceback.print_exc()
                    continue

                if not isinstance(resp, dict):
                    continue

                fired = bool(resp.get("buy") or resp.get("fired", False))
                if not fired:
                    continue

                # slot çıkarımı
                slot = str(resp.get("slot", "")).lower().strip()
                if slot not in ("pred", "dip", "news", "ob"):
                    slot = self._slot_for_class(cls)

                # güven & neden
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
