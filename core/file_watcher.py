# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, json, re, fnmatch, traceback, threading, importlib, importlib.util, hashlib, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# --- TG fallback
try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, parse_mode="HTML", **k): pass

# --- küçük yardımcılar
def _now_ts() -> int: return int(time.time())
def _read_text(p: Path) -> str:
    try: return p.read_text(encoding="utf-8", errors="ignore")
    except Exception: return ""
def _write_text(p: Path, s: str) -> bool:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s, encoding="utf-8")
        return True
    except Exception:
        traceback.print_exc()
        return False
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# ---- notify dedup
_last_notify: Dict[str, float] = {}
def _notify_once(key: str, text: str, ttl=120):
    now = time.time()
    if key in _last_notify and (_last_notify[key] + ttl > now):
        return
    _last_notify[key] = now
    try: tg_send(text, parse_mode="HTML")
    except Exception: pass

# ---- journal append
def _journal_append(path: Path, row: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass

# =========================
#   DIAGNOSE & PATCH API
# =========================
def diagnose(path: Path, err_msg: str | None, flags: List[str] | None = None,
             journal_path: Optional[Path] = None, notify_tag: str = "diagnose") -> dict:
    flags = flags or []
    info = {"file": str(path), "flags": flags, "err": err_msg or "", "ts": _now_ts()}
    if journal_path:
        _journal_append(journal_path, {"type": "diagnose", **info})
    key = f"{notify_tag}:{path.name}:{','.join(flags)}"
    human = (err_msg or "").replace("`","´")
    _notify_once(key, f"🛠️ *{notify_tag}*: <code>{path.name}</code> • flags={flags} • err=<code>{human[:220]}</code>")
    return info

def attempt_patch(path: Path, err_msg: str | None, backup_dir: Path | None = None,
                  journal_path: Optional[Path] = None) -> bool:
    """
    Basit otomatik yamalar:
      - Eksik import’lar (Path, time, json gibi)
      - reason_tag upper() guard
    """
    txt = _read_text(path)
    if not txt:
        return False
    orig = txt
    changed = False

    emsg = (err_msg or "")

    # 1) NameError common imports
    imports_to_fix = []
    if "name 'Path' is not defined" in emsg and "from pathlib import Path" not in txt:
        imports_to_fix.append("from pathlib import Path")
    if "name 'time' is not defined" in emsg and "import time" not in txt:
        imports_to_fix.append("import time")
    if "name 'json' is not defined" in emsg and "import json" not in txt:
        imports_to_fix.append("import json")
    if imports_to_fix:
        txt = "\n".join(imports_to_fix) + "\n" + txt
        changed = True

    # 2) .upper() None/float guard
    if "object has no attribute 'upper'" in emsg and "def reason_tag_safe" not in txt:
        txt += """

# --- AutoPatch injected: reason_tag_safe ---
def reason_tag_safe(reason):
    m = {
        "DIP": "🟣 DIP","PRED": "🔮 PRED","NEWS": "🚨 NEWS","ORDERBOOK": "🧱 OB",
        "TP": "✅ TP","TP/DECAY": "⏳ TP/DECAY","SL": "🛑 SL","TIME_SL": "⏱️ TIME-SL",
        "BREAKEVEN": "⚖️ BE","BREAKEVEN/MINPROFIT": "⚖️ BE",
    }
    try:
        key = str(reason).upper()
        return m.get(key, str(reason))
    except Exception:
        return str(reason)
"""
        changed = True

    if not changed:
        return False

    try:
        if backup_dir:
            backup_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            (backup_dir / f"{path.name}.{ts}.bak").write_text(orig, encoding="utf-8")
        ok = _write_text(path, txt)
        if ok:
            if journal_path:
                _journal_append(journal_path, {"type": "patch", "file": str(path), "ts": _now_ts(), "sha": _sha1(txt)})
            _notify_once(f"patch:{path.name}", f"✅ *AutoPatch*: <code>{path.name}</code> güncellendi.")
        else:
            _notify_once(f"patchfail:{path.name}", f"❌ *AutoPatch fail*: <code>{path.name}</code> yazılamadı.")
        return ok
    except Exception:
        traceback.print_exc()
        _notify_once(f"patchfail:{path.name}", f"❌ *AutoPatch fail*: <code>{path.name}</code> beklenmeyen hata.")
        return False

# =========================
#  IMPORT TEST (paketli)
# =========================

# Proje kökleri
PKG_ROOT = Path(__file__).resolve().parents[1]           # .../Proje1
PKG_NAME = "Proje1"
KNOW_DIR = PKG_ROOT / "knowledge"
KNOW_PKG = "Proje1.knowledge"

def _dotted_for(py_path: Path) -> Optional[str]:
    """
    Dosya yolundan doğru dotted modül adını türet.
    Ör:  Proje1/knowledge/utils.py    -> Proje1.knowledge.utils
         Proje1/strategies/foo.py     -> Proje1.strategies.foo
         Proje1/knowledge/__init__.py -> Proje1.knowledge
    """
    try:
        rp = py_path.resolve()
        root = PKG_ROOT.resolve()
        if not rp.is_file():
            return None
        if root in rp.parents or rp == root:
            rel = str(rp.relative_to(root)).replace("\\", "/")
            if rel.endswith("__init__.py"):
                rel = rel[: -len("__init__.py")].rstrip("/").replace("/", ".")
                return f"{PKG_NAME}.{rel}" if rel else PKG_NAME
            if rel.endswith(".py"):
                rel = rel[:-3].replace("/", ".")
                return f"{PKG_NAME}.{rel}"
    except Exception:
        pass
    return None

def _import_ok(py_path: Path) -> Tuple[bool, str]:
    """
    1) Paket adı türetilebiliyorsa importlib.import_module(dotted) ile yükle.
    2) Olmuyorsa fallback: spec_from_file_location + __package__ düzeltmesi.
       - knowledge altında ise __package__ = 'Proje1.knowledge'
       - Proje1 altında ise üst pakete göre set edilir
    """
    try:
        dotted = _dotted_for(py_path)
        importlib.invalidate_caches()
        if dotted:
            # Paket bağlamında yükle / yeniden yükle
            if dotted in sys.modules:
                importlib.reload(sys.modules[dotted])
            else:
                importlib.import_module(dotted)
            return True, ""
        # Fallback (paketsiz)
        spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
        if not spec or not spec.loader:
            return False, "no spec"
        mod = importlib.util.module_from_spec(spec)
        # paket bağlamını elle kur (relative importlar kırılmasın)
        if KNOW_DIR in py_path.parents:
            mod.__package__ = KNOW_PKG
        elif PKG_ROOT in py_path.parents:
            # Proje1/alt/yol için en yakın paket kökünü bul
            rel = py_path.resolve().relative_to(PKG_ROOT.resolve())
            parts = rel.parts[:-1]
            pkg = PKG_NAME + "." + ".".join(parts) if parts else PKG_NAME
            mod.__package__ = pkg
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return True, ""
    except Exception as e:
        return False, f"{e}"

# ====================
#  QUALITY GATE HELP
# ====================
def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def _must_contain_all(txt: str, needles: List[str]) -> bool:
    return all(n in txt for n in needles)

# =========================
#  AUTO-ADAPTER / IMPORT
# =========================
def _safe_patch_bot_import(module_name: str, project_root: Path, journal: Optional[Path]) -> bool:
    bot_path = project_root / "core" / "bot.py"
    if not bot_path.exists():
        return False
    code = _read_text(bot_path)
    if not code:
        return False

    import_line = f"from .{module_name} import *  # [autogen]"
    if import_line in code:
        return True  # zaten var

    marker = "# === FILE WATCHER AUTOGEN IMPORTS (do not remove) ==="
    if marker in code:
        new_code = code.replace(marker, marker + "\n" + import_line)
    else:
        m = re.search(r"(?s)^from\s+\.__future__.*?\n", code)
        if m:
            idx = m.end()
            new_code = code[:idx] + import_line + "\n" + code[idx:]
        else:
            new_code = import_line + "\n" + code

    ok = _write_text(bot_path, new_code)
    if ok and journal:
        _journal_append(journal, {"type": "auto_adapter", "bot_import_added": module_name, "ts": _now_ts()})
    if ok:
        _notify_once(f"autoimp:{module_name}", f"🧩 *Auto-Adapter*: <code>core/bot.py</code> içine import eklendi → <code>{module_name}</code>")
    return ok

# ======================
#  FILE WATCHER CLASS
# ======================
class FileWatcher:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        fw = (self.cfg.get("file_watcher") or {})
        self.enabled = bool(fw.get("enabled", True))
        self.scan_interval = int(fw.get("scan_interval_sec", 15))

        actions = fw.get("actions") or {}
        self.on_stale = actions.get("on_stale", "patch")
        self.on_broken = actions.get("on_broken", "disable")

        ap = actions.get("auto_patch") or {}
        self.ap_enabled = bool(ap.get("enabled", True))
        self.ap_backup_dir = Path(ap.get("backup_dir", ".patch_backups"))
        self.ap_max_backups = int(ap.get("max_backups", 20))

        adis = actions.get("auto_disable") or {}
        self.auto_disable = bool(adis.get("enabled", True))
        self.auto_disable_tag = bool(adis.get("write_tag_comment", True))

        aada = actions.get("auto_adapter") or {}
        self.auto_adapter = bool(aada.get("enabled", True))
        self.adapter_dir = Path(aada.get("dir", "live/adapters"))

        notify = fw.get("notify") or {}
        self.notify_mode = (notify.get("mode") or "hybrid")
        self.notify_tg = bool(notify.get("telegram", True))
        self.digest_time_utc = notify.get("digest_time_utc", "20:00")
        self.journal_path = Path(notify.get("journal_path", "logs/filewatch_journal.jsonl"))
        self.rate_limit_per_min = int(notify.get("rate_limit_per_min", 8))
        if not self.journal_path and notify.get("log_path"):
            self.journal_path = Path(notify.get("log_path"))

        self.scan_dirs = [Path(p) for p in (fw.get("scan_dirs") or ["core","live","model","scripts","Proje1"])]
        self.include_globs = fw.get("include_globs") or ["**/*.py", "**/*.json", "**/*.yaml", "**/*.yml"]
        self.exclude_globs = fw.get("exclude_globs") or ["**/__pycache__/**", "**/.patch_backups/**", "**/.git/**"]

        q = fw.get("quality_rules") or {}
        self.min_lines = q.get("min_lines") or {}
        self.must_contain = q.get("must_contain") or {}

        self.state_path = Path("runtime/filewatch_state.json")
        self.state: Dict[str, Any] = self._load_state()

        self.root = Path(os.getcwd())

    def _load_state(self) -> Dict[str, Any]:
        try:
            if self.state_path.exists():
                return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {"files": {}}

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _rel(self, p: Path) -> str:
        try: return str(p.relative_to(self.root))
        except Exception: return str(p)

    def _match_exclude(self, rel: str) -> bool:
        for pat in self.exclude_globs:
            if fnmatch.fnmatch(rel, pat):
                return True
        return False

    def _iter_files(self) -> List[Path]:
        out: List[Path] = []
        for d in self.scan_dirs:
            if not d.exists():
                continue
            for g in self.include_globs:
                for p in d.rglob(g):
                    rel = self._rel(p)
                    if self._match_exclude(rel):
                        continue
                    out.append(p)
        return out

    # ------- kalite/health kontrolleri ----------
    def _quality_gate(self, p: Path) -> None:
        rel = self._rel(p)
        if isinstance(self.min_lines, dict) and rel in self.min_lines:
            try:
                need = int(self.min_lines[rel])
            except Exception:
                need = 0
            have = _count_lines(p)
            if need and have < need:
                diagnose(p, f"short_file:{have}", flags=["low_quality"], journal_path=self.journal_path)
                if self.ap_enabled:
                    attempt_patch(p, "low_quality", backup_dir=self.ap_backup_dir, journal_path=self.journal_path)

        if isinstance(self.must_contain, dict) and rel in self.must_contain:
            needles = self.must_contain[rel] or []
            txt = _read_text(p)
            if not _must_contain_all(txt, needles):
                diagnose(p, "missing_required_snippets", flags=["low_quality"], journal_path=self.journal_path)

    def _try_import(self, p: Path) -> None:
        if p.suffix != ".py":
            return
        ok, err = _import_ok(p)   # <<< PAKETLİ IMPORT TEST
        if ok:
            return
        diagnose(p, err, flags=["broken_import"], journal_path=self.journal_path)
        if self.ap_enabled:
            if attempt_patch(p, err, backup_dir=self.ap_backup_dir, journal_path=self.journal_path):
                ok2, err2 = _import_ok(p)
                if not ok2:
                    diagnose(p, err2, flags=["patch_applied_but_still_broken"], journal_path=self.journal_path)
            else:
                diagnose(p, err, flags=["no_patch"], journal_path=self.journal_path)

    def _auto_adapter_if_needed(self, p: Path) -> None:
        if not self.auto_adapter or p.suffix != ".py":
            return
        name = p.stem
        if name in ("__init__", "bot"):
            return
        rel = self._rel(p)
        if not (rel.startswith("core/") or rel.startswith("live/")):
            return
        _safe_patch_bot_import(name, self.root, self.journal_path)

    # ---------------- ana loop ----------------
    def tick(self) -> None:
        files = self._iter_files()
        state_files = self.state.get("files", {})
        for p in files:
            rel = self._rel(p)
            try:
                st = p.stat()
                mt = int(st.st_mtime)
                key = state_files.get(rel) or {}
                last_mt = int(key.get("mt", 0)) if isinstance(key.get("mt", 0), (int, float)) else 0

                if mt != last_mt:
                    self._quality_gate(p)
                    self._try_import(p)
                    self._auto_adapter_if_needed(p)

                    txt = _read_text(p)
                    state_files[rel] = {"mt": mt, "sha": _sha1(txt)}
            except Exception as e:
                diagnose(p, f"{e}", flags=["scan_error"], journal_path=self.journal_path)

        # silinenleri temizle
        for rel in list(state_files.keys()):
            if not (self.root / rel).exists():
                del state_files[rel]

        self.state["files"] = state_files
        self._save_state()

    def run_forever(self) -> None:
        if not self.enabled:
            return
        _notify_once("fw:start", "👀 *FileWatcher*: başlatıldı (L9).")
        while True:
            try:
                self.tick()
            except Exception as e:
                diagnose(Path("core/file_watcher.py"), f"{e}", flags=["loop_error"], journal_path=self.journal_path)
            time.sleep(max(1, int(self.scan_interval)))

# ------------- dış API -------------
_watcher_thread: Optional[threading.Thread] = None
def ensure_file_watcher(cfg: dict) -> None:
    global _watcher_thread
    fw = (cfg or {}).get("file_watcher") or {}
    if not bool(fw.get("enabled", True)):
        return
    if _watcher_thread and _watcher_thread.is_alive():
        return
    watcher = FileWatcher(cfg)
    t = threading.Thread(target=watcher.run_forever, name="file_watcher", daemon=True)
    t.start()
    _watcher_thread = t
