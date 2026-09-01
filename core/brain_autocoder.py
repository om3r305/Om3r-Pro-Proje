# -*- coding: utf-8 -*-
from __future__ import annotations
import time, json, re, difflib, hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# --- Sessiz TG & Log fallback (renkli emoji'lerle) ---
try:
    from Proje1.core.auto_creator import ensure_module_skeleton as _ensure_mod
except Exception:
    _ensure_mod = None

try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, **k): pass

try:
    from Proje1.core.logger_utils import log_brain
except Exception:
    def log_brain(kind, data): pass

# --- Autopatch sandbox (opsiyonel) ---
try:
    from Proje1.core.autopatch_sandbox import sandbox_try as _sandbox_try
except Exception:
    _sandbox_try = None

# --- File watcher (opsiyonel) ---
try:
    from Proje1.core.file_watcher import ensure_file_watcher as _ensure_fw
except Exception:
    def _ensure_fw(cfg): pass

# ---------------- Paths ----------------
ROOT            = Path(".")
CANDIDATES_DIR  = Path("runtime/patch_candidates")
PATCH_DIR       = Path("runtime/patches")
FLAGS_PRIMARY   = Path("runtime/flags") / "autocoder_auto_apply.json"
FLAGS_FALLBACK1 = Path("runtime") / "autocoder_auto_apply.json"              # eski olasılık
FLAGS_FALLBACK2 = Path("flags") / "autocoder_auto_apply.json"                # yanlış yere konduysa
FLAGS_FALLBACK3 = Path("Proje1/runtime/flags") / "autocoder_auto_apply.json" # senin dizin yapısı
FORCE_FLAG      = Path("runtime/flags/force_autocoder.json")
JOURNAL         = Path("logs/autocoder.jsonl")

for p in (CANDIDATES_DIR, PATCH_DIR, JOURNAL.parent, FORCE_FLAG.parent):
    p.mkdir(parents=True, exist_ok=True)

# ---------------- Küçük yardımcılar ----------------
def _cfg(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None

def _write_text(path: Path, text: str) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return True
    except Exception:
        return False

def _append_journal(rec: Dict[str, Any]) -> None:
    rec = {"ts": time.time(), **rec}
    try:
        with JOURNAL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _unified_diff(old: str, new: str, rel_path: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(True),
        new.splitlines(True),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
        n=3,
    )
    return "".join(diff)

def _compile_ok(src: str, path_hint: str = "") -> bool:
    try:
        compile(src, path_hint or "<autocoder>", "exec")
        return True
    except Exception:
        return False

# ---------------- Auto-apply flag logic ----------------
def _auto_apply_enabled() -> Tuple[bool, Optional[str]]:
    """
    Auto-apply flag'ini birden fazla olası konumda ara.
    Bulursak (True/False, 'kaynak-yol') döndür.
    """
    candidates = [
        FLAGS_PRIMARY,
        FLAGS_FALLBACK1,
        FLAGS_FALLBACK2,
        FLAGS_FALLBACK3,  # Proje1/... fallback
    ]
    for p in candidates:
        try:
            if p.exists():
                cfg = json.loads(p.read_text(encoding="utf-8"))
                on = bool(cfg.get("enabled", False))
                try:
                    tg_send(f"🧠 AutoCoder • auto-apply bayrağı bulundu • Kaynak: {p} • enabled={on}")
                except Exception:
                    pass
                return on, str(p)
        except Exception:
            continue
    return False, None

# ======================================================
# ===============  HEURISTIC TARA & ÖNER  ==============
# ======================================================
def _scan_smells(path: Path) -> List[Dict[str, Any]]:
    """
    Mini kural seti:
      1) .upper() None-guard
      2) UPPER kıyas -> lower normalize
      3) drift_watch için journal/baseline stub
    """
    smells: List[Dict[str, Any]] = []
    text = _read_text(path)
    if text is None:
        return smells
    rel = str(path)

    # 1) .upper() guard (str(x).upper() ise dokunma)
    if ".upper()" in text and "(or \"\")" not in text and "( or '' )" not in text:
        new = re.sub(
            r"(?<!str\()([A-Za-z0-9_\.]+)\.upper\(\)",
            lambda m: m.group(0) if m.group(1).strip().startswith("str(") else f"({m.group(1)} or \"\").upper()",
            text
        )
        if new != text:
            smells.append({
                "kind":"upper_guard","title":"None-guard for .upper()",
                "rel":rel,"patch":new,"risk":"low","emoji":"🟢"
            })

    # 2) UPPER kıyas normalize → lower() + guard
    if re.search(r"\.upper\(\)\s*==\s*['\"]([A-Z0-9_]+)['\"]", text):
        tmp = text.replace(".upper()", ".lower()")
        tmp = re.sub(
            r"(?<!str\()([A-Za-z0-9_\.]+)\.lower\(\)",
            lambda m: m.group(0) if ("or" in m.group(0)) else f"({m.group(1)} or \"\").lower()",
            tmp
        )
        if tmp != text:
            smells.append({
                "kind":"upper_cmp_normalize","title":"Normalize slot comparisons to lower()",
                "rel":rel,"patch":tmp,"risk":"low","emoji":"🟢"
            })

    # 3) drift_watch journaling stub
    if ("drift_watch" in rel) and (("append_j" not in text and "JOURNAL" not in text) or ("BASELINE" not in text)):
        add: List[str] = []
        if "JOURNAL" not in text:
            add.append('JOURNAL = Path("logs/drift_watch.jsonl")\nJOURNAL.parent.mkdir(parents=True, exist_ok=True)')
        if "BASELINE" not in text:
            add.append('BASELINE = Path("model/drift_baseline.json")\nBASELINE.parent.mkdir(parents=True, exist_ok=True)')
        if "def _append_j(" not in text:
            add.append(
                "def _append_j(rec: Dict[str,Any]) -> None:\n"
                "    rec = {\"ts\": time.time(), **rec}\n"
                "    try:\n"
                "        with JOURNAL.open(\"a\", encoding=\"utf-8\") as f:\n"
                "            f.write(json.dumps(rec, ensure_ascii=False) + \"\\n\")\n"
                "    except Exception:\n"
                "        pass\n"
            )
        smells.append({
            "kind":"drift_journal_stub","title":"Drift journaling & baseline guards",
            "rel":rel,"patch": text + "\n\n# --- AutoCoder: inserted drift journaling guards ---\n" + "\n".join(add) + "\n",
            "risk":"low","emoji":"🟢"
        })
    return smells

# ======================================================
# ===============  PATCH ADAYI ÜRET/TEST  ==============
# ======================================================
def _build_candidate(path: Path, new_text: str) -> Optional[Dict[str, Any]]:
    old = _read_text(path)
    if old is None or old == new_text:
        return None
    if not _compile_ok(new_text, str(path)):
        return None
    diff = _unified_diff(old, new_text, str(path))
    if not diff.strip():
        return None
    cid = f"{int(time.time())}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
    cand = {"id": cid, "file": str(path), "diff": diff, "ts": time.time(), "risk": "low"}
    _write_text(CANDIDATES_DIR / f"{cid}.diff", diff)
    return cand

def _list_python_files() -> List[Path]:
    globs = [
        "core/**/*.py","live/**/*.py","model/**/*.py","scripts/**/*.py","Proje1/**/*.py"
    ]
    skip_parts = {"__pycache__", ".git", "patch_backups", ".patch_backups", "venv", ".venv", "runtime"}
    files: List[Path] = []
    for g in globs:
        for f in ROOT.glob(g):
            if f.suffix != ".py" or not f.is_file():
                continue
            if any(sp in f.parts for sp in skip_parts):
                continue
            files.append(f)
    # uniq
    uniq, seen = [], set()
    for f in files:
        if f not in seen:
            uniq.append(f); seen.add(f)
    return uniq

def propose_patches(cfg: Dict[str,Any]) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []
    for f in _list_python_files():
        for sm in _scan_smells(f):
            cand = _build_candidate(f, sm["patch"])
            if cand:
                cand.update({"title": sm["title"], "kind": sm["kind"], "emoji": sm["emoji"]})
                out.append(cand)
    return out

# ======================================================
# ===================  SANDBOX & UYGULA  ===============
# ======================================================
def _sandbox_test(cfg: Dict[str,Any]) -> bool:
    ok = True
    if _sandbox_try is not None:
        try:
            _sandbox_try(cfg)
        except Exception:
            ok = False
    return ok

def _apply_patch(cand: Dict[str,Any]) -> bool:
    return _write_text(PATCH_DIR / f"{cand['id']}.diff", str(cand["diff"]))

# ======================================================
# ======================  ANA TICK  =====================
# ======================================================
_LAST_TICK = 0.0


def _maybe_autocreate_from_error(err_msg: str, cfg: Dict[str, Any]):
    if not _ensure_mod or not isinstance(err_msg, str):
        return
    if not _cfg(cfg, "auto_create_missing_modules.enabled", False):
        return
    # “No module named 'xxx'” → xxx
    m = re.search(r"No module named ['\"]([A-Za-z_][A-Za-z0-9_\.]+)['\"]", err_msg)
    if m:
        dotted = m.group(1)
        try:
            created = _ensure_mod(dotted, cfg)
            if created:
                try:
                    from Proje1.core.utils_io import tg_send
                    tg_send(f"🧩 Auto-create: <code>{dotted}</code> → <code>{created.as_posix()}</code>", parse_mode="HTML")
                except Exception:
                    pass
        except Exception:
            pass

def brain_tick(cfg: Dict[str,Any], ctx: Dict[str,Any] | None = None) -> None:
    global _LAST_TICK
    tick_sec = int(_cfg(cfg, "autocoder.tick_sec", 600))
    now = time.time()
    if now - _LAST_TICK < max(30, tick_sec):
        return

    _ensure_fw(cfg)  # file-watcher ayakta olsun

    # Force tetik?
    forced = False
    if FORCE_FLAG.exists():
        forced = True
        try: FORCE_FLAG.unlink(missing_ok=True)
        except Exception: pass

    # Aday üret
    cands = propose_patches(cfg)
    if not cands and not forced:
        _LAST_TICK = now
        return

    # Sandbox testi
    sandbox_ok = _sandbox_test(cfg)

    # Auto-apply durumu (her tick’te tekrar oku)
    auto_on, auto_src = _auto_apply_enabled()

    applied = 0
    if sandbox_ok and auto_on:
        for c in cands:
            if _apply_patch(c):
                applied += 1

    # TG raporu
    total = len(cands)
    lines = []
    for it in cands[:8]:
        lines.append(f"{'🟢' if sandbox_ok else '🔴'} {it.get('title','')} — <code>{it['file']}</code>")
    more = "" if total <= 8 else f"\n… ve {total-8} ek aday"

    msg = (
        f"🧠 <b>AutoCoder</b> — aday patch: <b>{total}</b>\n"
        f"⚙️ Auto-apply: <b>{'ON' if auto_on else 'OFF'}</b>"
        f" • Kaynak: <code>{auto_src or 'yok'}</code>"
        f" • Sandbox: {'var' if sandbox_ok else 'yok'}\n"
        + ("\n".join(lines) if lines else "—")
        + more
    )
    try: tg_send(msg, parse_mode="HTML")
    except Exception: pass

    log_brain("autocoder", {
        "forced": forced, "found": total, "applied": applied,
        "auto_apply": auto_on, "auto_src": auto_src, "sandbox_ok": sandbox_ok
    })
    _append_journal({
        "event":"tick","forced":forced,"found":total,"applied":applied,
        "auto_apply":auto_on,"auto_src":auto_src,"sandbox_ok":sandbox_ok
    })

    _LAST_TICK = now
