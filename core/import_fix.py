# -*- coding: utf-8 -*-
from __future__ import annotations
import re, os
from pathlib import Path
from typing import Optional, Dict, List, Tuple

PROJECT_ROOT_HINTS = ("core", "live", "model", "scripts", "proje1", "proj", "src")

_MISSING_PATTERNS = [
    re.compile(r"No module named '([^']+)'"),
    re.compile(r"cannot import name '([^']+)'"),
    re.compile(r"Unresolved reference: ([A-Za-z_][A-Za-z0-9_\.]*)"),
]

def _find_project_root(start: Path) -> Path:
    p = start.resolve()
    while p.parent != p:
        # "package kökü" varsayımları
        for hint in PROJECT_ROOT_HINTS:
            if (p / hint).exists():
                return p
        if (p / ".git").exists():
            return p
        p = p.parent
    return start.resolve()

def _scan_candidates(root: Path, mod_name: str) -> List[Path]:
    """ telegram_utils  ->  [*/telegram_utils.py, */telegram_utils/__init__.py] """
    names = []
    if "." in mod_name:
        # foo.bar  ->  sondaki gerçek dosya ismini ara
        mod_name = mod_name.split(".")[-1]
    for base in PROJECT_ROOT_HINTS:
        b = root / base
        if not b.exists():
            continue
        for p in b.rglob("*.py"):
            if p.name == f"{mod_name}.py" or (p.name == "__init__.py" and p.parent.name == mod_name):
                names.append(p)
    return names

def _dotted_from(root: Path, target: Path) -> str:
    """ /proj/core/telegram_utils.py -> core.telegram_utils """
    rel = target.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)

def _backup_file(path: Path) -> Path:
    bk = path.with_suffix(path.suffix + ".autofix.bak")
    try:
        if not bk.exists():
            bk.write_bytes(path.read_bytes())
    except Exception:
        pass
    return bk

def _rewrite_import_lines(code: str, old_base: str, new_base: str) -> Tuple[str, int]:
    """
    from telegram_utils import tg_send   -> from Proje1.core.telegram_utils import tg_send
    import telegram_utils                -> from core import telegram_utils as telegram_utils
    """
    count = 0

    # case-1: from X import Y
    pat1 = re.compile(rf"(^\s*from\s+){re.escape(old_base)}(\s+import\s+)", re.M)
    code, n1 = pat1.subn(rf"\1{new_base}\2", code)
    count += n1

    # case-2: import X  (tam eşleşme)
    def _repl_import(m):
        nonlocal count
        count += 1
        name = m.group(2)
        alias = m.group(3) or ""
        # import telegram_utils        -> from core import telegram_utils as telegram_utils
        return f"from {new_base.rsplit('.',1)[0]} import {new_base.rsplit('.',1)[-1]} as {name}{alias}"

    pat2 = re.compile(rf"(^\s*import\s+)({re.escape(old_base)})(\s+as\s+\w+)?\s*$", re.M)
    code = pat2.sub(_repl_import, code)

    return code, count

def guess_missing_name(error_text: str) -> Optional[str]:
    for pat in _MISSING_PATTERNS:
        m = pat.search(error_text)
        if m:
            return m.group(1)
    return None

def auto_fix_imports(
    file_path: str | Path,
    error_text: str,
    prefer_packages: Tuple[str, ...] = ("core", "live"),
) -> Dict[str, object]:
    """
    Bir dosyada ImportError/Unresolved reference görüldüğünde çağır.
    error_text'ten eksik modül adını çeker, projede doğru yolunu bulur,
    ilgili import satırlarını yeniden yazar.
    """
    file_p = Path(file_path)
    if not file_p.exists():
        return {"ok": False, "reason": "file_not_found", "file": str(file_p)}

    missing = guess_missing_name(error_text) or ""
    if not missing:
        return {"ok": False, "reason": "missing_name_not_parsed"}

    root = _find_project_root(file_p.parent)
    cands = _scan_candidates(root, missing)
    if not cands:
        return {"ok": False, "reason": "module_not_found_in_project", "missing": missing}

    # tercih: core/.. > live/.. > diğerleri
    ranked = sorted(
        cands,
        key=lambda p: (0 if p.parts[ p.parts.index(root.name)+1 ] in prefer_packages else 1, len(str(p)))
        if root.name in p.parts else (1, len(str(p)))
    )
    target = ranked[0]
    new_base = _dotted_from(root, target)            # core.telegram_utils
    old_base = missing.split(".")[0]                 # telegram_utils

    code = file_p.read_text(encoding="utf-8", errors="ignore")
    new_code, changed = _rewrite_import_lines(code, old_base, new_base)

    if changed <= 0:
        # Belki import satırı "from Proje1.core.telegram_utils import ..." şeklinde?
        # Yine de üstüne yazmayalım, bilgi dönelim.
        return {
            "ok": False,
            "reason": "no_import_line_matched",
            "missing": missing,
            "suggested_import": new_base,
        }

    _backup_file(file_p)
    file_p.write_text(new_code, encoding="utf-8")

    return {
        "ok": True,
        "fixed": changed,
        "file": str(file_p),
        "missing": missing,
        "resolved_to": new_base,
        "backup": str(file_p.with_suffix(file_p.suffix + ".autofix.bak")),
    }
