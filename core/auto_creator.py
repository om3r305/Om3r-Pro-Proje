# -*- coding: utf-8 -*-
from __future__ import annotations
import re, time
from pathlib import Path
from typing import Dict, Any, Optional

def _cfg(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _tmpl(dotted: str, cfg: Dict[str, Any]) -> str:
    t = _cfg(cfg, "auto_create_missing_modules.templates.py_module",
             "# -*- coding: utf-8 -*-\n\"\"\"Auto-created: {dotted}\"\"\"\n")
    return t.format(dotted=dotted, ts=time.strftime("%Y-%m-%d %H:%M:%S"))

def _is_allowed_name(name: str, cfg: Dict[str, Any]) -> bool:
    rx = _cfg(cfg, "auto_create_missing_modules.allowed_names_regex", r"^[A-Za-z_][A-Za-z0-9_]{0,64}$")
    deny = set(_cfg(cfg, "auto_create_missing_modules.deny_names", []))
    if name in deny: return False
    try:
        return re.match(rx, name) is not None
    except Exception:
        return False

def _ensure_pkg_path(p: Path, ensure_init: bool):
    p.mkdir(parents=True, exist_ok=True)
    if ensure_init:
        init = p / "__init__.py"
        if not init.exists():
            init.write_text("# auto-created package\n", encoding="utf-8")

def ensure_module_skeleton(dotted: str, cfg: Dict[str, Any]) -> Optional[Path]:
    """
    Yok modül ismini (dotted) alır, uygun yerde boş bir .py iskeleti oluşturur.
    Dönüş: oluşturulan dosya yolu (veya None).
    """
    if not dotted or not isinstance(dotted, str):
        return None
    parts = dotted.split(".")
    if any(not _is_allowed_name(x, cfg) for x in parts):
        return None

    roots = [Path(r) for r in _cfg(cfg, "auto_create_missing_modules.search_roots", ["Proje1", "live", "scripts"])]
    default_dir = Path(_cfg(cfg, "auto_create_missing_modules.default_dir_for_new", "Proje1/knowledge"))
    ensure_init = bool(_cfg(cfg, "auto_create_missing_modules.ensure_pkg_init", True))

    # 1) Eğer dotted 'Proje1....' ile başlıyorsa kökü koru; değilse default_dir'e yaz
    if parts[0] in [r.name for r in roots]:
        base = Path(parts[0])
        sub_parts = parts[1:]
        pkg_dir = base.joinpath(*sub_parts[:-1]) if sub_parts else base
        _ensure_pkg_path(pkg_dir, ensure_init)
        py = (pkg_dir / f"{(sub_parts[-1] if sub_parts else '__init__')}.py")
    else:
        _ensure_pkg_path(default_dir, ensure_init)
        py = default_dir / f"{parts[-1]}.py"

    if not py.exists():
        # ara paket init'leri
        if py.parent != default_dir:
            cur = py.parent
            while str(cur).strip() and cur.exists():
                if ensure_init:
                    init = cur / "__init__.py"
                    if not init.exists():
                        init.write_text("# auto-created package\n", encoding="utf-8")
                if cur.name in [r.name for r in roots]:
                    break
                cur = cur.parent
        py.write_text(_tmpl(dotted, cfg), encoding="utf-8")
        return py
    return None
