# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

NINJA = Path("ninjamappers.json")  # kullanıcı sağladı

def _load_map() -> Dict[str, List[str]]:
    if not NINJA.exists(): return {}
    try:
        return json.loads(NINJA.read_text(encoding="utf-8"))
    except Exception:
        return {}

def cluster_decide(active_syms: List[str], candidate: str, cfg: Dict[str,Any] | None) -> Tuple[bool, str]:
    """
    Aynı kümeye ait coin’den zaten poz varsa veto/azalt.
    cfg.modules.cluster_guard: {mode:"veto|shrink", shrink_mult:0.6}
    """
    cg = (cfg or {}).get("modules", {}).get("cluster_guard", {})
    mode = str(cg.get("mode", "veto")).lower()
    cmap = _load_map()
    cand_group = None
    for g, arr in cmap.items():
        if candidate in arr:
            cand_group = g
            break
    if not cand_group: return False, "ok"
    conflict = any((s in (cmap.get(cand_group) or [])) for s in active_syms)
    if not conflict: return False, "ok"
    if mode == "shrink":
        return True, "shrink"
    return True, "veto"
