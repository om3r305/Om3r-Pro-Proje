# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Alias thin-layer: Drift Detector
- Var olan drift izleme/koruma mekanizmasını tek yerden expose eder.
"""

from typing import Any, Dict

try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, parse_mode: str = "HTML", **k):  # type: ignore
        return

def _try_import(path: str):
    try:
        import importlib
        return importlib.import_module(path)
    except Exception:
        return None

_drift = _try_import("Proje1.core.drift_watch") or _try_import("Proje1.core.brain_selfheal")

def tick(cfg: Dict[str, Any]) -> None:
    """
    Tek adım drift kontrolü. Mevcut drift modülü varsa çağırır; yoksa no-op.
    """
    try:
        if _drift and hasattr(_drift, "drift_tick"):
            _drift.drift_tick(cfg)
        elif _drift and hasattr(_drift, "ensure_selfheal_watcher"):
            # en azından health izleyici aktif
            _drift.ensure_selfheal_watcher(cfg)
    except Exception as e:
        try:
            tg_send(f"🟠 <b>DriftDetector</b> err: <code>{e}</code>", parse_mode="HTML")
        except Exception:
            pass
