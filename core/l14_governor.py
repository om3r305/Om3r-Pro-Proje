# -*- coding: utf-8 -*-
from __future__ import annotations

# Bu dosya, L14 Governor isimlerini mevcut brain_governor modülüne köprüler.
try:
    from Proje1.core import brain_governor as _gov
except Exception:  # yoksa güvenli no-op
    _gov = None

def ensure_l14(cfg: dict) -> None:
    if _gov and hasattr(_gov, "ensure"):
        try: _gov.ensure(cfg)  # varsa kullan
        except Exception: pass

def l14_tick(cfg: dict, ctx: dict | None = None) -> None:
    if _gov and hasattr(_gov, "tick"):
        try:
            # Önce (cfg, ctx) dene; TypeError gelirse (cfg)
            try: _gov.tick(cfg, ctx)
            except TypeError: _gov.tick(cfg)
        except Exception: pass

def l14_on_exception(where: str, err: Exception, cfg: dict) -> None:
    if _gov and hasattr(_gov, "on_exception"):
        try: _gov.on_exception(where, err, cfg)
        except Exception: pass

def l14_heartbeat() -> None:
    if _gov and hasattr(_gov, "heartbeat"):
        try: _gov.heartbeat()
        except Exception: pass
