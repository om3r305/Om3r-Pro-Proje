# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Alias thin-layer: Brain Governor
- Slot/strateji ağırlıklarını ve veto eşiğini, mevcut kolektif/niyet evrim/dinamik tahsis kodlarına delege eder.
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

_weights   = _try_import("Proje1.core.brain_collective")
_intent    = _try_import("Proje1.core.intent_evolver")
_overrides = _try_import("Proje1.core.brain_hook") or _try_import("Proje1.core.strategy_loader")

def rebalance(cfg: Dict[str, Any]) -> None:
    """
    Portföy/slot ağırlıklarını performansa göre güncelle.
    """
    try:
        if _weights and hasattr(_weights, "auto_rebalance"):
            _weights.auto_rebalance(cfg)
        elif _weights and hasattr(_weights, "collective_tick"):
            _weights.collective_tick(cfg)
    except Exception as e:
        try:
            tg_send(f"🟠 <b>BrainGovernor</b> rebalance err: <code>{e}</code>", parse_mode="HTML")
        except Exception:
            pass

def tighten_veto_if_needed(cfg: Dict[str, Any], bump: float = 0.02) -> None:
    """
    Kötüleşen koşullarda veto_conf_min'i nazikçe artır (mevcut override mekanizması üzerinden).
    """
    try:
        if _intent and hasattr(_intent, "tighten_veto"):
            _intent.tighten_veto(cfg, bump=bump)
        elif _overrides and hasattr(_overrides, "brain_overrides"):
            cur = float((cfg.get("brain") or {}).get("veto_conf_min", 0.55))
            _overrides.brain_overrides({"brain.veto_conf_min": round(cur + bump, 3)}, cfg)
            tg_send(f"⚖️ <b>BrainGovernor</b>: veto_conf_min +{bump:.2f}", parse_mode="HTML")
    except Exception as e:
        try:
            tg_send(f"🟠 <b>BrainGovernor</b> tighten err: <code>{e}</code>", parse_mode="HTML")
        except Exception:
            pass
