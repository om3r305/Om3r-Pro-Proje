# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Alias thin-layer: Online Learner
- Var olan öğrenme/adaptasyon mekaniklerine köprü:
  * learning.autopatch  (cfg tabanlı parametre iyileştirme)
  * brain.auto_weights  (slot/entry ağırlıkları)
  * dyn_alloc           (portföy paylarını dinamik ayarlama)
  * (ops) intent_evolver / evo_runner
- Dış dünya bu modülü çağırdığında, içerideki mevcut sistemleri tetikliyor.
"""

import time, traceback
from typing import Any, Dict, Optional

# TG fallback
try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, parse_mode: str = "HTML", **k):  # type: ignore
        return

# Güvenli import yardımcıları
def _try_import(path: str):
    try:
        import importlib
        return importlib.import_module(path)
    except Exception:
        return None

# Mevcut parçalar (opsiyonel)
_autopatch = _try_import("Proje1.core.intent_evolver") or _try_import("Proje1.core.auto_merge_gate")
_evo       = _try_import("Proje1.core.evo_runner")
_weights   = _try_import("Proje1.core.brain_collective")
_dynalloc  = _try_import("Proje1.core.intent_evolver") or _try_import("Proje1.core.brain_collective")

_last_tick = 0.0

def tick_intra(cfg: Dict[str, Any]) -> None:
    """
    Gün içi mini öğrenme tetikleyicisi.
    - learning.autopatch / auto_weights / dyn_alloc fırsatı varsa çağırır.
    - Her çağrıda hızlı ve idempotent; modüller yoksa sessiz no-op.
    """
    global _last_tick
    now = time.time()
    if now - _last_tick < 30:  # spam koruması
        return

    try:
        # 1) Auto-weights (slot/entry ağırlıkları)
        if _weights and hasattr(_weights, "auto_rebalance"):
            _weights.auto_rebalance(cfg)
        elif _weights and hasattr(_weights, "collective_tick"):
            _weights.collective_tick(cfg)

        # 2) Dyn alloc (portföy paylarını güncelle)
        if _dynalloc and hasattr(_dynalloc, "dyn_alloc_tick"):
            _dynalloc.dyn_alloc_tick(cfg)

        # 3) Autopatch / küçük ayar
        if _autopatch and hasattr(_autopatch, "intent_tick"):
            _autopatch.intent_tick(cfg)
        elif _autopatch and hasattr(_autopatch, "autopatch_tick"):
            _autopatch.autopatch_tick(cfg)

        # 4) (Ops.) Evo’ya ufak itme
        if _evo and hasattr(_evo, "maybe_schedule"):
            _evo.maybe_schedule(cfg)

        _last_tick = now
    except Exception as e:
        try:
            tg_send(f"🟠 <b>OnlineLearner</b> tick_intra err: <code>{e}</code>", parse_mode="HTML")
        except Exception:
            pass

def tick_day_end(cfg: Dict[str, Any]) -> None:
    """
    Gün sonu öğrenme/adaptasyon.
    - Evo/GA veya geniş autopatch mantığını tetiklemek için hafif köprü.
    """
    try:
        if _evo and hasattr(_evo, "day_end"):
            _evo.day_end(cfg)
        elif _autopatch and hasattr(_autopatch, "day_end"):
            _autopatch.day_end(cfg)

        tg_send("🧪 <b>OnlineLearner</b>: day-end tick tamamlandı.", parse_mode="HTML")
    except Exception as e:
        try:
            tg_send(f"🟠 <b>OnlineLearner</b> day_end err: <code>{e}</code>", parse_mode="HTML")
        except Exception:
            pass
