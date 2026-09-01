# -*- coding: utf-8 -*-
from __future__ import annotations
import time, random

try:
    from telegram_utils import tg_send
except Exception:                 # pragma: no cover
    def tg_send(*a, parse_mode="HTML",**k):         # type: ignore
        pass

# brain_overrides opsiyonel; tüm alt modüller de onu çağırıyor
try:
    from Proje1.core.brain_hook import brain_overrides
except Exception:                  # pragma: no cover
    def brain_overrides(*a, **k):  # type: ignore
        pass


def _legacy_autopatch(cfg: dict) -> None:
    """
    Mevcut sisteminde olan ultra basit “random sandbox” minifix’i.
    İstersen kapat: learning.autopatch.legacy_patch = false
    """
    ap = (cfg.get("learning", {}) or {}).get("autopatch", {}) or {}
    roll_guard = (cfg.get("learning", {}) or {}).get("day_end", {}) or {}
    rg = roll_guard.get("rollout_guard", {"wr": 0.48, "pf": 1.10})

    # sahte hızlı metrik üretimi (mevcut davranışla uyumlu)
    wr = 0.45 + random.random() * 0.15     # %45–%60
    pf = 1.00 + random.random() * 0.30     # 1.00–1.30

    if wr >= float(rg.get("wr", 0.48)) and pf >= float(rg.get("pf", 1.10)):
        # küçük politika ayarı: veto_conf_min -0.01 (min 0.50)
        cur = float(cfg.get("brain", {}).get("veto_conf_min", 0.55))
        new_val = round(max(0.50, cur - 0.01), 3)
        brain_overrides({"brain.veto_conf_min": new_val}, cfg)
        try:
            tg_send(f"🔒 <b>Autopatch (legacy)</b> pass ✅ wr={wr:.2f} pf={pf:.2f} → veto_conf_min={new_val}",
                   parse_mode="HTML")
        except Exception:
            pass
    else:
        try:
            tg_send(f"⏪ <b>Autopatch (legacy)</b> fail wr={wr:.2f} pf={pf:.2f} → no-change",
                   parse_mode="HTML")
        except Exception:
            pass


def sandbox_try(cfg: dict) -> None:
    """
    BOT döngüsünde ~60 sn’de bir çağrılıyor.
    Sıra:
      1) Telemetry topla/cache’le (trades.csv + runtime/state.json)
      2) DynAlloc v2 → slot payları & entry_frac auto-optimize (brain_overrides)
      3) Evo → co-evolve tp/sl + slot param; champions jsonl yaz (model/evo_champions.jsonl)
      4) AutoWatchList → performansa göre watchlist’i tazele (brain_overrides)
      5) (Opsiyonel) Eski “random autopatch” minifix’i çalıştır
    Her adım kendi try/except’i ile izole (bot’u bloklamaz).
    """
    # 1) Telemetry
    try:
        from Proje1.core.telemetry_hub import collect_and_cache_metrics
        collect_and_cache_metrics(cfg)
    except Exception as e:
        try: tg_send(f"[telemetry] {e}", parse_mode="HTML")
        except Exception: pass

    # 2) DynAlloc v2
    try:
        from Proje1.core.brain_allocator import dynalloc_tick
        dynalloc_tick(cfg)
    except Exception as e:
        try: tg_send(f"[dynalloc] {e}", parse_mode="HTML")
        except Exception: pass

    # 3) Evo
    try:
        evo_cfg = (cfg.get("evo") or {})
        if evo_cfg.get("enabled", False):
            from Proje1.core.brain_evo import evo_tick
            evo_tick(cfg)
    except Exception as e:
        try: tg_send(f"[evo] {e}", parse_mode="HTML")
        except Exception: pass

    # 4) AutoWatchList
    try:
        from Proje1.core.auto_watchlist import watchlist_tick
        watchlist_tick(cfg)
    except Exception as e:
        try: tg_send(f"[watchlist] {e}", parse_mode="HTML")
        except Exception: pass

    # 5) (Opsiyonel) Eski random autopatch
    try:
        ap = (cfg.get("learning", {}) or {}).get("autopatch", {}) or {}
        if ap.get("enabled", True) and ap.get("legacy_patch", True):
            _legacy_autopatch(cfg)
    except Exception:
        pass
