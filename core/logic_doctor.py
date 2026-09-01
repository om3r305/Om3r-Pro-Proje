# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    from telegram_utils import tg_send
except Exception:
    def tg_send(*a,parse_mode="HTML",**k): pass

try:
    from brain_hook import brain_overrides
except Exception:
    def brain_overrides(changes: Dict[str, Any], cfg: dict):  # no-op fallback
        pass

LOG = Path("logs/logic_doctor.jsonl"); LOG.parent.mkdir(parents=True, exist_ok=True)

def _wjsonl(obj: dict):
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _normalize_portfolio(port: Dict[str, float]) -> Tuple[Dict[str, float], bool, float]:
    total = sum(v for v in port.values() if isinstance(v, (int, float)))
    if total <= 0:
        return port, False, total
    if abs(total - 1.0) < 1e-6:
        return port, False, total
    new = {k: float(v)/total for k, v in port.items()}
    return new, True, total

def _clamp_entry_frac(cfg: dict) -> Dict[str, float]:
    ef = dict(cfg.get("entry_frac", {}))
    bounds = (cfg.get("dyn_alloc", {}) or {}).get("entry_frac_bounds", {})
    default_min, default_max = 0.20, 0.80
    changed = {}
    for k, v in ef.items():
        v0 = float(v)
        b = bounds.get(k, [default_min, default_max])
        v1 = max(b[0], min(b[1], v0))
        if v1 != v0:
            ef[k] = v1
            changed[k] = (v0, v1)
    return ef if changed else {}

def run_logic_selfheal(cfg: dict) -> None:
    """Düşük risk otomatik düzeltmeler + runtime override üretimi."""
    try:
        # 1) portfolio normalize
        port = cfg.get("portfolio", {})
        if isinstance(port, dict) and port:
            newp, did, old_sum = _normalize_portfolio(port)
            if did:
                cfg["portfolio"] = newp
                _wjsonl({"ts": time.time(), "kind": "normalize_portfolio", "old_sum": old_sum, "new_sum": 1.0})
                try: tg_send("🩹 L9: portfolio normalize edildi (sum ≠ 1).", parse_mode="HTML")
                except Exception: pass

        # 2) entry_frac clamp (bounds’a göre)
        ef_changes = _clamp_entry_frac(cfg)
        if ef_changes:
            cfg["entry_frac"] = {**cfg.get("entry_frac", {}), **{k: v1 for k, (_, v1) in ef_changes.items()}}
            _wjsonl({"ts": time.time(), "kind": "clamp_entry_frac", "changes": ef_changes})
            try: tg_send("🩹 L9: entry_frac clamp uygulandı (bounds).", parse_mode="HTML")
            except Exception: pass

        # 3) performans sezgisi (runtime override – hafif)
        # basit sezgi: DIP uzun süredir kaybediyorsa (events/trades yoksa atla) → entry_frac.dip -0.05
        # Not: Detaylı metrik toplama L8/L9+ sürümlerinde zaten var; burada yumuşak ayar yapıyoruz.
        # Skip edilirse sorun değil.
        # (Burada dosya okumayı opsiyonel bırakıyoruz; yoksa sessiz geçer.)
        try:
            from reporting import recent_slot_perf  # varsa kullan
            perf = recent_slot_perf(lookback_sec=6*3600)  # {"dip":{"pf":..,"wr":..}, ...}
            dip = perf.get("dip", {})
            if dip and dip.get("pf", 1.0) < 0.95 and dip.get("wr", 1.0) < 0.48:
                brain_overrides({"entry_frac.dip": max(0.20, float(cfg.get("entry_frac", {}).get("dip", 0.40)) - 0.05)}, cfg)
                _wjsonl({"ts": time.time(), "kind": "runtime_override", "what": "entry_frac.dip -0.05", "why": "weak_dip_perf"})
                try: tg_send("🧪 L9: runtime override → entry_frac.dip -0.05 (zayıf DIP metrikleri).", parse_mode="HTML")
                except Exception: pass
        except Exception:
            pass

    except Exception as e:
        _wjsonl({"ts": time.time(), "kind": "error", "err": str(e)})
