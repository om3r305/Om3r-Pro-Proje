# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, time, math
from pathlib import Path
from typing import Dict, Any, Tuple, List, DefaultDict
from collections import defaultdict

from Proje1.core.brain_hook import brain_overrides

SLOTS = ("dip","pred","news","ob")

def _cfg(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _trades_path(cfg: Dict[str,Any]) -> Path:
    p = _cfg(cfg, "logging.trade_csv", "logs/trades.csv")
    return Path(p)

def _read_trades(path: Path, lookback: int) -> List[Dict[str,str]]:
    if not path.exists(): return []
    rows: List[Dict[str,str]] = []
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    return rows[-lookback:] if lookback>0 and len(rows)>lookback else rows

def _safe_f(x, d=0.0):
    try: return float(x)
    except: return d

def _slot_perf(rows: List[Dict[str,str]]) -> Dict[str, float]:
    # son işlemlerden slot bazlı PnL toplamı
    perf = {s:0.0 for s in SLOTS}
    for r in rows:
        slot = (r.get("slot") or "").lower()
        if slot in perf:
            perf[slot] += _safe_f(r.get("pnl") or r.get("pnl_usd") or 0.0, 0.0)
    return perf

def _softmax_scale(perf: Dict[str,float], temp: float, floors: Dict[str,float], caps: Dict[str,float]) -> Dict[str,float]:
    # negatiflerin de katkı vermesi için offset
    vals = list(perf.values())
    off = -min(0.0, min(vals)) + 1e-6
    exps = {k: math.exp((v+off)/max(1e-6, temp)) for k,v in perf.items()}
    s = sum(exps.values()) or 1.0
    raw = {k: exps[k]/s for k in exps}
    # slot bazında floor/cap uygula + normalize
    out = {}
    for k, w in raw.items():
        w = max(floors.get(k,0.0), min(caps.get(k,1.0), w))
        out[k] = w
    tot = sum(out.values()) or 1.0
    return {k: v/tot for k,v in out.items()}

def _blend(old_map: Dict[str,float], new_map: Dict[str,float], alpha: float) -> Dict[str,float]:
    # EMA karışımı: alpha yeni ağırlık
    out = {}
    keys = set(old_map.keys()) | set(new_map.keys())
    for k in keys:
        o = float(old_map.get(k,0.0))
        n = float(new_map.get(k,0.0))
        out[k] = (1.0-alpha)*o + alpha*n
    # normalize
    tot = sum(out.values()) or 1.0
    return {k: v/tot for k,v in out.items()}

def _cur_map(cfg: Dict[str,Any], key: str, default: Dict[str,float]) -> Dict[str,float]:
    m = cfg.get(key, {}) or {}
    out = {}
    for s in SLOTS:
        out[s] = float(m.get(s, default.get(s,0.0)))
    # normalize (güvenlik)
    tot = sum(out.values()) or 1.0
    return {k: v/tot for k,v in out.items()}

def tick(cfg: Dict[str,Any]) -> bool:
    """
    Dinamik ağırlıklandırma: son işlemlere göre portfolio & entry_frac ayarla.
    Ayarlar:
      brain.auto_weights.enabled: bool
      brain.auto_weights.lookback_trades: int
      brain.auto_weights.temperature: float
      brain.auto_weights.alpha: float (EMA karışım)
      brain.auto_weights.floors: {slot: min}
      brain.auto_weights.caps:   {slot: max}
      brain.auto_weights.entry_frac_floor_cap: [floor, cap]  (toplam için değil, slot bazlı çarpan)
    """
    aw = _cfg(cfg, "brain.auto_weights", {})
    if not aw or not aw.get("enabled", False):
        return False

    lookback   = int(aw.get("lookback_trades", 400))
    temp       = float(aw.get("temperature", 0.6))
    alpha      = float(aw.get("alpha", 0.35))
    floors     = {k: float(v) for k,v in (aw.get("floors") or {}).items()}
    caps       = {k: float(v) for k,v in (aw.get("caps") or {}).items()}
    ef_floor, ef_cap = (0.3, 1.7)
    if isinstance(aw.get("entry_frac_floor_cap"), (list, tuple)) and len(aw["entry_frac_floor_cap"])==2:
        ef_floor, ef_cap = float(aw["entry_frac_floor_cap"][0]), float(aw["entry_frac_floor_cap"][1])

    trades_path = _trades_path(cfg)
    rows = _read_trades(trades_path, lookback)
    if not rows:
        return False

    perf = _slot_perf(rows)
    # softmax → hedef paylar
    target = _softmax_scale(perf, temp, floors, caps)

    # mevcut harita
    cur_port = _cur_map(cfg, "portfolio", {"dip":0.42,"pred":0.30,"news":0.20,"ob":0.08})
    new_port = _blend(cur_port, target, alpha)

    # entry_frac → performansa göre çarpanla yeniden şekillendir
    # baz harita
    cur_entry = _cur_map(cfg, "entry_frac", {"dip":0.40,"pred":0.40,"news":0.60,"ob":0.50})
    # normalize edilmiş target’ı referans alıp slot bazlı çarpan üret
    # referans = 1/len(SLOTS)
    ref = 1.0/len(SLOTS)
    scale = {}
    for s in SLOTS:
        ratio = (target.get(s,ref)/max(1e-9, ref))
        mult  = max(ef_floor, min(ef_cap, ratio))
        scale[s] = mult
    # uygula + normalize etme (entry_frac toplam 1 olmak zorunda değil)
    upd_entry = {s: round(cur_entry.get(s,0.0)*scale[s], 4) for s in SLOTS}

    # override yaz
    brain_overrides({"portfolio": {k: round(v,4) for k,v in new_port.items()},
                     "entry_frac": upd_entry}, cfg)
    return True
