## -*- coding: utf-8 -*-
from __future__ import annotations
import time, json
from typing import Dict, Tuple
from Proje1.core.telemetry_hub import get_slot_perf, snapshot

try:
    from telegram_utils import tg_send
except Exception:
    def tg_send(*a,parse_mode="HTML",**k): pass

try:
    from Proje1.core.brain_hook import brain_overrides
except Exception:
    def brain_overrides(suggest: Dict, cfg: Dict|None=None): pass

_last_emit = 0.0

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _step_towards(cur: float, target: float, step: float) -> float:
    if target > cur: return min(cur + step, target)
    if target < cur: return max(cur - step, target)
    return cur

def _normalize_alloc(d: Dict[str,float], eps: float=1e-9) -> Dict[str,float]:
    s = sum(max(0.0, x) for x in d.values())
    if s < eps:
        n = len(d) or 1
        return {k: 1.0/n for k in d}
    return {k: max(0.0,x)/s for k,x in d.items()}

def dynalloc_tick(cfg: dict) -> None:
    da = cfg.get("dyn_alloc", {}) or {}
    if not da.get("enabled", False): return

    # hedefler / parametreler
    lookback = int(da.get("lookback_sec", 21600))
    step = float(da.get("step", 0.05))
    slot_bounds = da.get("slot_bounds", {
        "dip":[0.1,0.5],"pred":[0.1,0.5],"news":[0.05,0.4],"ob":[0.05,0.4]
    })
    entry_bounds = da.get("entry_frac_bounds", {
        "dip":[0.2,0.6],"pred":[0.2,0.6],"news":[0.3,0.8],"ob":[0.25,0.7]
    })
    targets = da.get("targets", {"pf_min":1.10,"wr_min":0.50})

    # mevcut ayarlar
    cur_alloc = dict(cfg.get("portfolio", {"dip":0.42,"pred":0.30,"news":0.20,"ob":0.08}))
    cur_entry = dict(cfg.get("entry_frac", {"dip":0.40,"pred":0.40,"news":0.60,"ob":0.50}))

    # performans → hedef paylar
    slots = ["dip","pred","news","ob"]
    raw_score = {}
    for s in slots:
        perf = get_slot_perf(s)
        pf = float(perf.get("pf", 1.0)); wr = float(perf.get("wr", 0.5))
        # basit skor: PF ağırlıklı + WR katkısı, eşiğin altındaysa zayıflat
        score = (pf) * (0.7 + 0.6*(wr - 0.5))  # wr=0.5 → 0.7x; wr artarsa bonus
        if wr < targets.get("wr_min",0.5): score *= 0.6
        if pf < targets.get("pf_min",1.0): score *= 0.7
        raw_score[s] = max(0.0, score)

    # normalize + bounded hedef
    tot = sum(raw_score.values()) or 1.0
    tgt_alloc = {}
    for s in slots:
        base = raw_score[s] / tot
        lo, hi = slot_bounds.get(s, [0.05,0.5])
        tgt_alloc[s] = _clamp(base, lo, hi)

    # adımlı yaklaş
    new_alloc = {}
    for s in slots:
        new_alloc[s] = _step_towards(cur_alloc.get(s,0.0), tgt_alloc[s], step)
    new_alloc = _normalize_alloc(new_alloc)

    # entry_frac’ı da aynı oranda iter
    new_entry = {}
    for s in slots:
        lo, hi = entry_bounds.get(s, [0.2,0.6])
        target_frac = _clamp(new_alloc[s] * 0.8 + cur_entry.get(s,0.4)*0.2, lo, hi)
        new_entry[s] = _step_towards(cur_entry.get(s,0.4), target_frac, step)

    # override yaz
    brain_overrides({"portfolio": new_alloc, "entry_frac": new_entry}, cfg)

    global _last_emit
    if time.time() - _last_emit > 60:
        try:
            snap = snapshot()
            tg_send(
                "🧠 DynAlloc v2\n"
                f"• alloc: { {k:round(v,3) for k,v in new_alloc.items()} }\n"
                f"• entry: { {k:round(v,3) for k,v in new_entry.items()} }\n"
                f"• slot_pf: { {k:round((get_slot_perf(k).get('pf') or 0.0),2) for k in ['dip','pred','news','ob']} }"
            , parse_mode="HTML")
        except Exception: pass
        _last_emit = time.time()
