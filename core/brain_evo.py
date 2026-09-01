# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, random, time
from pathlib import Path
from typing import Dict, Any, List

from Proje1.core.telemetry_hub import get_slot_perf, get_sym_perf

try:
    from Proje1.core.brain_hook import brain_overrides
except Exception:
    def brain_overrides(*a, **k): pass

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
MODEL.mkdir(parents=True, exist_ok=True)

def _cfg(d: Dict[str,Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

def _rand_range(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)

def _co_evolve_pair(gs: Dict[str,Any]) -> Dict[str,float]:
    """
    Basit: tp/sl birlikte örneklenir (co-evolve).
    """
    tp_lo, tp_hi = gs["tp_pct"]; sl_lo, sl_hi = gs["sl_pct"]
    tp = _rand_range(tp_lo, tp_hi)
    # risk-ödül dengesi: sl’yi genelde tp’den düşük tut
    sl = min(_rand_range(sl_lo, sl_hi), max(tp*0.8, sl_lo))
    return {"tp_pct": round(tp,4), "sl_pct": round(sl,4)}

def _score_from_live() -> float:
    # canlı skor proxysi: slot PF/WR’nin ağırlıklı ortalaması
    w = {"ob":1.2,"pred":1.0,"news":0.9,"dip":0.8}
    pf = sum( (get_slot_perf(k).get("pf",1.0))*w[k] for k in w ) / sum(w.values())
    wr = sum( (get_slot_perf(k).get("wr",0.5))*w[k] for k in w ) / len(w)
    return 0.6*pf + 0.4*(wr/(1-wr+1e-6))

def evo_tick(cfg: dict) -> None:
    e = cfg.get("evo", {}) or {}
    if not e.get("enabled", False): return

    champs_path = Path(_cfg(cfg, "evo.champions_path", "model/evo_champions.jsonl"))
    champs_path.parent.mkdir(parents=True, exist_ok=True)

    gs = _cfg(cfg, "evo.genome_space.global", {"tp_pct":[0.03,0.30],"sl_pct":[0.02,0.20]})
    by_slot = _cfg(cfg, "evo.genome_space.by_slot", {})
    per_symbol = _cfg(cfg, "evo.genome_space.per_symbol", {})

    # hızlı popülasyon örnekle
    pop = int(_cfg(cfg, "evo.population", 32))
    muts= float(_cfg(cfg, "evo.mut_rate", 0.10))

    pop_gen = []
    for _ in range(pop):
        g = {"pair": _co_evolve_pair(gs)}
        # slot paramları
        sg = {}
        for slot, space in (by_slot or {}).items():
            one = {}
            for k, rng in (space or {}).items():
                if isinstance(rng, list) and len(rng)==2:
                    one[k] = round(_rand_range(float(rng[0]), float(rng[1])), 4)
            sg[slot] = one
        g["slot"] = sg
        pop_gen.append(g)

    # mutasyon
    for g in pop_gen:
        if random.random() < muts:
            g["pair"] = _co_evolve_pair(gs)

    # canlı skor proxysi
    score = _score_from_live()
    ts = time.time()

    # elit çıkarımı: bu hızlı prototipte canlı skor tek değer → hepsine aynı skor
    # ancak çeşitlilik için küçük rastgelelik ekleyelim
    for g in pop_gen:
        g["score"] = round(max(0.0, score + random.uniform(-0.05, 0.05)), 4)
        g["ts"] = ts

    # şampiyonları yaz (append)
    with champs_path.open("a", encoding="utf-8") as f:
        for g in pop_gen:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    # en iyi eğilim → küçük bir auto-overrides denemesi (temkinli)
    # tp/sl hedeflerini dynamic_tpsl.scale gibi kullanmak yerine,
    # sadece bilgi amaçlı override alanına not düşelim (okuyanlar kullanır).
    best = max(pop_gen, key=lambda x: x["score"])
    brain_overrides({"evo_hint": {"tp_pct": best["pair"]["tp_pct"],
                                  "sl_pct": best["pair"]["sl_pct"],
                                  "ts": ts}}, cfg)
