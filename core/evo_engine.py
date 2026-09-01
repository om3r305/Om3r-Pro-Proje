# -*- coding: utf-8 -*-
from __future__ import annotations
import time, json, math, random, csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

LOGS = Path("logs")
TRADES_CSV = LOGS / "trades.csv"

_DEF_CFG = {
    "evo": {
        "enabled": True,
        "refresh_sec": 30,          # ne kadar sıklıkla evrim/istatistik güncellensin
        "pop_size": 24,             # toplam genom adedi (sembol başına)
        "elite_k": 6,               # elitte tutulacak adet
        "mutate_prob": 0.30,        # mutasyon olasılığı
        "cross_prob": 0.50,         # çaprazlama olasılığı
        "window_trades": 200,       # WR/PF hesaplamasında bakılacak son trade sayısı
        "min_trades_conf": 25,      # confidence için sembol başına minimum trade
        "rr_bounds": {              # risk/ödül sınırları (yüzde)
            "tp_min": 0.40, "tp_max": 2.00,
            "sl_min": 0.30, "sl_max": 2.00
        },
        "alpha_thresh_bounds": {    # pred/alpha eşiği evrimleşsin
            "min": 0.55, "max": 0.85
        }
    }
}

def _cfg(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _clamp(x, a, b): return max(a, min(b, x))

# ------------ genom modeli ------------
# rr: tp_pct (+), sl_pct (+); alpha_th: alpha sinyali için eşik
def _new_genome(bounds: Dict[str, Any]) -> Dict[str, Any]:
    tp = random.uniform(bounds["tp_min"], bounds["tp_max"])
    sl = random.uniform(bounds["sl_min"], bounds["sl_max"])
    th = random.uniform( _DEF_CFG["evo"]["alpha_thresh_bounds"]["min"],
                         _DEF_CFG["evo"]["alpha_thresh_bounds"]["max"] )
    return {"rr": {"tp": round(tp, 4), "sl": round(sl, 4)}, "alpha_th": round(th, 3)}

def _mutate(g: Dict[str, Any], bounds: Dict[str, Any]) -> Dict[str, Any]:
    g = json.loads(json.dumps(g))
    for k in ("tp", "sl"):
        if random.random() < 0.6:
            step = random.uniform(-0.25, 0.25) * g["rr"][k]
            lo = bounds[f"{k}_min"]; hi = bounds[f"{k}_max"]
            g["rr"][k] = round(_clamp(g["rr"][k] + step, lo, hi), 4)
    if random.random() < 0.6:
        b = _DEF_CFG["evo"]["alpha_thresh_bounds"]
        step = random.uniform(-0.08, 0.08)
        g["alpha_th"] = round(_clamp(g["alpha_th"] + step, b["min"], b["max"]), 3)
    return g

def _crossover(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    c = {
        "rr": {
            "tp": round((a["rr"]["tp"] + b["rr"]["tp"]) / 2.0, 4),
            "sl": round((a["rr"]["sl"] + b["rr"]["sl"]) / 2.0, 4),
        },
        "alpha_th": round((a["alpha_th"] + b["alpha_th"]) / 2.0, 3),
    }
    return c

# ------------ performans okuma ------------
def _load_trades_tail(path: Path, n: int) -> List[Dict[str, str]]:
    if not path.exists(): return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        rd = list(csv.DictReader(f))
    if not rd: return []
    tail = rd[-n:]
    for r in tail:
        rows.append(r)
    return rows

def _to_f(x, d=0.0):
    try: return float(x)
    except: return d

def _wr_pf_for_symbol(rows: List[Dict[str, str]], symbol: str) -> Tuple[float, float, int]:
    """Symbol bazında winrate ve profit factor (top kazanç / top kayıp)."""
    wins = 0.0; losses = 0.0; n = 0; w = 0
    for r in rows:
        if (r.get("sym") or r.get("symbol") or "") != symbol: continue
        pnl = _to_f(r.get("pnl") or 0.0, 0.0)
        n += 1
        if pnl > 0: wins += pnl; w += 1
        elif pnl < 0: losses += abs(pnl)
    wr = (w / n) if n > 0 else 0.5
    pf = (wins / losses) if losses > 0 else (2.0 if wins > 0 else 1.0)
    return wr, pf, n

# ------------ EvoEngine ------------
class EvoEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.evcfg = {**_DEF_CFG["evo"], **(_cfg(cfg, "evo", {}) or {})}
        self.enabled = bool(self.evcfg.get("enabled", True))
        self.last_refresh = 0.0
        self.bounds = self.evcfg["rr_bounds"]

        # sembol -> genom listesi (her genom: {"g":..., "score":..., "wr":..., "pf":..., "n":...})
        self.pops: Dict[str, List[Dict[str, Any]]] = {}

    # sembol için popülasyon hazır mı?
    def _ensure_pop(self, symbol: str):
        if symbol in self.pops: return
        pop = [{"g": _new_genome(self.bounds), "score": 0.0, "wr": 0.5, "pf": 1.0, "n": 0}
               for _ in range(self.evcfg["pop_size"])]
        self.pops[symbol] = pop

    # fitness: WR ve PF birleşik skoru (RR paramları dolaylı etki ediyor)
    def _score(self, wr: float, pf: float) -> float:
        # 0..1 arası: WR ağırlık 0.6, PF log ölçeğinde 0.4
        pf_norm = math.tanh(max(0.0, pf - 1.0))  # pf 1.0 üzerini ödüllendir
        return 0.6 * wr + 0.4 * pf_norm

    def _evolve_symbol(self, symbol: str, rows_tail: List[Dict[str, str]]):
        self._ensure_pop(symbol)

        # sembol performansı — tüm pop için aynı veriyi referans alıyoruz (hızlı gölge test)
        wr, pf, n = _wr_pf_for_symbol(rows_tail, symbol)

        # mevcut pop skorla
        for it in self.pops[symbol]:
            it["wr"], it["pf"], it["n"] = wr, pf, n
            it["score"] = self._score(wr, pf)

        # elitleri seç
        elite_k = int(self.evcfg["elite_k"])
        pop_sorted = sorted(self.pops[symbol], key=lambda x: x["score"], reverse=True)
        elites = pop_sorted[:elite_k]

        # yeni kuşak: elitleri koru + çaprazla + mutasyon
        new_pop: List[Dict[str, Any]] = [json.loads(json.dumps(e)) for e in elites]  # kopya
        while len(new_pop) < int(self.evcfg["pop_size"]):
            if random.random() < self.evcfg["cross_prob"] and len(elites) >= 2:
                a, b = random.sample(elites, 2)
                child_g = _crossover(a["g"], b["g"])
            else:
                parent = random.choice(elites) if elites else random.choice(pop_sorted)
                child_g = json.loads(json.dumps(parent["g"]))
            if random.random() < self.evcfg["mutate_prob"]:
                child_g = _mutate(child_g, self.bounds)
            new_pop.append({"g": child_g, "score": 0.0, "wr": wr, "pf": pf, "n": n})
        self.pops[symbol] = new_pop

    def step(self, symbol_engines: List[Tuple[str, Any]]):
        if not self.enabled: return
        now = time.time()
        if now - self.last_refresh < float(self.evcfg["refresh_sec"]): return
        self.last_refresh = now

        rows = _load_trades_tail(TRADES_CSV, int(self.evcfg["window_trades"]))
        # her sembol için ayrı evrim
        for symbol, _st in symbol_engines:
            self._evolve_symbol(symbol, rows)

    # Canlı plugin üretimi: sembol için en iyi genomu kullan; confidence = WR+PF tabanlı
    def live_plugins_for_symbol(self, symbol: str, st) -> List[Dict[str, Any]]:
        if not self.enabled: return []
        self._ensure_pop(symbol)
        pop_sorted = sorted(self.pops[symbol], key=lambda x: x["score"], reverse=True)
        best = pop_sorted[0] if pop_sorted else None
        if not best: return []

        g = best["g"]; wr = best.get("wr", 0.5); pf = best.get("pf", 1.0); n = int(best.get("n", 0))

        # alpha sinyali basit eşik: st.alphas.get('pred',0) > g['alpha_th']
        alpha = float(getattr(st, "alphas", {}).get("pred", 0.5))
        if alpha < float(g.get("alpha_th", 0.7)):
            return []

        # confidence = 0.55 taban + WR/PF katkıları (n azsa cezalandır)
        conf = 0.55
        conf += 0.30 * _clamp((wr - 0.5) * 2.0, -1.0, 1.0)     # WR etkisi (±0.30)
        conf += 0.15 * _clamp(pf / 1.5, 0.0, 1.0)              # PF etkisi (0..0.15)
        if n < int(self.evcfg["min_trades_conf"]):
            conf *= 0.9                                        # veri azsa %10 kıstık
        conf = _clamp(conf, 0.60, 0.98)

        # rr paramlarını not olarak geçir (TP/SL yüzdesel; SymbolEngine tarafı levels() ile absolute üretiyor)
        reason = f"evo(alpha>{g['alpha_th']}, rr=+{g['rr']['tp']}%/-{g['rr']['sl']}%)"
        return [{
            "slot": "pred",
            "reason": reason,
            "confidence": conf,
            "rr": g["rr"],           # istersen ileride close/levels ayarında kullan
        }]
