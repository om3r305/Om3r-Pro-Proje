# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, json, time, random
from pathlib import Path
from typing import Dict, Any, List, Tuple

# --- sessiz TG/override fallback ---
try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, **k): pass

try:
    from Proje1.core.brain_hook import brain_overrides, log_brain
except Exception:
    def brain_overrides(*a, **k): pass
    def log_brain(kind, data): pass

# --------- paths / flags (config ile override edilebilir) ---------
CHAMPIONS = Path("model/evo_champions.jsonl")
FLAGDIR   = Path("runtime/flags")
FORCEFLAG = FLAGDIR / "force_evo.json"
PAUSEFLAG = FLAGDIR / "evo_pause.flag"    # yeni: çalışmayı anlık durdurmak için

CHAMPIONS.parent.mkdir(parents=True, exist_ok=True)
FLAGDIR.mkdir(parents=True, exist_ok=True)

_LAST_RUN_TS = 0.0

# ----------------------- helpers -----------------------
def _cfg(d: Dict[str,Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _f(x, default=0.0):
    try:
        s = str(x).replace(",", ".")
        return float(s)
    except Exception:
        try: return float(x)
        except Exception: return default

def _s(x) -> str:
    try: return str(x)
    except Exception: return ""

def _clamp(v: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo

def _read_rows(path: Path) -> List[Dict[str,Any]]:
    if not path.exists(): return []
    out: List[Dict[str,Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            out.append(r)
    return out

def _metrics_by_slot(rows: List[Dict[str,Any]]) -> Dict[str, Dict[str, float]]:
    """
    SELL/CLOSE işlemlerinden slot bazlı WR/PF hesapla.
    slot etiketi, csv'de reason/slot/kind/label alanlarından birinde olabilir.
    """
    # slot bulucu
    def slot_of(r: Dict[str,Any]) -> str:
        for k in ("slot","Slot","label","reason","tag","enter_reason"):
            v = _s(r.get(k) or "")
            if v:
                v = v.lower()
                # normalize
                if "dip" in v: return "dip"
                if "pred" in v or "plug:pred" in v: return "pred"
                if "news" in v: return "news"
                if "ob" in v or "orderbook" in v: return "ob"
        return "any"

    buckets: Dict[str, List[float]] = { "dip": [], "pred": [], "news": [], "ob": [], "any": [] }
    for r in rows:
        side = _s(r.get("side") or r.get("event") or "").upper()
        if side not in ("SELL","CLOSE","close","sell"):
            continue
        pnl = _f(r.get("pnl") or r.get("realized_pnl"))
        s = slot_of(r)
        buckets.setdefault(s, []).append(pnl)
        buckets["any"].append(pnl)

    def wr_pf(v: List[float]) -> Tuple[float,float,int]:
        n = len(v)
        if n == 0: return 0.5, 1.0, 0
        wins = sum(1 for x in v if x > 0)
        wr = wins / float(n)
        gross_pos = sum(x for x in v if x > 0) or 1e-9
        gross_neg = -sum(x for x in v if x < 0) or 1e-9
        pf = gross_pos / gross_neg
        return wr, pf, n

    out: Dict[str, Dict[str, float]] = {}
    for k, arr in buckets.items():
        wr, pf, n = wr_pf(arr)
        out[k] = {"wr": wr, "pf": pf, "n": n}
    return out

def _choose_source(cfg: Dict[str,Any]) -> Path:
    tfl = Path(_cfg(cfg, "logging.trades_full_log", "logs/trades_full_log.csv"))
    if tfl.exists(): return tfl
    return Path(_cfg(cfg, "paths.events_csv", "logs/events.csv"))

def _champions_path(cfg: Dict[str,Any]) -> Path:
    p = Path(_cfg(cfg, "evo.champions_path", str(CHAMPIONS)))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _write_champion(rec: Dict[str,Any], cfg: Dict[str,Any] | None = None) -> None:
    rec = {"ts": time.time(), **rec}
    try:
        p = _champions_path(cfg or {})
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _norm_por(por: Dict[str,float]) -> Dict[str,float]:
    ks = ("dip","pred","news","ob")
    s = sum(por.get(k,0.0) for k in ks) or 1.0
    return {k: round(float(por.get(k,0.0))/s, 3) for k in ks}

def _apply_bounds(cfg: Dict[str,Any], por: Dict[str,float], ef: Dict[str,float]) -> Tuple[Dict[str,float], Dict[str,float]]:
    """
    dyn_alloc sınırlarına saygı duy (varsa). Yoksa no-op.
    """
    sb = _cfg(cfg, "dyn_alloc.slot_bounds", {}) or {}
    eb = _cfg(cfg, "dyn_alloc.entry_frac_bounds", {}) or {}

    # slot payları (normalize öncesi/sonrası)
    for k, bounds in sb.items():
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
            por[k] = _clamp(por.get(k, 0.0), lo, hi)
        except Exception:
            pass

    por = _norm_por(por)

    # entry_frac sınırları
    for k, bounds in eb.items():
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
            ef[k] = round(_clamp(ef.get(k, 0.4), lo, hi), 2)
        except Exception:
            pass

    return por, ef

# ----------------------- core evo tick -----------------------
def _suggest_from_metrics(cfg: Dict[str,Any], m: Dict[str, Dict[str, float]]) -> Dict[str,Any]:
    """
    Çok hafif 'evrim': iyi slotların payını ufak artır, zayıflarınkini azalt.
    Ayrıca entry_frac'leri yumuşakça it.
    """
    por = dict(_cfg(cfg, "portfolio", {}))
    ef  = dict(_cfg(cfg, "entry_frac", {}))
    if not por: por = {"dip":0.42,"pred":0.30,"news":0.20,"ob":0.08}
    if not ef: ef = {"dip":0.40,"pred":0.40,"news":0.60,"ob":0.50}

    # skor = 0.6*wr + 0.4*min(pf/1.2, 1.0)
    scores = {}
    for k in ("dip","pred","news","ob"):
        wr = float(m.get(k,{}).get("wr",0.5))
        pf = float(m.get(k,{}).get("pf",1.0))
        score = 0.6*wr + 0.4*min(pf/1.2, 1.0)
        scores[k] = score

    # en iyi ve en zayıfı bul
    best  = max(scores, key=scores.get)
    worst = min(scores, key=scores.get)

    step = float(_cfg(cfg, "evo.step_portfolio", 0.02))
    por[best]  = por.get(best,0.0)  + step
    por[worst] = max(0.05, por.get(worst,0.0) - step)

    por = _norm_por(por)

    # entry_frac: best yukarı %+5, worst aşağı %-5
    ef[best]  = round(min(0.80, max(0.20, float(ef.get(best,0.4)) * 1.05)), 2)
    ef[worst] = round(min(0.80, max(0.20, float(ef.get(worst,0.4)) * 0.95)), 2)

    # biraz jitter (antifragile dokunuş)
    for k in list(ef.keys()):
        ef[k] = round(min(0.80, max(0.20, float(ef[k]) * (1.0 + (random.random()-0.5)*0.02))), 2)

    # dyn_alloc sınırlarını uygula (varsa)
    por, ef = _apply_bounds(cfg, por, ef)

    out = {}
    for k,v in por.items():
        out[f"portfolio.{k}"] = v
    for k,v in ef.items():
        out[f"entry_frac.{k}"] = v
    return out

def _should_run(cfg: Dict[str,Any]) -> Tuple[bool, str]:
    global _LAST_RUN_TS
    if PAUSEFLAG.exists():
        return False, "paused_flag"

    if FORCEFLAG.exists():
        return True, "force_flag"

    tick_sec = int(_cfg(cfg, "evo.tick_sec", 300))
    if time.time() - _LAST_RUN_TS >= max(60, tick_sec):
        return True, "interval"
    return False, ""

def tick(cfg: Dict[str,Any], ctx: Dict[str,Any] | None = None) -> None:
    """
    Periyodik 'fast-pass' evrim. Çok hızlı ve güvenli olacak şekilde tasarlandı.
    """
    global _LAST_RUN_TS

    if not bool(_cfg(cfg, "evo.enabled", True)):
        return

    ok, why = _should_run(cfg)
    if not ok:
        return

    src = _choose_source(cfg)
    look = int(_cfg(cfg, "evo.lookback_trades", _cfg(cfg, "brain.learn.lookback_trades", 400)))
    rows = _read_rows(src)
    if look > 0 and len(rows) > look:
        rows = rows[-look:]

    metrics = _metrics_by_slot(rows)
    overrides = _suggest_from_metrics(cfg, metrics)

    # Güvenlik: toplam değişim küçük olsun
    max_delta = float(_cfg(cfg, "evo.max_override_count", 12))
    if len(overrides) > max_delta:
        # limitlersek yine de en etkilileri uygula
        keep = {}
        for k in ("portfolio.dip","portfolio.pred","entry_frac.dip","entry_frac.pred",
                  "portfolio.news","portfolio.ob","entry_frac.news","entry_frac.ob"):
            if k in overrides:
                keep[k] = overrides[k]
        overrides = keep

    # override yaz
    try:
        brain_overrides(overrides, cfg)
    except Exception:
        pass

    # TG + champion kaydı
    info = {
        "why": why,
        "metrics": {k:{kk: round(vv,3) for kk,vv in m.items()} for k,m in metrics.items()},
        "overrides": overrides,
        "ctx": (ctx or {})
    }
    _write_champion(info, cfg)
    log_brain("evo_fastpass", info)
    try:
        s = ", ".join(f"{k}→{v}" for k,v in overrides.items())
        tg_send(f"🧬 Evo: {why} • {s}", parse_mode="HTML")
    except Exception:
        pass

    # force bayrağını temizle
    try:
        if FORCEFLAG.exists():
            FORCEFLAG.unlink(missing_ok=True)
    except Exception:
        pass

    _LAST_RUN_TS = time.time()

__all__ = ["tick"]
