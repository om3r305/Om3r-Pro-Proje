# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, json, time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, **k): pass

try:
    from Proje1.core.brain_hook import brain_overrides, log_brain
except Exception:
    def brain_overrides(*a, **k): pass
    def log_brain(kind, data): pass

BASELINE = Path("model/drift_baseline.json")
JOURNAL  = Path("logs/drift_watch.jsonl")
FLAGDIR  = Path("runtime/flags")
FORCEEVO = FLAGDIR / "force_evo.json"

for p in (BASELINE.parent, JOURNAL.parent, FLAGDIR):
    p.mkdir(parents=True, exist_ok=True)

_ST = {"flag": "ok", "cool_until": 0.0, "last_ping": 0.0}

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

def _append_j(rec: Dict[str,Any]) -> None:
    rec = {"ts": time.time(), **rec}
    try:
        with JOURNAL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _read_csv_rows(path: Path) -> List[Dict[str,Any]]:
    if not path.exists(): return []
    out: List[Dict[str,Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            out.append(r)
    return out

def _choose_source(cfg: Dict[str,Any]) -> Path:
    tfl = Path(_cfg(cfg, "logging.trades_full_log", "logs/trades_full_log.csv"))
    if tfl.exists(): return tfl
    return Path(_cfg(cfg, "paths.events_csv", "logs/events.csv"))

def _metrics_from_rows(rows: List[Dict[str,Any]]) -> Tuple[float,float,float,int]:
    closes: List[float] = []
    for r in rows:
        side = _s(r.get("side") or r.get("event") or "").upper()
        if side in ("SELL","CLOSE","close","sell"):
            pnl = _f(r.get("pnl") or r.get("realized_pnl"))
            closes.append(pnl)

    n = len(closes)
    if n == 0:
        return 0.5, 1.0, 0.0, 0

    wins = sum(1 for x in closes if x > 0)
    wr = wins / float(n)

    gross_pos = sum(x for x in closes if x > 0) or 1e-9
    gross_neg = -sum(x for x in closes if x < 0) or 1e-9
    pf = gross_pos / gross_neg

    cum = 0.0
    peak = 0.0
    maxdd = 0.0
    for x in closes:
        cum += x
        peak = max(peak, cum)
        dd = cum - peak
        maxdd = min(maxdd, dd)
    return wr, pf, maxdd, n

def _ewma(prev: Optional[float], x: float, alpha: float) -> float:
    if prev is None: return x
    return (1 - alpha) * prev + alpha * x

def _load_baseline() -> Dict[str,Any]:
    if BASELINE.exists():
        try: return json.loads(BASELINE.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}

def _save_baseline(b: Dict[str,Any]) -> None:
    BASELINE.write_text(json.dumps(b, ensure_ascii=False, indent=2), encoding="utf-8")

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _apply_bounds(cfg: Dict[str,Any], por: Optional[Dict[str,float]], ef: Optional[Dict[str,float]]):
    if por is not None:
        sb = dict(_cfg(cfg, "dyn_alloc.slot_bounds", {}))
        for k,v in list(por.items()):
            lo, hi = sb.get(k, [0.0,1.0])
            por[k] = round(_clamp(float(v), float(lo), float(hi)), 3)
        ks = ["dip","pred","news","ob"]
        s = sum(por.get(k,0.0) for k in ks) or 1.0
        for k in ks:
            por[k] = round(float(por.get(k,0.0))/s, 3)
    if ef is not None:
        eb = dict(_cfg(cfg, "dyn_alloc.entry_frac_bounds", {}))
        for k,v in list(ef.items()):
            lo, hi = eb.get(k, [0.20,0.80])
            ef[k] = round(_clamp(float(v), float(lo), float(hi)), 2)
    return por, ef

def _write_override(kv: Dict[str,Any], cfg: Dict[str,Any]) -> None:
    wrote = False
    try:
        brain_overrides(kv, cfg)
        wrote = True
    except Exception:
        pass
    if not wrote:
        try:
            path = Path(_cfg(cfg, "paths.runtime_overrides", "runtime/runtime_overrides.jsonl"))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "overrides": kv}, ensure_ascii=False) + "\n")
        except Exception:
            pass

def _status_flag(ew_wr: float, ew_pf: float, ew_dd: float,
                 wr_warn: float, pf_warn: float, dd_warn: float) -> str:
    unhealthy = (ew_wr < wr_warn) or (ew_pf < pf_warn) or (ew_dd <= dd_warn)
    if not unhealthy:
        return "ok"
    bads = 0
    if ew_wr < wr_warn: bads += 1
    if ew_pf < pf_warn: bads += 1
    if ew_dd <= dd_warn: bads += 1
    return "hard" if bads >= 2 else "soft"

def _probe_and_decide(cfg: Dict[str,Any]) -> Dict[str,Any]:
    alpha     = float(_cfg(cfg, "drift.ewma_alpha", 0.20))
    wr_warn   = float(_cfg(cfg, "drift.thresholds.wr_warn", 0.48))
    pf_warn   = float(_cfg(cfg, "drift.thresholds.pf_warn", 1.05))
    dd_warn   = float(_cfg(cfg, "drift.thresholds.maxdd_warn", -5.0))
    min_n     = int(_cfg(cfg, "drift.min_trades", 50))
    lookback  = int(_cfg(cfg, "drift.lookback_trades", 300))

    src = _choose_source(cfg)
    rows = _read_csv_rows(src)
    if lookback > 0 and len(rows) > lookback:
        rows = rows[-lookback:]

    wr, pf, maxdd, n = _metrics_from_rows(rows)

    base = _load_baseline()
    ew_wr = _ewma(float(base["ewma_wr"]) if "ewma_wr" in base else None, wr, alpha)
    ew_pf = _ewma(float(base["ewma_pf"]) if "ewma_pf" in base else None, pf, alpha)
    ew_dd = _ewma(float(base["ewma_dd"]) if "ewma_dd" in base else None, maxdd, alpha)

    base.update({"ewma_wr": ew_wr, "ewma_pf": ew_pf, "ewma_dd": ew_dd, "last_ts": time.time(), "n": n})
    _save_baseline(base)

    flag = "ok" if n < min_n else _status_flag(ew_wr, ew_pf, ew_dd, wr_warn, pf_warn, dd_warn)

    _append_j({
        "kind":"probe",
        "src": str(src),
        "wr": round(wr,4), "pf": round(pf,3), "maxdd": round(maxdd,2), "n": n,
        "ew_wr": round(ew_wr,4), "ew_pf": round(ew_pf,3), "ew_dd": round(ew_dd,2),
        "warns": {"wr": wr_warn, "pf": pf_warn, "dd": dd_warn},
        "flag": flag
    })

    return {
        "wr":wr, "pf":pf, "maxdd":maxdd, "n":n,
        "ew_wr":ew_wr, "ew_pf":ew_pf, "ew_dd":ew_dd,
        "wr_warn":wr_warn, "pf_warn":pf_warn, "dd_warn":dd_warn,
        "flag":flag
    }

def _act_on_flag(cfg: Dict[str,Any], probe: Dict[str,Any]) -> None:
    now = time.time()
    wr = probe["wr"]; pf = probe["pf"]; n = probe["n"]
    ew_wr = probe["ew_wr"]; ew_pf = probe["ew_pf"]; ew_dd = probe["ew_dd"]
    wr_warn = probe["wr_warn"]; pf_warn = probe["pf_warn"]; dd_warn = probe["dd_warn"]
    flag = probe["flag"]

    soft_cd = int(_cfg(cfg, "drift.actions.cooldown_min", _cfg(cfg, "drift.cooldown_min", 10)))
    hard_cd = int(_cfg(cfg, "drift.cooldown_min_hard", max(soft_cd * 2, 10)))
    tg_on   = bool(_cfg(cfg, "drift.actions.telegram", True))

    if now < float(_ST["cool_until"]):
        return

    if flag == "ok":
        if _ST["flag"] != "ok" and tg_on:
            try: tg_send(f"🟢 Drift OK oldu (wr={wr:.2f} pf={pf:.2f} n={n})")
            except Exception: pass
        _ST["flag"] = "ok"
        _ST["cool_until"] = 0.0
        return

    # --- Tek kaynak: autopatch bounds
    lo_bound = float(_cfg(cfg, "learning.autopatch.lo", 0.45))
    hi_bound = float(_cfg(cfg, "learning.autopatch.hi", 0.80))
    cur_veto = float(_cfg(cfg, "brain.veto_conf_min", 0.55))

    ef = dict(_cfg(cfg, "entry_frac", {})) or {"dip":0.40,"pred":0.40,"news":0.60,"ob":0.50}

    if flag == "soft":
        dv    = float(_cfg(cfg, "drift.bump_veto", 0.02))
        scale = float(_cfg(cfg, "drift.entry_scale", 0.90))

        new_veto = _clamp(cur_veto + dv, lo_bound, hi_bound)
        _write_override({"brain.veto_conf_min": round(new_veto, 3)}, cfg)

        for k in ("dip","pred","news","ob"):
            if k in ef:
                ef[k] = round(max(0.20, min(0.80, float(ef[k]) * scale)), 2)
        _, ef = _apply_bounds(cfg, None, ef)
        if ef:
            _write_override({f"entry_frac.{k}": v for k,v in ef.items()}, cfg)

        if tg_on:
            try: tg_send(f"🟠 Drift (soft): wr={wr:.2f} pf={pf:.2f} n={n} → veto+{dv:.02f}, entry×{scale:.2f}")
            except Exception: pass

        msg = (
            f"⚠️ Drift detected • ew_wr={ew_wr:.2f} (<{wr_warn:.2f}) or "
            f"ew_pf={ew_pf:.2f} (<{pf_warn:.2f}) or ew_dd={ew_dd:.2f} (≤{dd_warn:.2f}) • n={n}"
        )
        log_brain("drift", {"level":"soft","metrics":{"wr":wr,"pf":pf,"maxdd":probe["maxdd"],"n":n},
                            "ewma":{"wr":ew_wr,"pf":ew_pf,"dd":ew_dd}})
        _append_j({"kind":"soft","msg":msg})

        _ST["flag"] = "soft"
        _ST["cool_until"] = now + max(60, soft_cd * 60)

        if bool(_cfg(cfg, "drift.actions.trigger_evo_flag", True)):
            try: FORCEEVO.write_text(json.dumps({"ts": time.time(), "why":"drift_soft"}), encoding="utf-8")
            except Exception: pass
        return

    if flag == "hard":
        dv    = float(_cfg(cfg, "drift.bump_veto_hard", 0.04))
        scale = float(_cfg(cfg, "drift.entry_scale_hard", 0.80))

        new_veto = _clamp(cur_veto + dv, lo_bound, hi_bound)
        _write_override({"brain.veto_conf_min": round(new_veto, 3)}, cfg)

        for k in ("dip","pred","news","ob"):
            if k in ef:
                ef[k] = round(max(0.20, min(0.80, float(ef[k]) * scale)), 2)
        _, ef = _apply_bounds(cfg, None, ef)
        if ef:
            _write_override({f"entry_frac.{k}": v for k,v in ef.items()}, cfg)

        if tg_on:
            try: tg_send(f"🔴 Drift (HARD): wr={wr:.2f} pf={pf:.2f} n={n} → veto+{dv:.02f}, entry×{scale:.2f} (cooldown)")
            except Exception: pass

        msg = (
            f"⚠️ Drift detected • ew_wr={ew_wr:.2f} (<{wr_warn:.2f}) or "
            f"ew_pf={ew_pf:.2f} (<{pf_warn:.2f}) or ew_dd={ew_dd:.2f} (≤{dd_warn:.2f}) • n={n}"
        )
        log_brain("drift", {"level":"hard","metrics":{"wr":wr,"pf":pf,"maxdd":probe["maxdd"],"n":n},
                            "ewma":{"wr":ew_wr,"pf":ew_pf,"dd":ew_dd}})
        _append_j({"kind":"hard","msg":msg})

        _ST["flag"] = "hard"
        _ST["cool_until"] = now + max(60, hard_cd * 60)

        if bool(_cfg(cfg, "drift.actions.trigger_evo_flag", True)):
            try: FORCEEVO.write_text(json.dumps({"ts": time.time(), "why":"drift_hard"}), encoding="utf-8")
            except Exception: pass
        return

def drift_ping(cfg: Dict[str,Any], ctx: Dict[str,Any] | None = None) -> None:
    min_tick = int(_cfg(cfg, "drift.min_tick_sec", 60))
    now = time.time()
    if now - float(_ST["last_ping"]) < max(30, min_tick):
        return
    _ST["last_ping"] = now
    pr = _probe_and_decide(cfg)
    _act_on_flag(cfg, pr)

def tick(cfg: Dict[str,Any]) -> None:
    drift_ping(cfg, None)
