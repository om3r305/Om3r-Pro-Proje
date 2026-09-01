# -*- coding: utf-8 -*-
from __future__ import annotations
import time, csv, json
from pathlib import Path
from typing import Dict, Any, Tuple, List, DefaultDict
from collections import defaultdict

# ---------------- cache ----------------
_CACHE = {
    "last_load_ts": 0.0,
    "symbol_wr": {},         # { "ETHUSDT": (wr, n) }
    "sym_strat_wr": {},      # { ("ETHUSDT","MACD"): (wr, n) }
    "meta": {},              # yüklü meta model (ölçek faktörleri + global wr/pf)
}

# ---------------- cfg helpers ----------------
def _cfg(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# ---------------- IO helpers ----------------
def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        try:
            return list(csv.DictReader(f))
        except Exception:
            return []

def _load_events_csv(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Eski davranışı koru (events.csv)
    events_path = Path(_cfg(cfg, "paths.events_csv", "logs/events.csv"))
    return _read_csv(events_path)

def _load_trades_full_csv(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    # L10: varsa trades_full_log.csv öncelikli
    p = Path(_cfg(cfg, "logging.trades_full_log", "logs/trades_full_log.csv"))
    return _read_csv(p)

def _to_float(x, default=0.0):
    try:
        s = str(x).strip()
        if s == "":
            return default
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        try:
            return float(x)
        except Exception:
            return default

def _to_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return ""

def _warm_decay(idx_from_end: int, decay: float) -> float:
    # sondan geriye: 0 -> 1.0, 1 -> decay, 2 -> decay^2 ...
    return decay ** idx_from_end

# ---------------- metrics ----------------
def _compute_wr(rows: List[Dict[str, Any]], lookback: int, decay: float):
    """SELL/CLOSE işlemlerinden per-symbol ve (symbol,strategy) WR çıkar."""
    sells: List[Tuple[str, str, float]] = []
    for r in rows:
        side  = _to_str(r.get("side") or r.get("event") or "").upper()
        # trades_full_log: side SELL, events.csv: event close/sell olabilir
        if side not in ("SELL", "CLOSE"):
            continue
        sym   = _to_str(r.get("symbol") or r.get("sym") or r.get("pair") or "").strip()
        pnl   = _to_float(r.get("pnl") or r.get("realized_pnl"))
        strat = _to_str(r.get("strategy") or r.get("reason") or "any").strip()
        if sym:
            sells.append((sym, strat, pnl))

    if lookback > 0 and len(sells) > lookback:
        sells = sells[-lookback:]

    per_sym: DefaultDict[str, Tuple[float,float]] = defaultdict(lambda: (0.0,0.0))
    per_sym_strat: DefaultDict[Tuple[str,str], Tuple[float,float]] = defaultdict(lambda: (0.0,0.0))
    for i, (sym, strat, pnl) in enumerate(reversed(sells)):
        w = _warm_decay(i, decay)
        is_win = 1.0 if pnl > 0 else 0.0
        s_w, s_t = per_sym[sym]
        per_sym[sym] = (s_w + is_win*w, s_t + w)
        key = (sym, strat or "any")
        ss_w, ss_t = per_sym_strat[key]
        per_sym_strat[key] = (ss_w + is_win*w, ss_t + w)

    symbol_wr = {sym: ((w/t) if t>0 else 0.5, t) for sym,(w,t) in per_sym.items()}
    sym_strat_wr = {key: ((w/t) if t>0 else 0.5, t) for key,(w,t) in per_sym_strat.items()}
    return symbol_wr, sym_strat_wr

def _compute_global_wr_pf(rows: List[Dict[str, Any]], lookback: int, decay: float) -> Tuple[float, float, int]:
    """Decay’li global WR / PF / N (yalnızca SELL/CLOSE kayıtlarından)."""
    sells: List[float] = []
    for r in rows:
        side  = _to_str(r.get("side") or r.get("event") or "").upper()
        if side not in ("SELL", "CLOSE"):
            continue
        sells.append(_to_float(r.get("pnl") or r.get("realized_pnl")))

    if lookback > 0 and len(sells) > lookback:
        sells = sells[-lookback:]

    wins_w = 0.0
    tot_w  = 0.0
    pos = 0.0
    neg = 0.0
    for i, pnl in enumerate(reversed(sells)):
        w = _warm_decay(i, decay)
        tot_w += w
        if pnl > 0:
            wins_w += w
            pos += pnl * w
        else:
            neg += abs(pnl) * w

    n = len(sells)
    wr = (wins_w / tot_w) if tot_w > 0 else 0.0
    pf = (pos / neg) if neg > 1e-9 else (pos > 0 and 99.0 or 0.0)
    return wr, pf, n

# ---------------- meta IO ----------------
def _load_meta(cfg: Dict[str,Any]):
    path = _cfg(cfg, "brain.learn.meta_path", "model/brain_meta.json")
    p = Path(path)
    if p.exists():
        try:
            _CACHE["meta"] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            _CACHE["meta"] = {}
    else:
        _CACHE["meta"] = {}

def _save_meta(cfg: Dict[str,Any], meta: Dict[str,Any]):
    path = _cfg(cfg, "brain.learn.meta_path", "model/brain_meta.json")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- refresh & daily ----------------
def refresh_cache(cfg: Dict[str,Any]) -> None:
    lookback    = int(_cfg(cfg, "brain.learn.lookback_trades", 400) or 400)
    decay       = float(_cfg(cfg, "brain.learn.decay", 0.92) or 0.92)

    # Önce trades_full_log.csv; yoksa events.csv
    rows = _load_trades_full_csv(cfg)
    if not rows:
        rows = _load_events_csv(cfg)

    sym_wr, sym_strat_wr = _compute_wr(rows, lookback, decay)
    _CACHE["symbol_wr"] = sym_wr
    _CACHE["sym_strat_wr"] = sym_strat_wr
    _load_meta(cfg)
    _CACHE["last_load_ts"] = time.time()

def maybe_refresh(cfg: Dict[str,Any]) -> None:
    if time.time() - float(_CACHE["last_load_ts"]) > 60.0:
        refresh_cache(cfg)

def retrain_daily(cfg: Dict[str,Any]) -> None:
    """
    Günlük hafif “meta” üretir:
      - global wr/pf/n (decay’li)
      - her (sym,strat) için ölçek katsayısı meta['scale'][...]
    """
    lookback    = int(_cfg(cfg, "brain.learn.lookback_trades", 400) or 400)
    decay       = float(_cfg(cfg, "brain.learn.decay", 0.92) or 0.92)
    min_trades  = int(_cfg(cfg, "brain.learn.min_trades", 30) or 30)
    max_boost   = float(_cfg(cfg, "brain.learn.max_boost", 0.35) or 0.35)

    rows = _load_trades_full_csv(cfg)
    if not rows:
        rows = _load_events_csv(cfg)

    # per-symbol/strategy
    sym_wr, sym_strat_wr = _compute_wr(rows, lookback, decay)

    # global
    g_wr, g_pf, g_n = _compute_global_wr_pf(rows, lookback, decay)

    # meta paketle
    meta: Dict[str, Any] = {
        "ts": time.time(),
        "wr": round(float(g_wr), 4),
        "pf": round(float(g_pf), 4),
        "n":  int(g_n),
        "scale": {}
    }

    for (sym,strat),(wr,n) in sym_strat_wr.items():
        if n < min_trades:
            continue
        scale = (wr - 0.5) * 2.0     # [-1..+1]
        factor = 1.0 + max_boost * scale
        meta["scale"][f"{sym}::{strat}"] = round(max(0.0, min(2.0, factor)), 4)

    _save_meta(cfg, meta)
    _CACHE["meta"] = meta
    _CACHE["symbol_wr"] = sym_wr
    _CACHE["sym_strat_wr"] = sym_strat_wr
    _CACHE["last_load_ts"] = time.time()

# ---------------- adjustment ----------------
def adjust_confidence(cfg: Dict[str, Any],
                      symbol: str,
                      strategy_name: str,
                      reason: str,
                      conf: float) -> float:
    """
    Sende mevcut davranış olduğu gibi:
    - (symbol,strategy) meta scale → çarp
    - yeterince işlem varsa symbol WR’a göre ek çarpan uygula
    """
    if not _cfg(cfg, "brain.learn.enabled", True):
        return float(conf)
    maybe_refresh(cfg)

    symbol = _to_str(symbol)
    strategy_name = _to_str(strategy_name)

    # 1) sembol + strateji meta faktörü
    meta = _CACHE.get("meta") or {}
    scale_map = (meta.get("scale") or {})
    key = f"{symbol}::{strategy_name}"
    factor = float(scale_map.get(key, 1.0))

    # 2) sembol bazlı wr (fallback / ek kuvvet)
    wr_map = _CACHE.get("symbol_wr") or {}
    wr, n  = wr_map.get(symbol, (0.5, 0))
    min_tr = int(_cfg(cfg, "brain.learn.min_trades", 30) or 30)
    if n >= min_tr:
        max_boost = float(_cfg(cfg, "brain.learn.max_boost", 0.35) or 0.35)
        scale = max(-1.0, min(1.0, (wr - 0.5) * 2.0))
        factor *= (1.0 + max_boost * scale)

    adj = float(conf) * factor
    return max(0.0, min(1.0, adj))
