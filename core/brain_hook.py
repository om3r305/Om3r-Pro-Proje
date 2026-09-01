# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, random, os
from pathlib import Path
from typing import Dict, Any, Optional

# --- Telegram (sessiz fallback) ---
try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, parse_mode="HTML", **k): pass

# --- Yol & dosyalar ---
BRAINLOG   = Path("logs/brain_log.jsonl")
OVERRIDES  = Path("runtime/runtime_overrides.jsonl")
META_PATH  = Path("model/brain_meta.json")                  # learn meta
CMEM_PATH  = Path("logs/collective_memory.jsonl")           # collective memory (opsiyonel)
STATE_PATH = Path("runtime/state.json")                     # dd guard için (opsiyonel)

for p in [BRAINLOG, OVERRIDES, META_PATH.parent]:
    p.parent.mkdir(parents=True, exist_ok=True)

# =============== IO HELPERS ===============
def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _write_json(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def _log(kind: str, data: Dict[str, Any]) -> None:
    rec = {"ts": time.time(), "kind": kind}
    rec.update(data or {})
    try:
        with BRAINLOG.open("a", encoding="utf-8") as f:
            f.write(_j(rec) + "\n")
    except Exception:
        pass

def log_brain(kind: str, data: Dict[str, Any]) -> None:
    _log(kind, data)

# =============== CFG HELPERS ===============
def _cfg_get(cfg: Dict[str, Any] | None, path: str, default: Any = None) -> Any:
    if not cfg: return default
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict): return default
        cur = cur.get(key)
        if cur is None: return default
    return cur

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo

def _clampf(x: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo

# =============== OVERRIDES ===============
def _update_meta_from_override(suggest: Dict[str, Any]) -> None:
    meta = _read_json(META_PATH, {})
    last = dict(meta.get("last_overrides", {}))

    if "brain.veto_conf_min" in suggest:
        try:
            meta["veto_conf_min"] = float(suggest["brain.veto_conf_min"])
        except Exception:
            pass

    port = dict(meta.get("portfolio", {}))
    efrac = dict(meta.get("entry_frac", {}))

    for k, v in suggest.items():
        last[k] = v
        if k.startswith("portfolio."):
            port[k.split(".", 1)[1]] = float(v)
        elif k.startswith("entry_frac."):
            efrac[k.split(".", 1)[1]] = float(v)

    if port:  meta["portfolio"]  = port
    if efrac: meta["entry_frac"] = efrac

    meta["last_overrides"] = last
    meta["ts_overrides"] = time.time()
    meta.setdefault("wr", 0.0)
    meta.setdefault("pf", 0.0)
    meta.setdefault("n",  0)
    _write_json(META_PATH, meta)

def brain_overrides(suggest: Dict[str, Any], cfg: Dict[str, Any] | None = None) -> None:
    if not (_cfg_get(cfg, "brain.auto_overrides", True)):
        return
    try:
        rec = {"ts": time.time(), "set": suggest}
        with OVERRIDES.open("a", encoding="utf-8") as f:
            f.write(_j(rec) + "\n")
        _log("override", {"set": suggest})
        _update_meta_from_override(suggest)
        tg_send(f"🛠️ <b>Override</b>: <code>{_j(suggest)}</code>", parse_mode="HTML")
    except Exception:
        pass

# =============== COLLECTIVE / META TAIL ===============
def _tail_jsonl(path: Path, limit: int) -> list[dict]:
    if not path.exists(): return []
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: out.append(json.loads(line))
                except Exception: pass
    except Exception:
        return []
    return out[-limit:] if limit > 0 else out

# =============== CONF BLEND ===============
def _soft_dd_guard(cfg: Dict[str, Any] | None) -> float:
    """
    Hafif DD koruması: DNA'daki brain.dd_soft_usd > 0 ise,
    runtime/state.json içinde (varsa) 'pnl_today' veya 'realized_today' alanına bakar.
    Yumuşak risk baskısı olarak veto eşiğine +0.02 ekleyebilir.
    (Mevcut verin yoksa 0 döner; tetikleyici tamamen opsiyoneldir.)
    """
    try:
        cap = float(_cfg_get(cfg, "brain.dd_soft_usd", 0.0))
        if cap <= 0:
            return 0.0
        st = _read_json(STATE_PATH, {})
        pnl_today = float(st.get("pnl_today", st.get("realized_today", 0.0)) or 0.0)
        if pnl_today <= -abs(cap):
            return 0.02
        return 0.0
    except Exception:
        return 0.0

def _blend_confidence(symbol: str,
                      base_c: float,
                      market: Dict[str, Any],
                      cfg: Dict[str, Any] | None) -> tuple[float, Dict[str, float]]:
    parts: Dict[str, float] = {"base": float(base_c)}
    c = float(base_c)

    if _cfg_get(cfg, "brain.learn.enabled", True):
        meta = _read_json(META_PATH, {})
        wr = float(meta.get("wr", 0.0) or 0.0)
        pf = float(meta.get("pf", 0.0) or 0.0)
        max_boost = float(_cfg_get(cfg, "brain.learn.max_boost", 0.35))
        score = max(0.0, (pf - 1.0)) + max(0.0, (wr - 0.5))
        boost = min(max_boost, score) * 0.5
        c = _clamp(c + boost); parts["learn"] = boost

    if _cfg_get(cfg, "brain.collective.enabled", True):
        look_sec = int(_cfg_get(cfg, "brain.collective.lookback_sec", 21600))
        cmax = float(_cfg_get(cfg, "brain.collective.max_boost", 0.15))
        now = time.time()
        cmem = _tail_jsonl(CMEM_PATH, 1000)
        score = 0.0
        for rec in cmem:
            if rec.get("symbol") != symbol: continue
            ts = float(rec.get("ts", 0.0))
            if ts > 0 and (now - ts) <= look_sec:
                try: score += float(rec.get("score", 0.0))
                except Exception: pass
        if score != 0.0:
            add = max(-cmax, min(cmax, score * 0.02))
            c = _clamp(c + add); parts["collective"] = add
        else:
            parts["collective"] = 0.0

    if _cfg_get(cfg, "brain.antifragile.enabled", True):
        j = float(_cfg_get(cfg, "brain.antifragile.jitter_conf", 0.02))
        jitter = (random.random() - 0.5) * 2.0 * j
        c = _clamp(c + jitter); parts["anti"] = jitter

    if _cfg_get(cfg, "qrng.enabled", False):
        w = float(_cfg_get(cfg, "qrng.weight", 0.05))
        rq = (random.random() - 0.5) * 2.0 * w
        c = _clamp(c + rq); parts["qrng"] = rq
    else:
        parts["qrng"] = 0.0

    regime = str(market.get("regime", "UNKNOWN"))
    if regime == "TREND":
        bonus = float(_cfg_get(cfg, "brain.conf_blend.trend_bonus", 0.15))
        c = _clamp(c + bonus); parts["trend"] = bonus
    else:
        parts["trend"] = 0.0

    _log("conf-blend", {"symbol": symbol, "parts": parts, "final": c})
    return c, parts

# =============== DECIDE ===============
def decide_trade(symbol: str, slot: str, proposal: Dict[str, Any],
                 market: Dict[str, Any], cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    mode = str(_cfg_get(cfg, "brain.mode", "driver")).lower()  # "driver" | "governor"
    authority = True if mode == "driver" else bool(_cfg_get(cfg, "brain.authority", False))
    adj_cfg   = _cfg_get(cfg, "brain.adjust", {}) or {}
    cmin_base = float(_cfg_get(cfg, "brain.veto_conf_min", 0.55))

    # --- Tek otorite: autopatch clamp (DNA: learning.autopatch.{lo,hi})
    ap_lo = float(_cfg_get(cfg, "learning.autopatch.lo", 0.45))
    ap_hi = float(_cfg_get(cfg, "learning.autopatch.hi", 0.80))
    cmin_base = _clampf(cmin_base, ap_lo, ap_hi)

    # --- Adjust clamp güvenli sınırlar
    chop_q = _clampf(adj_cfg.get("chop_qty_mult", 0.6), 0.3, 1.0)
    tr_tp  = _clampf(adj_cfg.get("trend_tp_mult", 1.06), 0.90, 1.50)
    tr_sl  = _clampf(adj_cfg.get("trend_sl_mult", 0.97), 0.80, 1.10)
    spr_pb = _clampf(adj_cfg.get("spread_qty_penalty_bps", 8), 0.0, 50.0)

    dmin      = float(_cfg_get(cfg, "brain.conf_blend.driver_min", 0.35))
    gmin      = float(_cfg_get(cfg, "brain.conf_blend.governor_min", 0.40))

    regime = str(market.get("regime", "UNKNOWN"))
    spread_bps = float(market.get("spread_bps", 1.0))
    ext = market.get("ext", {"news_shock":0.0, "macro_risk":1.0, "flow":"neutral"})

    base_conf = float(proposal.get("confidence", 0.5))
    c_eff, parts = _blend_confidence(symbol, base_conf, market, cfg)

    if mode == "driver": c_eff = max(dmin, c_eff)
    else:                c_eff = max(gmin, c_eff)

    cmin = cmin_base + _soft_dd_guard(cfg)
    if float(ext.get("macro_risk", 1.0)) > 1.3 or float(ext.get("news_shock", 0.8)) > 0.8:
        cmin += 0.05

    if (mode == "governor") and (not authority):
        dec = {"action": "approve", "reason": "governor_no_authority", "confidence": c_eff}
        _log("decide-governor", {"symbol": symbol, "slot": slot, "proposal": proposal,
                                 "market": market, "decision": dec})
        try:
            tg_send(f"🧠 <b>Gov-Approve</b> <code>{symbol}</code> [{slot}] — reg={regime} conf={c_eff:.2f}",
                    parse_mode="HTML")
        except Exception: pass
        return dec

    if c_eff < cmin:
        dec = {"action": "reject", "reason": f"low_conf:{c_eff:.2f} < {cmin:.2f}", "confidence": c_eff}
        _log(f"decide-{mode}", {"symbol": symbol, "slot": slot, "proposal": proposal,
                                "market": market, "decision": dec, "parts": parts})
        try:
            tg_send(f"🧠 <b>Reject</b> <code>{symbol}</code> [{slot}] — conf={c_eff:.2f} (min {cmin:.2f}) reg={regime}",
                    parse_mode="HTML")
        except Exception: pass
        return dec

    upd: Dict[str, Any] = {}
    if regime == "CHOP":
        upd["qty"] = max(0.0, float(proposal.get("qty", 0.0)) * chop_q)
        if "sl" in proposal: upd["sl"] = float(proposal["sl"]) * 0.985
    elif regime == "TREND":
        if "tp" in proposal: upd["tp"] = float(proposal["tp"]) * tr_tp
        if "sl" in proposal: upd["sl"] = float(proposal["sl"]) * tr_sl

    if spread_bps > spr_pb:
        curq = float(upd.get("qty", proposal.get("qty", 0.0)))
        upd["qty"] = max(0.0, curq * 0.8)

    if _cfg_get(cfg, "brain.antifragile.enabled", True):
        j = 0.015
        upd["qty"] = max(0.0, float(upd.get("qty", proposal.get("qty", 0.0))) * (1.0 + (random.random()-0.5)*2*j))

    action = "adjust" if upd else "approve"
    dec = {"action": action, "update": upd, "reason": f"reg={regime}", "confidence": c_eff}

    _log(f"decide-{mode}", {
        "symbol": symbol, "slot": slot, "proposal": proposal, "market": market,
        "decision": dec, "parts": parts, "cmin": cmin
    })

    try:
        if action == "adjust":
            q = float(upd.get('qty', proposal.get('qty', 0.0)))
            tp = float(upd.get('tp', proposal.get('tp', 0.0))) if 'tp' in proposal or 'tp' in upd else None
            sl = float(upd.get('sl', proposal.get('sl', 0.0))) if 'sl' in proposal or 'sl' in upd else None
            msg = f"🧠 <b>Adjust</b> <code>{symbol}</code> [{slot}] → qty={q:.6f}"
            if tp is not None: msg += f" tp={tp:.6f}"
            if sl is not None: msg += f" sl={sl:.6f}"
            msg += f"\nreg={regime} conf={c_eff:.2f}"
            tg_send(msg, parse_mode="HTML")
        else:
            tg_send(f"🧠 <b>Approve</b> <code>{symbol}</code> [{slot}] — reg={regime} conf={c_eff:.2f}",
                    parse_mode="HTML")
    except Exception:
        pass

    return dec

__all__ = ["decide_trade", "log_brain", "brain_overrides"]
