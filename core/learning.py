# -*- coding: utf-8 -*-
from __future__ import annotations
import time, json, math, random
from pathlib import Path
from typing import Dict, Any, Tuple

from Proje1.core.brain_hook import brain_overrides
try:
    from telegram_utils import tg_send
except Exception:
    def tg_send(*a,  parse_mode="HTML",**k): pass

LOGS = Path("logs")
EVENTS = LOGS / "events.csv"

_state = {"last_intra": 0.0, "slot_score": {"dip":0.0,"pred":0.0,"news":0.0,"ob":0.0}}

def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())

def _read_events_today() -> list[dict]:
    if not EVENTS.exists(): return []
    out = []
    try:
        for line in EVENTS.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
        # naive CSV reader (fast)
        import csv
        with EVENTS.open("r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                if _today() in str(r.get("ts","")):
                    out.append(r)
    except Exception:
        pass
    return out

def intra_day_bandit(cfg: dict) -> None:
    """Her 10 dakikada slot skorlarını güncelle; kötü slotları veto eşiğini artırmaya yönlendir."""
    lc = cfg.get("learning", {}).get("intra_day", {})
    if not lc.get("enabled", True): return
    per = int(lc.get("update_every_min", 10)) * 60
    now = time.time()
    if now - _state["last_intra"] < per: return

    rows = _read_events_today()
    if not rows: return
    pnl_by_slot = {"dip":0.0,"pred":0.0,"news":0.0,"ob":0.0}
    trades = 0
    for r in rows:
        if str(r.get("kind","")).lower() in ("close","sell","exit","realize","realized"):
            slot = str(r.get("slot","")).lower()
            pnl = float(r.get("pnl",0) or 0)
            pnl_by_slot[slot] = pnl_by_slot.get(slot,0.0) + pnl
            trades += 1

    if trades < int(lc.get("min_trades", 8)):
        _state["last_intra"] = now
        return

    # skoru -1..+1 normalize et
    for k,v in pnl_by_slot.items():
        _state["slot_score"][k] = max(-1.0, min(1.0, v / max(1.0, abs(v)+5.0)))

    # örnek: kötüleşen slotta veto_conf_min +0.03
    bump = 0.0
    if min(_state["slot_score"].values()) < -0.3:
        bump = 0.03

    if bump > 0:
        brain_overrides({"brain.veto_conf_min": float(cfg.get("brain",{}).get("veto_conf_min",0.55))+bump}, cfg)
        try: tg_send(f"🧪 *Learn (intra)*: slot_score={_state['slot_score']} → bump veto_conf_min +{bump:.2f}", parse_mode="HTML")
        except Exception: pass

    _state["last_intra"] = now

def day_end_grid(cfg: dict) -> None:
    """Gün sonunda küçük grid araması → yarın için override."""
    lc = cfg.get("learning", {}).get("day_end", {})
    if not lc.get("enabled", True): return
    # çok kısa örnek: günün toplam PnL'ine göre tp/sl ayarı
    rows = _read_events_today()
    pnl_total = 0.0
    wins = losses = 0
    for r in rows:
        if str(r.get("kind","")).lower() in ("close","sell","exit","realize","realized"):
            p = float(r.get("pnl",0) or 0.0)
            pnl_total += p
            if p > 0: wins += 1
            elif p < 0: losses += 1
    wr = (wins / max(1, (wins+losses)))

    # basit grid adayları
    C = int(lc.get("grid_candidates", 12))
    best = {"score": -1e9, "tp": 1.10, "sl": 0.80}
    for _ in range(C):
        tp = 1.00 + random.random()*0.6   # %1.0–%1.6
        sl = 0.70 + random.random()*0.4   # %0.7–%1.1
        # fake skor: wr ve pnl_total ağırlıklı (yerine replay backtest bağlanabilir)
        score = pnl_total + (wr-0.45)*10.0 + (tp-sl)
        if score > best["score"]:
            best = {"score": score, "tp": tp, "sl": sl}

    guard = lc.get("rollout_guard", {"wr":0.48,"pf":1.10,"maxdd":-3.0})
    # basit kabul ölçütü: wr ~ %48 üzeri ise yaz
    if wr >= guard.get("wr", 0.48):
        brain_overrides({
            "brain.adjust.trend_tp_mult": round(best["tp"]/1.10, 3),
            "brain.adjust.trend_sl_mult": round(best["sl"]/0.80, 3)
        }, cfg)
        try:
            from telegram_utils import tg_send
            tg_send(f"🧪 *Learn (day-end)*: wr={wr:.2f} → overrides: "
                    f"trend_tp_mult≈{best['tp']/1.10:.2f}, trend_sl_mult≈{best['sl']/0.80:.2f}",
                    parse_mode="HTML")
        except Exception: pass
