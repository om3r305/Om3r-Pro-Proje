# -*- coding: utf-8 -*-
from __future__ import annotations
import os, csv, json, time
from pathlib import Path
from typing import Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
MODEL = ROOT / "model"
LOGS.mkdir(parents=True, exist_ok=True)
MODEL.mkdir(parents=True, exist_ok=True)

_TFL_PATH = LOGS / "trades_full_log.csv"
_MI_PATH  = LOGS / "market_intel.jsonl"
_CHAMP    = MODEL / "evo_champions.jsonl"

_TFL_HEADERS = [
    "ts","event","sym","slot","side",
    "price","qty","avg","pnl",
    "regime","confidence","spread_bps",
    "reason","enter_reason","tp_abs","sl_abs",
    "vol_norm","alphas_pred","alphas_dip"
]

def _ensure_csv(path: Path, headers) -> None:
    new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if new:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)

def write_trades_full_row(
    *,
    event: str,           # "open" | "close" | "dca"
    sym: str,
    slot: str,
    side: str = "",
    price: float = 0.0,
    qty: float = 0.0,
    avg: float = 0.0,
    pnl: float = 0.0,
    regime: str = "",
    confidence: float = 0.0,
    spread_bps: float = 0.0,
    reason: str = "",
    enter_reason: str = "",
    tp_abs: float = 0.0,
    sl_abs: float = 0.0,
    vol_norm: float = 0.0,
    alphas: Optional[Dict[str, Any]] = None,
) -> None:
    """Bağlamlı trade logu (idempotent, düşük maliyet)."""
    _ensure_csv(_TFL_PATH, _TFL_HEADERS)
    row = {
        "ts": int(time.time()),
        "event": event, "sym": sym, "slot": slot, "side": side,
        "price": float(price), "qty": float(qty), "avg": float(avg), "pnl": float(pnl),
        "regime": regime, "confidence": float(confidence), "spread_bps": float(spread_bps),
        "reason": str(reason), "enter_reason": str(enter_reason),
        "tp_abs": float(tp_abs), "sl_abs": float(sl_abs),
        "vol_norm": float(vol_norm or 0.0),
        "alphas_pred": float((alphas or {}).get("pred", 0.0)),
        "alphas_dip": float((alphas or {}).get("dip", 0.0)),
    }
    with _TFL_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_TFL_HEADERS)
        w.writerow(row)

def log_market_intel(
    *,
    kind: str,            # "news","shock","macro","flow","custom"
    sym: str = "",
    msg: str = "",
    payload: Optional[Dict[str, Any]] = None
) -> None:
    """Haber/olay zekâ günlükleri (JSONL)."""
    rec = {
        "ts": int(time.time()),
        "kind": kind,
        "sym": sym,
        "msg": msg,
        "data": payload or {}
    }
    _MI_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _MI_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def append_evo_champion(champ: Dict[str, Any]) -> None:
    """Evrim şampiyonlarını (genom, skorlar) hafızaya yaz (JSONL)."""
    champ = {"ts": int(time.time()), **(champ or {})}
    with _CHAMP.open("a", encoding="utf-8") as f:
        f.write(json.dumps(champ, ensure_ascii=False) + "\n")
