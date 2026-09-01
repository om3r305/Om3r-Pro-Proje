# -*- coding: utf-8 -*-
from __future__ import annotations
import os, csv, json, sys, time
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_CSV = LOGS_DIR / "events.csv"
TRADES_CSV = LOGS_DIR / "trades.csv"
BRAIN_LOG  = LOGS_DIR / "brain_log.jsonl"

PRINT_EVENTS = os.getenv("PRINT_EVENTS", "1").lower() in ("1","true","yes","on")
PRINT_TRADES = os.getenv("PRINT_TRADES", "1").lower() in ("1","true","yes","on")
FLUSH_STDOUT = True

EVENT_FIELDS = ["ts","kind","sym","slot","px","qty","avg","pnl","reason","cash","msg"]
TRADE_FIELDS = ["ts","sym","slot","side","px","qty","avg","pnl","reason","cash"]

def _utc_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

def _safe_float(v: Any) -> float | None:
    try: return float(v)
    except Exception: return None

def _append_csv(path: Path, fields: list[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        if new_file: wr.writeheader()
        clean = {k: row.get(k, "") for k in fields}
        wr.writerow(clean)

def _print_line(text: str) -> None:
    try:
        print(text)
        if FLUSH_STDOUT: sys.stdout.flush()
    except Exception:
        pass

def log_event(kind: str, **kwargs: Any) -> None:
    row = {"ts": _utc_ts(), "kind": str(kind).lower()}
    mapping = {
        "symbol": "sym", "sym": "sym", "slot": "slot",
        "price": "px", "px": "px", "qty": "qty", "avg": "avg",
        "pnl": "pnl", "reason": "reason", "cash": "cash",
        "message": "msg", "msg": "msg",
    }
    for k, v in kwargs.items():
        key = mapping.get(k, None)
        if key is None:
            prev = row.get("msg", ""); extra = f"{k}={v}"
            row["msg"] = f"{prev} | {extra}" if prev else extra
        else:
            if key in ("px","qty","avg","pnl","cash"):
                fv = _safe_float(v); row[key] = fv if fv is not None else v
            else:
                row[key] = v
    _append_csv(EVENTS_CSV, EVENT_FIELDS, row)
    if PRINT_EVENTS:
        emj = "🟣" if kind.lower() in ("open","buy","long","enter") else "📝"
        _print_line(f"[EVT] {emj} {str(kind).upper()} {row.get('sym','-')}/{row.get('slot','-')} "
                    f"px={row.get('px','-')} pnl={row.get('pnl','')} "
                    f"reason={row.get('reason','')} cash={row.get('cash','')} "
                    f"{('— '+row.get('msg','')) if row.get('msg') else ''}")

def log_trade(sym: Any, slot: Any, side: Any, **kwargs: Any) -> None:
    # her şeyi stringe çevirip sonra upper()
    row = {"ts": _utc_ts(), "sym": str(sym), "slot": str(slot), "side": str(side).upper()}
    for k in ("px","qty","avg","pnl","cash"):
        if k in kwargs:
            fv = _safe_float(kwargs[k])
            row[k] = fv if fv is not None else kwargs[k]
    if "reason" in kwargs:
        row["reason"] = kwargs["reason"]
    _append_csv(TRADES_CSV, TRADE_FIELDS, row)
    if PRINT_TRADES:
        emj = "🟢" if str(side).upper() == "BUY" else "🟣"
        _print_line(
            f"[TRD] {emj} {str(side).upper()} {row['sym']}/{row['slot']} "
            f"px={row.get('px','-')} qty={row.get('qty','-')} pnl={row.get('pnl','')} "
            f"reason={row.get('reason','')}"
        )

def log_brain(kind: str, payload: Dict[str, Any]) -> None:
    rec = {"ts": _utc_ts(), "kind": str(kind), **(payload or {})}
    try:
        with BRAIN_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        _print_line(f"[WARN] log_brain failed: {e}")

__all__ = ["log_event", "log_trade", "log_brain"]
