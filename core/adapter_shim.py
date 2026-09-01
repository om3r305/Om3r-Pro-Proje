# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, types, json, time, csv
from pathlib import Path

def _shim_log_ext():
    def write_trades_full_row(**kw):
        p = Path("logs/trades_full_log.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        header = ['ts','event','sym','slot','side','price','qty','avg','pnl','regime','confidence',
                  'spread_bps','reason','enter_reason','tp_abs','sl_abs','vol_norm','alphas_pred','alphas_dip']
        row = [int(time.time())]
        for k in header[1:]:
            row.append(kw.get(k, "" if k in ('event','sym','slot','side','regime','reason','enter_reason') else 0))
        new = not p.exists()
        with p.open('a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if new: w.writerow(header)
            w.writerow(row)
    def log_market_intel(kind, sym, msg, payload=None):
        rec = {'ts': int(time.time()), 'kind': kind, 'sym': sym, 'msg': msg, 'payload': payload or {}}
        p = Path('logs/market_intel.jsonl')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.open('a', encoding='utf-8').write(json.dumps(rec, ensure_ascii=False)+'\n')
    return {"write_trades_full_row": write_trades_full_row, "log_market_intel": log_market_intel}

def _shim_brain_selfheal():
    def report_exception(where: str, e: Exception):
        p = Path("logs/brain_log.jsonl"); p.parent.mkdir(parents=True, exist_ok=True)
        p.open('a', encoding='utf-8').write(json.dumps({"ts": int(time.time()), "where": where, "err": str(e)})+"\n")
    def ensure_selfheal_watcher():
        return True
    return {"report_exception": report_exception, "ensure_selfheal_watcher": ensure_selfheal_watcher}

SHIMS = {
    "core.log_ext": _shim_log_ext,
    "core.brain_selfheal": _shim_brain_selfheal,
}

def inject_missing(module_qualname: str, required: list[str]) -> tuple[bool,str]:
    """Eksik fonksiyonları runtime'da enjekte eder."""
    if module_qualname not in sys.modules:
        __import__(module_qualname)
    mod = sys.modules[module_qualname]
    make = SHIMS.get(module_qualname)
    if not make:
        return False, "no_shim"
    impls = make()
    added = []
    for fn in required:
        if not hasattr(mod, fn) and fn in impls:
            setattr(mod, fn, impls[fn])
            added.append(fn)
    if added:
        return True, f"shim_added:{'+'.join(added)}"
    return False, "no_missing"



