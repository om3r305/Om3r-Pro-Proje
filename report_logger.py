# report_logger.py
import os, json, time, csv
from datetime import datetime

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _day_dir() -> str:
    d = datetime.utcnow().strftime("%Y%m%d")
    p = os.path.join("logs", "reports", d)
    _ensure_dir(p)
    return p

def log_report(report: dict) -> str:
    dayp = _day_dir()
    jsonl = os.path.join(dayp, "reports.jsonl")
    with open(jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")

    csvp = os.path.join(dayp, "summary.csv")
    write_header = not os.path.exists(csvp)
    with open(csvp, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["ts","pnl","winrate","pf","maxdd","trades"])
        w.writerow([
            int(time.time()),
            f"{report.get('total_pnl',0.0):.6f}",
            f"{report.get('winrate',0.0):.4f}",
            "INF" if report.get("pf")==float("inf") else f"{report.get('pf',0.0):.4f}",
            f"{-report.get('max_dd',0.0):.6f}",
            report.get("trade_count",0)
        ])
    return jsonl

def recent_stats(hours: int = 6) -> dict:
    root = os.path.join("logs", "reports")
    if not os.path.isdir(root):
        return {"pnl":0.0,"trades":0,"wins":0,"pf":1.0,"winrate":0.0,"max_dd":0.0}

    cutoff = time.time() - hours*3600
    pnl = 0.0; gains = 0.0; losses = 0.0; trades = 0; wins = 0
    days = sorted(os.listdir(root))[-2:]
    for d in days:
        jsonl = os.path.join(root, d, "reports.jsonl")
        if not os.path.exists(jsonl): continue
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                ts = float(r.get("ts", 0))
                if ts < cutoff: continue
                pnl   += float(r.get("total_pnl", 0.0))
                trades+= int(r.get("trade_count", 0))
                wins  += int(r.get("wins", 0))
                gains += float(r.get("gains", 0.0))
                losses+= float(r.get("losses", 0.0))
    pf = (gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
    winrate = (wins/trades*100.0) if trades>0 else 0.0
    return {"pnl":pnl, "trades":trades, "wins":wins, "pf":pf, "winrate":winrate, "max_dd":0.0}
# MetaPatch: safe CSV writer (idempotent)
import os, csv, datetime
def _ensure_dir(p): 
    try: os.makedirs(p, exist_ok=True)
    except: pass
def log_csv(log_dir:str, row:dict, name:str="trades.csv"):
    try:
        _ensure_dir(log_dir)
        path = os.path.join(log_dir, name)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if new: w.writeheader()
            w.writerow(row)
    except Exception:
        pass
def write_trade_row(*, ts=None, slot="", sym="", pnl=0.0, extra=None):
    ts = ts or datetime.datetime.now().isoformat()
    row = {"ts": ts, "slot": slot, "sym": sym, "pnl": float(pnl), **(extra or {})}
    log_csv("logs", row, "trades.csv")


# MetaPatch: safe CSV writer (idempotent)
import os, csv, datetime
def _ensure_dir(p):
    try: os.makedirs(p, exist_ok=True)
    except: pass
def log_csv(log_dir:str, row:dict, name:str="trades.csv"):
    try:
        _ensure_dir(log_dir)
        path = os.path.join(log_dir, name)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if new: w.writeheader()
            w.writerow(row)
    except Exception:
        pass
def write_trade_row(*, ts=None, slot="", sym="", pnl=0.0, extra=None):
    ts = ts or datetime.datetime.now().isoformat()
    row = {"ts": ts, "slot": slot, "sym": sym, "pnl": float(pnl), **(extra or {})}
    log_csv("logs", row, "trades.csv")
