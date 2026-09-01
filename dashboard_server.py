# dashboard_server.py — Monster Coins Pro (stable + start/stop with logs)
from __future__ import annotations
import os, sys, csv, json, time, signal, subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Body
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# -------------------- Yol konfig --------------------
BASE_DIR    = Path(__file__).resolve().parent          # Proje kökü (Proje1)
PUBLIC_DIR  = BASE_DIR / "public"
LOGS_DIR    = BASE_DIR / "logs"
RUNTIME_DIR = BASE_DIR / "runtime"

EVENTS_CSV  = LOGS_DIR / "events.csv"
TRADES_CSV  = LOGS_DIR / "trades.csv"
BRAIN_JSONL = LOGS_DIR / "brain_log.jsonl"
STATE_JSON  = RUNTIME_DIR / "state.json"
PID_FILE    = BASE_DIR / "run_bot.pid"

DEFAULT_CONFIG = "config_live.json"  # main.py ile aynı klasörde olmalı

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RUNTIME_DIR, exist_ok=True)
os.makedirs(PUBLIC_DIR, exist_ok=True)

# -------------------- Şifre --------------------
DASH_PASS = os.getenv("DASH_PASS", "monster")

# -------------------- App --------------------
app = FastAPI(title="Monster Coins Pro — Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# /static mount (opsiyonel)
static_dir = PUBLIC_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# -------------------- Util --------------------
def _today_key() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())

def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists(): return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    except Exception:
        return []
    return rows

def _parse_float(x: Any, default: float = 0.0) -> float:
    try: return float(x)
    except Exception: return default

def _safe_symbol(p: str) -> str:
    return "".join(ch for ch in (p or "") if ch.isalnum() or ch in ("_", "-", "/"))

def _write_pid(pid: int) -> None:
    PID_FILE.write_text(str(pid), encoding="utf-8")

def _read_pid() -> Optional[int]:
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        return pid if pid > 0 else None
    except Exception:
        return None

def _kill_pid(pid: int) -> bool:
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        else:
            os.kill(pid, signal.SIGTERM); return True
    except Exception:
        return False

def _bot_running() -> bool:
    pid = _read_pid()
    if not pid: return False
    if sys.platform.startswith("win"):
        try:
            out = subprocess.check_output(["tasklist"], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return str(pid).encode() in out
        except Exception:
            return True
    else:
        try: os.kill(pid, 0); return True
        except Exception: return False

def _tail_text(path: Path, n: int = 200) -> str:
    if not path.exists(): return ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return ""

def _tail_jsonl(path: Path, limit: int) -> List[dict]:
    if not path.exists(): return []
    out: List[dict] = []
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

# -------------------- Sayfa --------------------
@app.get("/dashboard")
def dashboard() -> Response:
    html = PUBLIC_DIR / "monster.html"
    if html.exists():
        return FileResponse(str(html))
    return PlainTextResponse("monster.html bulunamadı (public/monster.html)", status_code=404)

@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)

# -------------------- Yardım ucu --------------------
@app.get("/api/where")
def api_where():
    return {
        "root_dir": str(BASE_DIR),
        "public": str(PUBLIC_DIR),
        "public_exists": PUBLIC_DIR.exists(),
        "monster_path": str(PUBLIC_DIR / "monster.html"),
        "monster_exists": (PUBLIC_DIR / "monster.html").exists(),
        "logs": str(LOGS_DIR),
        "runtime": str(RUNTIME_DIR),
        "pid_file": str(PID_FILE),
    }

# -------------------- Auth --------------------
@app.post("/api/login")
def api_login(payload: Dict[str, Any] = Body(...)):
    return {"ok": str(payload.get("password","")) == DASH_PASS}

@app.post("/api/logout")
def api_logout():
    try:
        if _bot_running():
            pid = _read_pid()
            if pid: _kill_pid(pid)
    except Exception:
        pass
    return {"ok": True}

# -------------------- Bot kontrol --------------------
@app.post("/api/start")
def api_start():
    # PID varsa ama process yoksa temizlik
    pid = _read_pid()
    if pid and not _bot_running():
        try: PID_FILE.unlink(missing_ok=True)
        except Exception: pass

    if _bot_running():
        return {"ok": True, "running": True, "msg": "Zaten çalışıyor", "pid": _read_pid()}

    py  = sys.executable
    cmd = [py, "-u", "main.py", "--config", DEFAULT_CONFIG]
    cwd = str(BASE_DIR)

    # --- DEBUG MOD: log dosyasına değil, bu terminale yazsın ---
    # Windows/uvicorn-reload bazen file handle ile takılıyor; önce hatayı çıplak görelim.
    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=None,   # <-- parent konsola akar
            stderr=None,   # <-- parent konsola akar
            creationflags=creation
        )
        _write_pid(proc.pid)
        # 1 sn bekleyip çöktü mü bakalım
        time.sleep(1.0)
        exited = (proc.poll() is not None)
        return {
            "ok": not exited,
            "running": not exited,
            "pid": proc.pid,
            "cmd": " ".join(cmd),
            "cwd": cwd,
            "note": "Hata varsa dashboard_server terminalinde görünür."
        }
    except Exception as e:
        try: PID_FILE.unlink(missing_ok=True)
        except Exception: pass
        return JSONResponse({"ok": False, "error": str(e), "cmd": " ".join(cmd), "cwd": cwd}, status_code=500)

@app.post("/api/stop")
def api_stop():
    pid = _read_pid()
    if not pid:
        return {"ok": True, "running": False}
    _kill_pid(pid)
    try: PID_FILE.unlink(missing_ok=True)
    except Exception: pass
    return {"ok": True, "running": False}

@app.get("/api/botlog")
def api_botlog(n: int = 200, kind: str = "stderr"):
    path = LOGS_DIR / ("bot_stderr.log" if kind.lower().startswith("err") else "bot_stdout.log")
    txt = _tail_text(path, max(10, min(n, 2000))) or "(log boş)"
    return PlainTextResponse(txt)

# -------------------- Data Endpoints --------------------
@app.get("/api/status")
def api_status(v: int = 1):
    state = _read_json(STATE_JSON, {"cash": None, "positions": {}})
    cash = float(state.get("cash") or 0.0)

    open_cnt = 0
    for slots in (state.get("positions") or {}).values():
        for v2 in (slots or {}).values():
            if v2: open_cnt += 1

    rows = _read_csv(EVENTS_CSV)
    today = _today_key()
    today_pnl = 0.0
    by_symbol: Dict[str, float] = {}
    for r in rows:
        ts = str(r.get("ts") or r.get("time") or "")
        if today not in ts:  # kaba
            continue
        kind = str(r.get("kind","")).lower()
        if kind not in ("sell","close","exit","realize","realized"):
            continue
        sym = _safe_symbol(r.get("sym") or r.get("symbol") or "")
        pnl = _parse_float(r.get("pnl", 0.0))
        today_pnl += pnl
        if sym:
            by_symbol[sym] = by_symbol.get(sym, 0.0) + pnl

    wins = losses = 0
    if TRADES_CSV.exists():
        for r in _read_csv(TRADES_CSV):
            pnl = _parse_float(r.get("pnl", 0.0))
            if pnl > 0: wins += 1
            elif pnl < 0: losses += 1
    else:
        for r in rows:
            kind = str(r.get("kind","")).lower()
            if kind not in ("sell","close","exit","realize","realized"): continue
            pnl = _parse_float(r.get("pnl", 0.0))
            if pnl > 0: wins += 1
            elif pnl < 0: losses += 1
    tot = wins + losses
    winrate = round(100.0 * wins / tot, 1) if tot > 0 else 0.0

    best_sym, best_p = ("", 0.0)
    worst_sym, worst_p = ("", 0.0)
    if by_symbol:
        best_sym, best_p = max(by_symbol.items(), key=lambda x: x[1])
        worst_sym, worst_p = min(by_symbol.items(), key=lambda x: x[1])

    return {
        "cash": round(cash, 2),
        "open": int(open_cnt),
        "today_pnl": round(today_pnl, 2),
        "winrate": winrate,
        "best_today": {"sym": best_sym, "pnl": round(best_p, 2)},
        "worst_today": {"sym": worst_sym, "pnl": round(worst_p, 2)},
        "status": "RUNNING" if _bot_running() else "STOPPED",
        "v": v,
    }

@app.get("/api/timeseries")
def api_timeseries(limit: int = 180):
    state = _read_json(STATE_JSON, {"cash": None})
    cash = float(state.get("cash") or 0.0)
    equity = [cash for _ in range(max(1, limit))]

    conf: List[float] = []
    for rec in _tail_jsonl(BRAIN_JSONL, limit):
        try:
            c = float(rec.get("confidence", 0.0))
            conf.append(c)
        except Exception:
            pass
    if not conf:
        conf = [0.5 for _ in range(max(1, min(50, limit)))]
    return {"equity": equity[-limit:], "confidence": conf[-limit:]}

@app.get("/api/brainlog")
def api_brainlog(limit: int = 100):
    return {"items": _tail_jsonl(BRAIN_JSONL, max(1, limit))}

@app.get("/api/debug")
def api_debug():
    rows = _read_csv(EVENTS_CSV)
    today = _today_key()
    symbols_today: Dict[str, float] = {}
    for r in rows:
        if today not in str(r.get("ts","")): continue
        if str(r.get("kind","")).lower() not in ("sell","close","exit","realize","realized"):
            continue
        sym = _safe_symbol(r.get("sym") or r.get("symbol") or "")
        symbols_today[sym] = symbols_today.get(sym, 0.0) + _parse_float(r.get("pnl",0.0))

    data = {
        "paths": {
            "base_dir": str(BASE_DIR),
            "public": str(PUBLIC_DIR),
            "events": str(EVENTS_CSV),
            "trades": str(TRADES_CSV),
            "brain_jsonl": str(BRAIN_JSONL),
            "state": str(STATE_JSON),
            "pid_file": str(PID_FILE),
        },
        "metrics": {
            "cash": _parse_float(_read_json(STATE_JSON, {"cash": 0}).get("cash", 0)),
            "by_symbol_today": symbols_today,
            "winrate_hint": "status endpointinden hesaplanır",
        },
        "state_tail": _read_json(STATE_JSON, {}),
    }
    return JSONResponse(data)

# -------------------- Run --------------------
if __name__ == "__main__":
    # Windows + uvicorn reload ile daha stabil
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    import uvicorn
    port = int(os.getenv("DASH_PORT", "8000"))
    uvicorn.run("dashboard_server:app", host="0.0.0.0", port=port, reload=True)
