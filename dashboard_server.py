# dashboard_server.py — Monster Coins Pro / Brian Control Center
from __future__ import annotations
import os, sys, csv, json, time, signal, subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Body
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
LOGS_DIR = BASE_DIR / "logs"
RUNTIME_DIR = BASE_DIR / "runtime"
EVENTS_CSV = LOGS_DIR / "events.csv"
TRADES_CSV = LOGS_DIR / "trades.csv"
BRAIN_JSONL = LOGS_DIR / "brain_log.jsonl"
STATE_JSON = RUNTIME_DIR / "state.json"
PID_FILE = BASE_DIR / "run_bot.pid"
DEFAULT_CONFIG = "config_live.json"

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RUNTIME_DIR, exist_ok=True)
os.makedirs(PUBLIC_DIR, exist_ok=True)

DASH_PASS = os.getenv("DASH_PASS", "monster")
app = FastAPI(title="Monster Coins Pro — Brian Control Center")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Kept for backward compatibility if a public/static folder is added later.
static_dir = PUBLIC_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _today_key() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    except Exception:
        return []
    return rows


def _parse_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_symbol(value: str) -> str:
    return "".join(ch for ch in (value or "") if ch.isalnum() or ch in ("_", "-", "/"))


def _first(row: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


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
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False


def _bot_running() -> bool:
    pid = _read_pid()
    if not pid:
        return False
    if sys.platform.startswith("win"):
        try:
            out = subprocess.check_output(["tasklist"], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return str(pid).encode() in out
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _tail_text(path: Path, n: int = 200) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return ""


def _tail_jsonl(path: Path, limit: int) -> List[dict]:
    if not path.exists():
        return []
    out: List[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return []
    return out[-limit:] if limit > 0 else out


@app.get("/dashboard")
def dashboard() -> Response:
    html = PUBLIC_DIR / "monster.html"
    if not html.exists():
        return PlainTextResponse("public/monster.html bulunamadı", status_code=404)
    return FileResponse(str(html), headers={"Cache-Control": "no-store"})


@app.get("/logo.png")
def logo() -> Response:
    path = PUBLIC_DIR / "logo.png"
    return FileResponse(str(path), media_type="image/png") if path.exists() else Response(status_code=404)


@app.get("/favicon.ico")
def favicon() -> Response:
    path = PUBLIC_DIR / "logo.png"
    return FileResponse(str(path), media_type="image/png") if path.exists() else Response(status_code=204)


@app.get("/manifest.webmanifest")
def manifest() -> Response:
    path = PUBLIC_DIR / "manifest.webmanifest"
    return FileResponse(str(path), media_type="application/manifest+json", headers={"Cache-Control": "no-cache"}) if path.exists() else Response(status_code=404)


@app.get("/sw.js")
def service_worker() -> Response:
    path = PUBLIC_DIR / "sw.js"
    return FileResponse(str(path), media_type="application/javascript", headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"}) if path.exists() else Response(status_code=404)


@app.get("/api/where")
def api_where():
    return {"root_dir": str(BASE_DIR), "public": str(PUBLIC_DIR), "monster_exists": (PUBLIC_DIR / "monster.html").exists(), "logs": str(LOGS_DIR), "runtime": str(RUNTIME_DIR), "pid_file": str(PID_FILE)}


@app.post("/api/login")
def api_login(payload: Dict[str, Any] = Body(...)):
    return {"ok": str(payload.get("password", "")) == DASH_PASS}


@app.post("/api/logout")
def api_logout():
    # Logout must never stop the runtime. The previous dashboard coupled these two actions.
    return {"ok": True}


@app.post("/api/start")
def api_start():
    pid = _read_pid()
    if pid and not _bot_running():
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    if _bot_running():
        return {"ok": True, "running": True, "msg": "Zaten çalışıyor", "pid": _read_pid()}
    py = sys.executable
    cmd = [py, "-u", "main.py", "--config", DEFAULT_CONFIG]
    cwd = str(BASE_DIR)
    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=None, stderr=None, creationflags=creation)
        _write_pid(proc.pid)
        time.sleep(1.0)
        exited = proc.poll() is not None
        return {"ok": not exited, "running": not exited, "pid": proc.pid, "cmd": " ".join(cmd), "cwd": cwd}
    except Exception as exc:
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        return JSONResponse({"ok": False, "error": str(exc), "cmd": " ".join(cmd), "cwd": cwd}, status_code=500)


@app.post("/api/stop")
def api_stop():
    pid = _read_pid()
    if not pid:
        return {"ok": True, "running": False}
    _kill_pid(pid)
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return {"ok": True, "running": False}


@app.get("/api/botlog")
def api_botlog(n: int = 200, kind: str = "stderr"):
    path = LOGS_DIR / ("bot_stderr.log" if kind.lower().startswith("err") else "bot_stdout.log")
    return PlainTextResponse(_tail_text(path, max(10, min(n, 2000))) or "(log boş)")


@app.get("/api/status")
def api_status(v: int = 1):
    state = _read_json(STATE_JSON, {"cash": None, "positions": {}})
    cash = float(state.get("cash") or 0.0)
    open_cnt = sum(1 for slots in (state.get("positions") or {}).values() for value in (slots or {}).values() if value)
    rows = _read_csv(EVENTS_CSV)
    today = _today_key()
    today_pnl = 0.0
    by_symbol: Dict[str, float] = {}
    for row in rows:
        ts = str(row.get("ts") or row.get("time") or "")
        if today not in ts:
            continue
        kind = str(row.get("kind", "")).lower()
        if kind not in ("sell", "close", "exit", "realize", "realized"):
            continue
        sym = _safe_symbol(str(row.get("sym") or row.get("symbol") or ""))
        pnl = _parse_float(row.get("pnl", 0.0))
        today_pnl += pnl
        if sym:
            by_symbol[sym] = by_symbol.get(sym, 0.0) + pnl
    win_rows = _read_csv(TRADES_CSV) if TRADES_CSV.exists() else [row for row in rows if str(row.get("kind", "")).lower() in ("sell", "close", "exit", "realize", "realized")]
    wins = sum(1 for row in win_rows if _parse_float(row.get("pnl", 0.0)) > 0)
    losses = sum(1 for row in win_rows if _parse_float(row.get("pnl", 0.0)) < 0)
    total = wins + losses
    winrate = round(100.0 * wins / total, 1) if total else 0.0
    best_sym, best_p = ("", 0.0)
    worst_sym, worst_p = ("", 0.0)
    if by_symbol:
        best_sym, best_p = max(by_symbol.items(), key=lambda x: x[1])
        worst_sym, worst_p = min(by_symbol.items(), key=lambda x: x[1])
    running = _bot_running()
    return {"cash": round(cash, 2), "open": int(open_cnt), "today_pnl": round(today_pnl, 2), "winrate": winrate, "best_today": {"sym": best_sym, "pnl": round(best_p, 2)}, "worst_today": {"sym": worst_sym, "pnl": round(worst_p, 2)}, "status": "RUNNING" if running else "STOPPED", "running": running, "shadow_only": True, "live_execution": False, "v": v}


@app.get("/api/trades")
def api_trades(limit: int = 50):
    limit = max(1, min(limit, 500))
    rows = _read_csv(TRADES_CSV)
    if not rows:
        rows = [row for row in _read_csv(EVENTS_CSV) if str(row.get("kind", "")).lower() in ("sell", "close", "exit", "realize", "realized")]
    items: List[Dict[str, Any]] = []
    for row in rows[-limit:][::-1]:
        symbol = _safe_symbol(str(_first(row, "symbol", "sym", "asset")))
        side = str(_first(row, "side", "action", "kind", default="CLOSE")).upper()
        items.append({
            "time": str(_first(row, "time", "ts", "timestamp", "closed_at")),
            "side": side,
            "symbol": symbol,
            "entry": _first(row, "entry", "entry_price", "open_price", "price_in", default="—"),
            "exit": _first(row, "exit", "exit_price", "close_price", "price_out", "price", default="—"),
            "qty": _first(row, "qty", "quantity", "size", "amount", default="—"),
            "pnl": round(_parse_float(_first(row, "pnl", "realized_pnl", "profit", default=0.0)), 8),
            "status": str(_first(row, "status", default="closed")),
        })
    return {"items": items, "count": len(items), "shadow_only": True}


@app.get("/api/positions")
def api_positions():
    state = _read_json(STATE_JSON, {"positions": {}})
    items: List[Dict[str, Any]] = []
    for symbol, slots in (state.get("positions") or {}).items():
        if not isinstance(slots, dict):
            continue
        for slot, position in slots.items():
            if not position:
                continue
            data = position if isinstance(position, dict) else {"value": position}
            items.append({
                "symbol": _safe_symbol(str(symbol)),
                "slot": str(slot),
                "side": str(data.get("side") or data.get("direction") or slot),
                "entry": data.get("entry") or data.get("entry_price") or data.get("price") or "—",
                "qty": data.get("qty") or data.get("quantity") or data.get("size") or "—",
                "pnl": data.get("pnl") if "pnl" in data else None,
            })
    return {"items": items, "count": len(items), "shadow_only": True}


@app.get("/api/timeseries")
def api_timeseries(limit: int = 180):
    state = _read_json(STATE_JSON, {"cash": None})
    cash = float(state.get("cash") or 0.0)
    equity = [cash for _ in range(max(1, limit))]
    conf: List[float] = []
    for rec in _tail_jsonl(BRAIN_JSONL, limit):
        try:
            conf.append(float(rec.get("confidence", 0.0)))
        except Exception:
            pass
    if not conf:
        conf = [0.5 for _ in range(max(1, min(50, limit)))]
    return {"equity": equity[-limit:], "confidence": conf[-limit:]}


@app.get("/api/brainlog")
def api_brainlog(limit: int = 100):
    return {"items": _tail_jsonl(BRAIN_JSONL, max(1, min(limit, 1000)))}


@app.get("/api/debug")
def api_debug():
    return JSONResponse({"paths": {"base_dir": str(BASE_DIR), "public": str(PUBLIC_DIR), "events": str(EVENTS_CSV), "trades": str(TRADES_CSV), "brain_jsonl": str(BRAIN_JSONL), "state": str(STATE_JSON), "pid_file": str(PID_FILE)}, "state_tail": _read_json(STATE_JSON, {})})


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    import uvicorn
    port = int(os.getenv("DASH_PORT", "8000"))
    uvicorn.run("dashboard_server:app", host="0.0.0.0", port=port, reload=True)
