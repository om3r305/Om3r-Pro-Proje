# -*- coding: utf-8 -*-
from __future__ import annotations
import time, json, random, importlib.util, sys
from pathlib import Path
from typing import Any, Dict, List

# --- sessiz telegram fallback ---
try:
    from Proje1.core.utils_io import tg_send
except Exception:
    def tg_send(*a, **k): pass

# Basit dosya yaz/log
BASE_DIR = Path("live/strategies")
BASE_DIR.mkdir(parents=True, exist_ok=True)
GEN_PATH  = BASE_DIR / "canavar_strat.py"
JOURNAL   = Path("logs/canavar_journal.jsonl")
JOURNAL.parent.mkdir(parents=True, exist_ok=True)

# “Kuralsız” mod ayar anahtarları:
DEFAULT_CFG = {
    "enabled": True,
    "tick_sec": 15,
    "write_to": str(GEN_PATH),
    "max_spawn_fail": 5,
    "no_guards": True,              # canavar mod: tüm guard’ları bypass et
    "open_limit_factor": 10.0,      # nakdin 10x’ine kadar sanal “kaldıraç” (tam kuralsız)
    "max_positions": 99,            # pratik limit; 0 = sınırsız (dosya/loop güvenliği için 99)
    "telemetry": True,
    "color_telegrams": True,
}

_STATE = {
    "last_tick": 0.0,
    "fail_spawns": 0,
    "loaded_epoch": 0,
    "loaded_hash": "",
}


def _cfg(d: Dict[str,Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _append_j(rec: Dict[str,Any]) -> None:
    rec = {"ts": time.time(), **rec}
    try:
        with JOURNAL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ----------------- Dinamik strateji runtime loader -----------------
def _load_dynamic_strategy(py_path: Path):
    """
    canavar_strat.py içinden create() fonksiyonunu yükler.
    create(cfg)-> Strategy nesnesi bekleriz.
    Strategy.run_for_symbol(symbol, st) -> list[dict] (plugin formatı)
    """
    if not py_path.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location("canavar_strat", str(py_path))
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        sys.modules["canavar_strat"] = mod          # type: ignore
        assert spec and spec.loader
        spec.loader.exec_module(mod)                # type: ignore
        if hasattr(mod, "create"):
            return mod.create
        return None
    except Exception as e:
        _append_j({"kind":"load_error", "err": str(e)})
        return None


# ----------------- Otomatik strateji üretimi (kuralsız) -----------------
SEED_SNIPPETS = [
    # Basit, agresif, momentum + mean-revert karışımı (kuralsız)
    """
import math, random

class Strategy:
    def __init__(self, cfg):
        self.cfg = cfg or {}
        self.bias = "AGGR"  # tamamen agresif

    def run_for_symbol(self, symbol, st):
        px = float(getattr(st, "last_px", 0.0) or 0.0)
        if px <= 0: 
            return []
        out = []
        # rastgelelik + mini momentum fikri:
        # reg/guard yok; sadece aç
        if random.random() < 0.30:
            slot = random.choice(["pred","dip","news","ob"])
            conf = min(1.0, max(0.35, random.random()*0.9))
            reason = f"CANAVAR:{self.bias}:{slot}"
            out.append({"slot": slot, "confidence": conf, "reason": reason})
        return out

def create(cfg):
    return Strategy(cfg)
""",
    # Daha yaratıcı varyant — sürü halinde pozisyon açma eğilimi
    """
import random

class Strategy:
    def __init__(self, cfg):
        self.cfg = cfg or {}
        self.p = self.cfg.get("p", 0.4)

    def run_for_symbol(self, symbol, st):
        px = float(getattr(st, "last_px", 0.0) or 0.0)
        if px <= 0: 
            return []
        out = []
        r = random.random()
        if r < self.p:
            # mini sürü: art arda 1-3 sinyal
            n = 1 + int(random.random()*3)
            for _ in range(n):
                slot = random.choice(["pred","dip","news","ob"])
                conf = min(1.0, max(0.30, random.random()))
                out.append({"slot": slot, "confidence": conf, "reason": f"SWARM:{slot}"})
        return out

def create(cfg):
    return Strategy(cfg)
""",
]


def _write_seed(path: Path):
    code = random.choice(SEED_SNIPPETS)
    path.write_text(code, encoding="utf-8")


# ----------------- Dış API: bot.py içinden çağrılacak -----------------
def ensure_canavar(cfg: Dict[str,Any]) -> None:
    c = dict(DEFAULT_CFG)
    c.update(_cfg(cfg, "canavar", {}) or {})
    p = Path(c.get("write_to") or str(GEN_PATH))
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        _write_seed(p)
        _append_j({"kind":"seed_written", "path": str(p)})


def canavar_tick(cfg: Dict[str,Any], ctx: Dict[str,Any], st) -> List[Dict[str,Any]]:
    """
    bot.run() içinde, sembol özelinde çağrılır.
    Kuralsız modda Strategy üretir/çağırır ve plugin çıktısı formatında sinyaller döndürür.
    """
    c = dict(DEFAULT_CFG)
    c.update(_cfg(cfg, "canavar", {}) or {})
    if not c.get("enabled", True):
        return []

    now = time.time()
    if (now - _STATE["last_tick"]) < max(1, int(c.get("tick_sec", 15))):
        # Sembol başına sıkı değil; genel throttling
        return []
    _STATE["last_tick"] = now

    py_path = Path(c.get("write_to") or str(GEN_PATH))
    if not py_path.exists():
        _write_seed(py_path)

    # %15 olasılıkla stratejiyi rastgele yeniden yazar (tam kuralsız evrim)
    if random.random() < 0.15:
        try:
            _write_seed(py_path)
            if c.get("telemetry"):
                try: tg_send("🧬 <b>CANAVAR</b>: Yeni strateji tohumlandı.", parse_mode="HTML")
                except Exception: pass
        except Exception:
            pass

    create_fn = _load_dynamic_strategy(py_path)
    if not create_fn:
        _STATE["fail_spawns"] += 1
        if _STATE["fail_spawns"] >= int(c.get("max_spawn_fail", 5)):
            _append_j({"kind":"fatal", "msg":"spawn_fail_limit"})
        return []

    _STATE["fail_spawns"] = 0
    try:
        strat = create_fn({"p": random.random()})
    except Exception as e:
        _append_j({"kind":"create_error", "err": str(e)})
        return []

    # Strategy.run_for_symbol → list[dict] (slot, confidence, reason)
    try:
        signals = strat.run_for_symbol(ctx.get("symbol"), st) or []
        # ‘kuralsız’ kipte limitleri gevşetildi diye logla
        if c.get("telemetry") and signals:
            try:
                txt = " ".join(f"{s.get('slot','?')}({s.get('confidence',0):.2f})" for s in signals[:4])
                tg_send(f"🟪 CANAVAR → {ctx.get('symbol')}: {txt}")
            except Exception: pass
        return signals
    except Exception as e:
        _append_j({"kind":"run_error", "err": str(e)})
        return []


def canavar_policy(cfg: Dict[str,Any]) -> Dict[str,Any]:
    """
    Kuralsız kipte bot’un iç politikalarını gevşetmek için rehber döndürür.
    bot.py bunu kullanıp guard’ları bypass edebilir.
    """
    c = dict(DEFAULT_CFG)
    c.update(_cfg(cfg, "canavar", {}) or {})
    if not c.get("enabled", True):
        return {"enabled": False}

    return {
        "enabled": True,
        "no_guards": bool(c.get("no_guards", True)),
        "open_limit_factor": float(c.get("open_limit_factor", 10.0)),
        "max_positions": int(c.get("max_positions", 99)),
    }
