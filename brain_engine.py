# brain_engine.py — Parametre Tuner (whitelist), hot-apply + TG bildirim
from __future__ import annotations
import json, time, os
from typing import Dict, Any, List, Tuple, Optional

ACTIONS_FILE = "brain_actions.json"

def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_json_atomic(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + f".{int(time.time())}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _append_action(act: Dict[str, Any]) -> None:
    try:
        data = {"ts": int(time.time()), "actions": []}
        if os.path.exists(ACTIONS_FILE):
            data = _read_json(ACTIONS_FILE)
        data["ts"] = int(time.time())
        data.setdefault("actions", []).append(act)
        data["actions"] = data["actions"][-96:]
        _write_json_atomic(ACTIONS_FILE, data)
    except Exception:
        pass

# Telegram (opsiyonel)
try:
    from core.utils_io import tg_send
except Exception:
    def tg_send(_msg: str):
        pass

class Brain:
    """
    Basit ama sağlam tuner:
      - Kötü gidişte: EV aç + ev_min↑ + trade freq↓ + TP↓ SL (daha korumacı) + slot ağırlıkları ↓
      - İyi gidişte: EV gevşet + freq↑ + TP↑ SL (daha geniş) + slot ağırlıkları ↑
      - Hot-apply: config_live.json'a atomik yazar (bot 5sn içinde alır)
    """
    def __init__(self, cfg_path: str = "config_live.json", telegram: bool = True):
        self.cfg_path = cfg_path
        self.telegram = telegram
        self.last_apply_ts: float = 0.0
        self.min_gap_sec: int = 30  # en az 30 sn arayla değiştir
        self.history: List[Tuple[float, Dict[str, Any]]] = []

        # whitelist yollar
        self.whitelist_paths = [
            ("use_ev_filter",),
            ("ev_min",),
            ("freq_ctrl","min_sec_between_trades"),
            ("freq_ctrl","max_trades_per_hour"),
            ("max_total_open_positions",),
            ("max_open_per_symbol",),
            ("entry_frac","dip"),
            ("entry_frac","pred"),
            ("entry_frac","news"),
            ("entry_frac","ob"),
            ("rules","pct","tp"),
            ("rules","pct","sl"),
            ("rules","pct","offset"),
            ("orderbook","enabled"),
            ("candles","exit_conf_thr"),
            ("dip","require_new_dip_after_start"),
        ]

    # ---- utils ----
    def _get(self, cfg: Dict[str, Any], path: Tuple[str, ...], default=None):
        cur = cfg
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    def _set(self, cfg: Dict[str, Any], path: Tuple[str, ...], value):
        cur = cfg
        for p in path[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[path[-1]] = value

    def _apply_delta(self, cfg: Dict[str, Any], deltas: Dict[Tuple[str,...], Any]):
        for path, value in deltas.items():
            if path not in self.whitelist_paths:
                continue
            self._set(cfg, path, value)

    def _hot_apply(self, deltas: Dict[Tuple[str,...], Any]) -> bool:
        now = time.time()
        if now - self.last_apply_ts < self.min_gap_sec:
            return False
        cfg = _read_json(self.cfg_path)
        if not cfg:
            return False
        before = {p: self._get(cfg, p) for p in self.whitelist_paths}
        self._apply_delta(cfg, deltas)
        after  = {p: self._get(cfg, p) for p in self.whitelist_paths}
        if before == after:
            return False
        _write_json_atomic(self.cfg_path, cfg)
        self.last_apply_ts = now
        # log
        pretty = { ".".join(p): after[p] for p in after if before.get(p)!=after[p] }
        _append_action({"type":"tuner","changed":pretty})
        if self.telegram:
            try:
                items = [f"{k} → {v}" for k,v in pretty.items()]
                tg_send("🧠 Brian (tuner): config güncellendi\n" + "\n".join(f"• {x}" for x in items))
            except Exception:
                pass
        return True

    # ---- public ----
    def reply(self, msg: str) -> str:
        m = msg.strip().lower()
        if m in ("durum","status","state"):
            cfg = _read_json(self.cfg_path)
            return f"Config OK. Hot-apply hazır. Whitelist {len(self.whitelist_paths)} anahtar."
        if m.startswith("set "):
            # ör: set freq_ctrl.min_sec_between_trades=30
            try:
                body = m[4:].strip()
                key, val = body.split("=",1)
                key = key.strip()
                val = val.strip()
                # sayı dönüştür
                if val.lower() in ("true","false"):
                    v = (val.lower()=="true")
                else:
                    try: v = float(val) if "." in val else int(val)
                    except: v = val
                parts = tuple(key.split("."))
                self._hot_apply({parts: v})
                return f"OK set {key}={v}"
            except Exception as e:
                return f"hata: {e}"
        return "Komutlar: 'durum', 'set x.y=val'"

    def tick(self, runtime_cfg: Dict[str, Any]) -> None:
        """
        runtime_cfg → bot’un RAM’deki değerleri; raporlardan trend çıkarıp ayar sık/gevşet.
        Beklenenler (opsiyonel): runtime_cfg['metrics24h'] = {'pnl': float, 'n': int, 'wins': int}
        """
        m = (runtime_cfg or {}).get("metrics24h") or {}
        pnl = float(m.get("pnl", 0.0))
        n   = int(m.get("n", 0))
        wins= int(m.get("wins", 0))
        # kötü faz
        if pnl <= -18 and n >= 4:
            deltas = {
                ("use_ev_filter",): True,
                ("ev_min",): 0.02,
                ("freq_ctrl","min_sec_between_trades"): max(20, int(self._guarded(runtime_cfg, ("freq_ctrl","min_sec_between_trades"), 35))),
                ("rules","pct","tp"): 0.8,
                ("rules","pct","sl"): -0.8,
                ("entry_frac","pred"): 0.30,
                ("entry_frac","dip"):  0.30,
                ("entry_frac","news"): 0.30,
                ("entry_frac","ob"):   0.10,
            }
            self._hot_apply(deltas)
            return
        # sert kötü faz
        if pnl <= -40 and n >= 6:
            deltas = {
                ("use_ev_filter",): True,
                ("ev_min",): 0.03,
                ("freq_ctrl","min_sec_between_trades"): max(25, int(self._guarded(runtime_cfg, ("freq_ctrl","min_sec_between_trades"), 45))),
                ("rules","pct","tp"): 0.6,
                ("rules","pct","sl"): -0.9,
                ("entry_frac","pred"): 0.20,
                ("entry_frac","dip"):  0.25,
                ("entry_frac","news"): 0.45,
                ("entry_frac","ob"):   0.10,
            }
            self._hot_apply(deltas)
            return
        # iyi faz
        if pnl >= 15 and n >= 6:
            deltas = {
                ("use_ev_filter",): False,
                ("ev_min",): 0.0,
                ("freq_ctrl","min_sec_between_trades"): max(10, int(self._guarded(runtime_cfg, ("freq_ctrl","min_sec_between_trades"), 22))),
                ("rules","pct","tp"): 1.2,
                ("rules","pct","sl"): -0.8,
                ("entry_frac","pred"): 0.40,
                ("entry_frac","dip"):  0.35,
                ("entry_frac","news"): 0.20,
                ("entry_frac","ob"):   0.05,
            }
            self._hot_apply(deltas)
            return
        # nötr faz: küçük düzeltmeler yok

    def _guarded(self, runtime_cfg: Dict[str, Any], path: Tuple[str,...], default: float) -> float:
        try:
            cur = runtime_cfg
            for p in path:
                cur = cur[p]
            v = float(cur)
            return v
        except Exception:
            return float(default)
