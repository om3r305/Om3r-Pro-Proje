# brain_supervisor.py — Brian Authority (feature/runbook/hot-patch/scaffold)
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.hot_patch import HotPatch
try:
    from core.utils_io import tg_send
except Exception:
    def tg_send(_msg: str): pass

ACTIONS = Path("brain_actions.json")
FEATURES = Path("runtime/features.json")

def _read_json(p: Path) -> Dict[str, Any]:
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}

def _write_json_atomic(p: Path, obj: Dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

def _append_actions(act: Dict[str, Any]):
    js = _read_json(ACTIONS) or {"ts": int(time.time()), "actions":[]}
    js["ts"] = int(time.time())
    js.setdefault("actions", []).append(act)
    js["actions"] = js["actions"][-96:]
    _write_json_atomic(ACTIONS, js)

class BrianSupervisor:
    def __init__(self, cfg_path: str = "config_live.json"):
        self.cfg_path = cfg_path
        self.hp = HotPatch()
        self._last_op: Dict[str, float] = {}
        self.cooldowns = {"low_activity_nudge": 1800, "soft_brake": 3600, "hard_brake": 7200}

        # runtime metrikleri (dashboard/brain_engine doldurabilir)
        self.pnl24 = 0.0
        self.trades24 = 0
        self.wins24 = 0

    # -------- feature toggle --------
    def feature_set(self, key: str, enabled: bool) -> bool:
        js = _read_json(FEATURES)
        js[key] = bool(enabled)
        _write_json_atomic(FEATURES, js)
        _append_actions({"type":"feature","key":key,"enabled":enabled})
        tg_send(f"🧠 Brian: feature `{key}` → {'ON' if enabled else 'OFF'}")
        return True

    # -------- patch ops (HotPatch) --------
    def apply_patch_ops(self, ops: List[Dict[str, Any]]):
        res = self.hp.apply(ops)
        _append_actions({"type":"patch","ops":ops})
        tg_send("🧠 Brian: HotPatch uygulandı.")
        return {"ok": True, "result": res}

    # -------- runbooks --------
    def runbook_news_spike(self, minutes: int=30, tighten: bool=True):
        # EV aç, entry’leri NEWS’e kaydır, süre bitince geri al (basit)
        delta = {
            "use_ev_filter": True,
            "ev_min": 0.02 if tighten else 0.0,
            "entry_frac": {"dip":0.2,"pred":0.2,"news":0.55,"ob":0.05}
        }
        self._cfg_delta(delta)
        _append_actions({"type":"runbook","name":"news_spike","minutes":minutes})
        tg_send(f"🧠 Brian: RUNBOOK news_spike ({minutes} dk)")
        return {"ok": True}

    def runbook_full_open(self):
        delta = {
            "use_ev_filter": False,
            "ev_min": 0.0,
            "entry_frac": {"dip":0.35,"pred":0.40,"news":0.20,"ob":0.05}
        }
        self._cfg_delta(delta)
        _append_actions({"type":"runbook","name":"full_open"})
        tg_send("🧠 Brian: RUNBOOK full_open")
        return {"ok": True}

    # -------- otomatik strateji scaffold --------
    def scaffold_strategy_rsi(self) -> bool:
        from pathlib import Path
        d = Path("strategies"); d.mkdir(exist_ok=True)
        (d/"__init__.py").write_text("# strategies package\n", encoding="utf-8")
        dst = d/"rsi_strategy.py"
        if dst.exists(): return False
        code = """from __future__ import annotations
from typing import Dict, Any, List
from core.strategy_api import StrategyPlugin, StrategySignal

def _rsi(closes: List[float], period: int=14) -> float:
    if len(closes) < period+1: return 50.0
    gains=losses=0.0
    for i in range(-period,0):
        ch = closes[i] - closes[i-1]
        if ch>0: gains += ch
        else: losses -= ch
    if losses <= 1e-12: return 70.0
    rs = gains / max(losses, 1e-12)
    return 100.0 - (100.0/(1.0+rs))

class RSIStrategy(StrategyPlugin):
    name = "rsi_oversold_rebound"
    slot = "pred"
    def __init__(self, cfg: Dict[str,Any]):
        super().__init__(cfg)
        s_cfg = (cfg.get("strategies") or {}).get("rsi", {})
        self.tf = s_cfg.get("tf", "5m")
        self.period = int(s_cfg.get("period", 14))
        self.buy_under = float(s_cfg.get("buy_under", 28.0))
        self.need_ema_support = bool(s_cfg.get("need_ema_support", True))
    def on_symbol(self, symbol: str, st, market) -> StrategySignal:
        candles = market.get_candles(symbol, self.tf, 200)
        closes = [c["close"] if isinstance(c, dict) else c[4] for c in candles] if candles else []
        if len(closes) < max(self.period+5, 30):
            return StrategySignal(symbol, self.slot, False, 0.0, "no_data")
        r = _rsi(closes, self.period)
        last = float(closes[-1])
        ema9 = st._last_reg.get("ema9")
        fired = (r <= self.buy_under)
        reason = f"RSI={r:.1f} <= {self.buy_under}"
        if self.need_ema_support and ema9 is not None:
            fired = fired and (last >= float(ema9))
            reason += f", ema9={'ok' if last>=float(ema9) else 'fail'}"
        conf = 0.55 if fired else 0.0
        return StrategySignal(symbol, self.slot, bool(fired), conf, reason)
"""
        dst.write_text(code, encoding="utf-8")
        try:
            cfg = _read_json(Path(self.cfg_path))
            sc = cfg.setdefault("strategies",{})
            sc.setdefault("rsi", {"tf":"5m","period":14,"buy_under":28.0,"need_ema_support": True})
            _write_json_atomic(Path(self.cfg_path), cfg)
        except Exception: pass
        _append_actions({"type":"scaffold","name":"rsi_strategy"})
        tg_send("🧠 Brian: RSI stratejisi otomatik oluşturuldu.")
        return True

    # -------- iç yardımcılar --------
    def _can_apply(self, key: str, cool_sec: int) -> bool:
        t = time.time()
        last = self._last_op.get(key, 0.0)
        if t - last < cool_sec: return False
        self._last_op[key] = t
        return True

    def _cfg_delta(self, delta: Dict[str, Any]) -> bool:
        cfgp = Path(self.cfg_path)
        js = _read_json(cfgp)
        # hızlı merge (sadece beklenen alanlar)
        if "use_ev_filter" in delta: js["use_ev_filter"] = bool(delta["use_ev_filter"])
        if "ev_min" in delta: js["ev_min"] = float(delta["ev_min"])
        if "entry_frac" in delta:
            js.setdefault("entry_frac", {}).update(delta["entry_frac"])
        if "freq_ctrl" in delta:
            js.setdefault("freq_ctrl", {}).update(delta["freq_ctrl"])
        if "rules" in delta:
            js.setdefault("rules", {}).setdefault("pct", {}).update(delta["rules"].get("pct",{}))
        _write_json_atomic(cfgp, js)
        return True
