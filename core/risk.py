# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from typing import Optional, Dict, Any, List, Tuple

# ------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------
def _now() -> float:
    return time.time()

def _get_last_trade_ts(st, slot: str) -> float:
    """SymbolEngine içindeki slot-bazlı son trade zamanını güvenli oku."""
    try:
        m = getattr(st, "_last_trade_ts", {}) or {}
        return float(m.get(slot, 0.0))
    except Exception:
        return 0.0

# ------------------------------------------------------------
# Frekans kontrolleri (min_gap + max_trades_per_hour)
# ------------------------------------------------------------
def can_trade_now_ex(cfg: Dict[str, Any],
                     st=None,                # SymbolEngine (opsiyonel)
                     slot: str = "pred",     # "dip" | "pred" | "news" | "ob" ...
                     trade_hist: Optional[List[Tuple[float, str]]] = None,
                     cool_until: float = 0.0
                     ) -> Tuple[bool, str, float]:
    """Ayrıntılı sürüm: (ok, reason, next_ts) döner."""
    now = _now()

    # 0) Global cooldown (opsiyonel)
    if cool_until and now < float(cool_until):
        wait_s = int(float(cool_until) - now)
        return False, f"cooldown({wait_s}s)", float(cool_until)

    freq = (cfg.get("freq_ctrl") or {})
    min_gap = int(freq.get("min_sec_between_trades", 20))
    max_tph = int(freq.get("max_trades_per_hour", 60))

    # 1) Slot bazlı min gap
    last_ts = _get_last_trade_ts(st, slot) if st is not None else 0.0
    if last_ts > 0:
        gap = now - last_ts
        if gap < min_gap:
            wait_s = int(min_gap - gap)
            return False, f"min_gap_wait({wait_s}s)", now + wait_s

    # 2) Saatlik limit (global pencere)
    hist = trade_hist or []
    one_hour_ago = now - 3600.0
    recent = [ts for (ts, _sym) in hist if ts >= one_hour_ago]
    if len(recent) >= max_tph:
        oldest_recent = min(recent) if recent else now
        next_ok = oldest_recent + 3600.0
        wait_s = max(0, int(next_ok - now))
        return False, f"max_tph_reached({max_tph}) wait({wait_s}s)", next_ok

    return True, "ok", now

def can_trade_now(cfg: Dict[str, Any],
                  st=None,
                  dip=None,
                  **kwargs) -> bool:
    """
    Geriye dönük uyumlu sade API (bool).
    `dip` parametresi string ise slot olarak yorumlanır.
    """
    slot = kwargs.get("slot")
    if slot is None and isinstance(dip, str):
        slot = dip
    if slot is None:
        slot = "pred"

    trade_hist = kwargs.get("trade_hist")
    cool_until = float(kwargs.get("cool_until", 0.0))

    ok, _reason, _next_ts = can_trade_now_ex(cfg, st=st, slot=slot,
                                             trade_hist=trade_hist,
                                             cool_until=cool_until)
    return bool(ok)

# ------------------------------------------------------------
# Günlük risk guard (günlük zarar limiti + cooldown)
# ------------------------------------------------------------
class DailyRisk:
    """Günlük realized PnL takibi. Limit aşılınca cool-down başlatır."""
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.daily_cap: float   = float(cfg.get("daily_max_loss_usd", 10.0))   # USD
        self.cooldown_min: int  = int(cfg.get("cooldown_min", 30))             # dakika
        self._day0: str         = ""       # "YYYY-MM-DD"
        self.realized_today: float = 0.0
        self.cool_until: float  = 0.0      # epoch seconds

    def _today_str(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _rollover_if_new_day(self) -> None:
        today = self._today_str()
        if today != self._day0:
            self._day0 = today
            self.realized_today = 0.0
            self.cool_until = 0.0

    def on_realized(self, pnl: float) -> float:
        """Satış sonrası PnL'i ekler. Aşıldıysa cooldown başlatır."""
        self._rollover_if_new_day()
        self.realized_today += float(pnl)
        if self.realized_today <= -abs(self.daily_cap):
            self.cool_until = _now() + (self.cooldown_min * 60)
            return self.cool_until
        return 0.0

    def check(self) -> Tuple[bool, str, float]:
        """Manuel sorgu: şu an trade serbest mi?"""
        self._rollover_if_new_day()
        if self.cool_until and _now() < self.cool_until:
            wait_s = int(self.cool_until - _now())
            return False, f"risk_cooldown({wait_s}s)", self.cool_until
        return True, "ok", _now()
