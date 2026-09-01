# strategies/macd_strategy.py
from __future__ import annotations
from typing import Dict, Any

class MACDStrategy:
    NAME = "MACD"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.fast = int(self.cfg.get("fast", 12))
        self.slow = int(self.cfg.get("slow", 26))
        self.sigp = int(self.cfg.get("signal_period", 9))

        self._ema_fast = None
        self._ema_slow = None
        self._macd = None
        self._signal_ema = None
        self._last_hist = None
        self._hist = None

        # Metot adıyla çakışma yok:
        self.signal_val = 0

        self._alpha_fast = 2 / (self.fast + 1.0)
        self._alpha_slow = 2 / (self.slow + 1.0)
        self._alpha_sig = 2 / (self.sigp + 1.0)

    def _ema(self, prev, price, alpha):
        return price if prev is None else (alpha * price + (1 - alpha) * prev)

    def update(self, price: float):
        p = float(price)
        self._ema_fast = self._ema(self._ema_fast, p, self._alpha_fast)
        self._ema_slow = self._ema(self._ema_slow, p, self._alpha_slow)
        if self._ema_fast is None or self._ema_slow is None:
            return
        self._macd = self._ema_fast - self._ema_slow
        self._signal_ema = self._ema(self._signal_ema, self._macd, self._alpha_sig)
        if self._signal_ema is None:
            return
        hist = self._macd - self._signal_ema
        self.signal_val = hist
        self._last_hist, self._hist = self._hist, hist

    def signal(self) -> Dict[str, Any]:
        if self._macd is None or self._signal_ema is None:
            return {"buy": False, "sell": False, "reason": "warming", "confidence": 0.0}
        prev, curr = self._last_hist, self._hist
        if prev is None or curr is None:
            return {"buy": False, "sell": False, "reason": "warming", "confidence": 0.0}

        buy = prev <= 0 and curr > 0
        sell = prev >= 0 and curr < 0
        conf = min(1.0, abs(curr) / 0.003)
        reason = "macd_cross_up" if buy else ("macd_cross_down" if sell else "macd_hold")
        return {"buy": bool(buy), "sell": bool(sell), "confidence": float(conf), "reason": reason}
