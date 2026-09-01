from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .types import Decision, MarketSnapshot


@dataclass(slots=True)
class RiskConfig:
    min_trade_confidence: float = 0.70
    max_risk_per_trade_pct: float = 0.75
    daily_loss_limit_pct: float = 2.0
    max_drawdown_pct: float = 6.0
    max_open_positions: int = 3
    max_spread_bps: float = 18.0
    high_vol_atr_pct: float = 3.5
    high_vol_size_scale: float = 0.50


class RiskGovernor:
    """Independent hard gate. The learning brain cannot override these limits."""

    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self.cfg = cfg or RiskConfig()

    def review(self, decision: Decision, snapshot: MarketSnapshot,
               account: Dict[str, Any] | None = None) -> tuple[bool, float, str]:
        account = account or {}
        if decision.action == "WAIT":
            return False, 0.0, "meta trader abstained"
        if decision.confidence < self.cfg.min_trade_confidence:
            return False, 0.0, "below risk confidence floor"

        daily_pnl_pct = float(account.get("daily_pnl_pct", 0.0))
        drawdown_pct = abs(float(account.get("drawdown_pct", 0.0)))
        open_positions = int(account.get("open_positions", 0))
        if daily_pnl_pct <= -abs(self.cfg.daily_loss_limit_pct):
            return False, 0.0, "daily loss limit reached"
        if drawdown_pct >= abs(self.cfg.max_drawdown_pct):
            return False, 0.0, "max drawdown limit reached"
        if open_positions >= self.cfg.max_open_positions:
            return False, 0.0, "position limit reached"

        spread = float(snapshot.features.get("spread_bps", 0.0))
        if spread > self.cfg.max_spread_bps:
            return False, 0.0, f"spread too wide ({spread:.1f}bps)"

        atr_pct = abs(float(snapshot.features.get("atr_pct", 0.0)))
        size_scale = min(1.0, max(0.10, decision.confidence))
        if atr_pct >= self.cfg.high_vol_atr_pct:
            size_scale *= self.cfg.high_vol_size_scale

        # This is a scale, not leverage. Position sizing remains capped by caller.
        return True, max(0.0, min(1.0, size_scale)), "risk approved"
