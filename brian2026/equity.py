from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict
import time


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    exposure: float
    drawdown: float
    drawdown_pct: float


@dataclass(slots=True)
class ShadowPosition:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    last_price: float


class ShadowEquityTracker:
    """Accounting-only mirror of observed legacy positions; never executes."""

    def __init__(self, starting_equity: float) -> None:
        if starting_equity <= 0:
            raise ValueError("starting equity must be positive")
        self.starting_equity = float(starting_equity)
        self.realized_pnl = 0.0
        self.fees = 0.0
        self.peak_equity = float(starting_equity)
        self.max_drawdown = 0.0
        self.positions: Dict[str, ShadowPosition] = {}
        self.curve: list[EquityPoint] = []
        self._snapshot(time.time())

    def open_position(self, key: str, symbol: str, side: str, entry_price: float,
                      quantity: float, timestamp: float | None = None) -> None:
        if key in self.positions:
            raise ValueError(f"position already open: {key}")
        if entry_price <= 0 or quantity < 0 or side not in {"LONG", "SHORT"}:
            raise ValueError("invalid shadow position")
        self.positions[key] = ShadowPosition(symbol, side, float(entry_price),
                                             float(quantity), float(entry_price))
        self._snapshot(timestamp or time.time())

    def mark_price(self, symbol: str, price: float, timestamp: float | None = None) -> None:
        if price <= 0:
            return
        for position in self.positions.values():
            if position.symbol == symbol:
                position.last_price = float(price)
        self._snapshot(timestamp or time.time())

    def close_position(self, key: str, realized_pnl: float, fees: float = 0.0,
                       timestamp: float | None = None) -> None:
        self.positions.pop(key, None)
        self.realized_pnl += float(realized_pnl)
        self.fees += max(0.0, float(fees))
        self._snapshot(timestamp or time.time())

    @property
    def unrealized_pnl(self) -> float:
        total = 0.0
        for p in self.positions.values():
            direction = 1.0 if p.side == "LONG" else -1.0
            total += (p.last_price - p.entry_price) * p.quantity * direction
        return total

    @property
    def exposure(self) -> float:
        return sum(abs(p.last_price * p.quantity) for p in self.positions.values())

    @property
    def equity(self) -> float:
        return self.starting_equity + self.realized_pnl + self.unrealized_pnl - self.fees

    def _snapshot(self, timestamp: float) -> EquityPoint:
        equity = self.equity
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = self.peak_equity - equity
        self.max_drawdown = max(self.max_drawdown, drawdown)
        point = EquityPoint(
            timestamp=float(timestamp), equity=equity, realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl, fees=self.fees, exposure=self.exposure,
            drawdown=drawdown,
            drawdown_pct=(drawdown / self.peak_equity * 100.0 if self.peak_equity else 0.0),
        )
        self.curve.append(point)
        return point

    def account_state(self) -> dict[str, float | int]:
        point = self.curve[-1]
        return {
            "starting_equity": self.starting_equity,
            "equity": point.equity,
            "realized_pnl": point.realized_pnl,
            "unrealized_pnl": point.unrealized_pnl,
            "fees": point.fees,
            "exposure": point.exposure,
            "peak_equity": self.peak_equity,
            "drawdown": point.drawdown,
            "drawdown_pct": point.drawdown_pct,
            "max_drawdown": self.max_drawdown,
            "open_positions": len(self.positions),
            "daily_pnl_pct": self.realized_pnl / self.starting_equity * 100.0,
        }
