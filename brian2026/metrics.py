from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import statistics


@dataclass(slots=True)
class PerformanceMetrics:
    trades: int
    wins: int
    losses: int
    win_rate: float
    net_pnl: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    sharpe_like: float

    def to_dict(self):
        return asdict(self)


def calculate(pnls: list[float]) -> PerformanceMetrics:
    vals = [float(x) for x in pnls]
    n = len(vals)
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    net = sum(vals)
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 1e-12 else (float("inf") if gross_win > 0 else 0.0)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for x in vals:
        equity += x
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    if n >= 2:
        sd = statistics.pstdev(vals)
        sharpe = (statistics.fmean(vals) / sd) * math.sqrt(n) if sd > 1e-12 else 0.0
    else:
        sharpe = 0.0
    return PerformanceMetrics(
        trades=n,
        wins=len(wins),
        losses=len(losses),
        win_rate=(len(wins) / n if n else 0.0),
        net_pnl=net,
        expectancy=(net / n if n else 0.0),
        profit_factor=pf,
        max_drawdown=max_dd,
        sharpe_like=sharpe,
    )
