from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from .metrics import PerformanceMetrics, calculate


@dataclass(slots=True)
class Candle:
    open: float
    high: float
    low: float
    close: float
    ts: float = 0.0


@dataclass(slots=True)
class ReplayConfig:
    tp_pct: float = 0.8
    sl_pct: float = 0.5
    entry_delay_bars: int = 0
    max_hold_bars: int = 30
    fee_bps_each_side: float = 10.0
    slippage_bps_each_side: float = 2.0

    def to_dict(self):
        return asdict(self)


def _cost_pct(cfg: ReplayConfig) -> float:
    return 2.0 * (cfg.fee_bps_each_side + cfg.slippage_bps_each_side) / 100.0


def simulate_long(candles: list[Candle], signal_index: int, cfg: ReplayConfig) -> float | None:
    i = signal_index + max(0, int(cfg.entry_delay_bars))
    if i >= len(candles) - 1:
        return None
    entry = float(candles[i].close)
    if entry <= 0:
        return None
    tp = entry * (1.0 + cfg.tp_pct / 100.0)
    sl = entry * (1.0 - cfg.sl_pct / 100.0)
    end = min(len(candles), i + 1 + max(1, int(cfg.max_hold_bars)))
    exit_px = float(candles[end - 1].close)
    for bar in candles[i + 1:end]:
        # Conservative same-bar rule: if TP and SL both hit, count SL first.
        if float(bar.low) <= sl:
            exit_px = sl
            break
        if float(bar.high) >= tp:
            exit_px = tp
            break
    raw_pct = (exit_px / entry - 1.0) * 100.0
    return raw_pct - _cost_pct(cfg)


def evaluate(candles: list[Candle], signal_indices: Iterable[int], cfg: ReplayConfig,
             notional_usd: float = 100.0) -> PerformanceMetrics:
    pnls: list[float] = []
    for idx in signal_indices:
        pct = simulate_long(candles, int(idx), cfg)
        if pct is not None:
            pnls.append(notional_usd * pct / 100.0)
    return calculate(pnls)


def deterministic_grid(base: ReplayConfig) -> list[ReplayConfig]:
    """Small bounded experiment grid; unlike the legacy arena, no random score exists."""
    out: list[ReplayConfig] = []
    for tp_mul in (0.8, 1.0, 1.2):
        for sl_mul in (0.8, 1.0, 1.2):
            for delay in (0, 1, 2):
                out.append(ReplayConfig(
                    tp_pct=max(0.05, base.tp_pct * tp_mul),
                    sl_pct=max(0.05, base.sl_pct * sl_mul),
                    entry_delay_bars=delay,
                    max_hold_bars=base.max_hold_bars,
                    fee_bps_each_side=base.fee_bps_each_side,
                    slippage_bps_each_side=base.slippage_bps_each_side,
                ))
    return out
