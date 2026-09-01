from __future__ import annotations

from typing import Any, Iterable
import time


def completed_klines(rows: Iterable[Any], now_ms: int | None = None) -> list[list[Any]]:
    """Keep only exchange klines whose declared close time has passed."""
    cutoff = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return [
        row for row in rows
        if isinstance(row, list) and len(row) >= 7 and int(row[6]) <= cutoff
    ]


def prior_high(highs: Iterable[float], lookback: int = 15) -> float | None:
    """Highest value before the observation being evaluated."""
    values = [float(value) for value in highs]
    prior = values[-(max(1, int(lookback)) + 1):-1]
    return max(prior) if prior else None
