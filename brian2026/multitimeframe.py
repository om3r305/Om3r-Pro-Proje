from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .features import FeatureSnapshot

SUPPORTED_TIMEFRAMES = ("1m", "5m", "15m", "1h")


@dataclass(frozen=True, slots=True)
class MultiTimeframeRow:
    symbol: str
    decision_timestamp: float
    values: tuple[tuple[str, float | None], ...]
    available_timeframes: tuple[str, ...]
    missing_timeframes: tuple[str, ...]


def join_completed_timeframes(base: Iterable[FeatureSnapshot], higher: Iterable[FeatureSnapshot],
                              timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES) -> tuple[MultiTimeframeRow, ...]:
    history: dict[tuple[str, str], list[FeatureSnapshot]] = {}
    for snapshot in higher:
        available_at = snapshot.candle_timestamp
        if available_at is None:
            raise ValueError("higher-timeframe snapshot needs candle close timestamp")
        if available_at > snapshot.timestamp:
            raise ValueError("higher-timeframe candle is incomplete")
        history.setdefault((snapshot.symbol, snapshot.timeframe), []).append(snapshot)
    for values in history.values():
        values.sort(key=lambda s: (s.candle_timestamp or 0.0, s.timestamp))
    rows: list[MultiTimeframeRow] = []
    for decision in sorted(base, key=lambda s: (s.timestamp, s.symbol)):
        joined: dict[str, float | None] = {}
        available: list[str] = []
        for timeframe in timeframes:
            candidates = history.get((decision.symbol, timeframe), [])
            eligible = [s for s in candidates if s.candle_timestamp is not None and s.candle_timestamp <= decision.timestamp and s.timestamp <= decision.timestamp]
            selected = eligible[-1] if eligible else (decision if decision.timeframe == timeframe else None)
            if selected is None:
                continue
            available.append(timeframe)
            all_names = set(selected.unavailable_features()) | set(selected.available_features())
            for name in all_names:
                joined[f"{timeframe}__{name}"] = getattr(selected, name, None)
            joined[f"{timeframe}__price"] = selected.price
        missing = tuple(tf for tf in timeframes if tf not in available)
        rows.append(MultiTimeframeRow(decision.symbol, decision.timestamp, tuple(sorted(joined.items())),
                                      tuple(available), missing))
    return tuple(rows)