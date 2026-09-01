from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .features import FeatureSnapshot

Label = Literal[-1, 0, 1]


@dataclass(frozen=True, slots=True)
class SupervisedSample:
    timestamp: float
    symbol: str
    features: tuple[tuple[str, float | None], ...]
    label: Label
    future_return: float
    target_timestamp: float
    dataset_id: str | None

    def feature_dict(self) -> dict[str, float | None]:
        return dict(self.features)


@dataclass(frozen=True, slots=True)
class TargetConfig:
    horizon: int = 1
    neutral_threshold_pct: float = 0.0


def build_samples(snapshots: Sequence[FeatureSnapshot], config: TargetConfig = TargetConfig()) -> tuple[SupervisedSample, ...]:
    if config.horizon <= 0 or config.neutral_threshold_pct < 0:
        raise ValueError("invalid target configuration")
    ordered = tuple(sorted(snapshots, key=lambda s: (s.symbol, s.timestamp)))
    by_symbol: dict[str, list[FeatureSnapshot]] = {}
    for snapshot in ordered:
        if snapshot.candle_timestamp is not None and snapshot.candle_timestamp > snapshot.timestamp:
            raise ValueError("future feature source")
        if any(ts > snapshot.timestamp for ts in snapshot.source_timestamps.values()):
            raise ValueError("future feature source")
        by_symbol.setdefault(snapshot.symbol, []).append(snapshot)
    result: list[SupervisedSample] = []
    for symbol, rows in sorted(by_symbol.items()):
        for index in range(len(rows) - config.horizon):
            current, future = rows[index], rows[index + config.horizon]
            ret = (future.price / current.price - 1.0) * 100.0
            threshold = config.neutral_threshold_pct
            label: Label = 1 if ret > threshold else (-1 if ret < -threshold else 0)
            values: dict[str, float | None] = {name: getattr(current, name) for name in current.unavailable_features()}
            values.update(current.available_features())
            values["price"] = current.price
            values["regime_code"] = float({"UNKNOWN": 0, "TREND": 1, "RANGE": 2, "VOLATILE": 3, "PANIC": 4}.get(current.regime.upper(), 0))
            values["legacy_signal_fired"] = (None if current.legacy_signal_fired is None else float(current.legacy_signal_fired))
            result.append(SupervisedSample(current.timestamp, symbol, tuple(sorted(values.items())),
                                           label, ret, future.timestamp, current.dataset_id))
    return tuple(sorted(result, key=lambda row: (row.timestamp, row.symbol)))