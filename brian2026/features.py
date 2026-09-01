from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence
import math
import statistics
import time

from .types import MarketSnapshot

FEATURE_SCHEMA_VERSION = "brian.features.v2"
BRIAN_VERSION = "0.2.0"


@dataclass(slots=True)
class FeatureSnapshot:
    """Point-in-time feature contract for the shadow engine.

    Optional values are deliberately preserved as unavailable.  ``to_market``
    only emits observed/calculated numeric features; it never manufactures a
    neutral value for missing data.
    """

    symbol: str
    timestamp: float
    price: float
    regime: str
    timeframe: str = "1m"
    candle_timestamp: float | None = None
    ema_fast: float | None = None
    ema_slow: float | None = None
    ema_slope_pct: float | None = None
    rsi: float | None = None
    return_5: float | None = None
    zscore: float | None = None
    bb_position: float | None = None
    atr_pct: float | None = None
    spread_bps: float | None = None
    book_imbalance: float | None = None
    wall_score: float | None = None
    breakout_score: float | None = None
    recent_high: float | None = None
    volume_z: float | None = None
    acceleration: float | None = None
    legacy_predictor_confidence: float | None = None
    legacy_signal_fired: bool | None = None
    legacy_slot: str | None = None
    sources: dict[str, str] = field(default_factory=dict)
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    source_timestamps: dict[str, float] = field(default_factory=dict)
    dataset_id: str | None = None
    brian_version: str = BRIAN_VERSION

    def __post_init__(self) -> None:
        if self.candle_timestamp is not None and self.candle_timestamp > self.timestamp:
            raise ValueError("future candle cannot enter a feature snapshot")
        future = [name for name, ts in self.source_timestamps.items() if float(ts) > self.timestamp]
        if future:
            raise ValueError(f"future sources cannot enter feature snapshot: {future}")

    def available_features(self) -> dict[str, float]:
        excluded = {
            "symbol", "timestamp", "price", "regime", "timeframe",
            "candle_timestamp", "legacy_signal_fired", "legacy_slot", "sources",
            "feature_schema_version", "source_timestamps", "dataset_id", "brian_version",
        }
        return {
            key: float(value)
            for key, value in asdict(self).items()
            if key not in excluded and value is not None
        }

    def unavailable_features(self) -> list[str]:
        candidates = (
            "ema_fast", "ema_slow", "ema_slope_pct", "rsi", "return_5",
            "zscore", "bb_position", "atr_pct", "spread_bps",
            "book_imbalance", "wall_score", "breakout_score", "recent_high",
            "volume_z", "acceleration", "legacy_predictor_confidence",
        )
        return [name for name in candidates if getattr(self, name) is None]

    def to_market(self) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=self.symbol,
            price=self.price,
            ts=self.timestamp,
            timeframe=self.timeframe,
            regime=self.regime,
            features=self.available_features(),
            context={
                "candle_timestamp": self.candle_timestamp,
                "legacy_signal_fired": self.legacy_signal_fired,
                "legacy_slot": self.legacy_slot,
                "sources": dict(self.sources),
                "source_timestamps": dict(self.source_timestamps),
                "dataset_id": self.dataset_id,
                "feature_schema_version": self.feature_schema_version,
                "feature_availability": {
                    name: name not in self.unavailable_features()
                    for name in self.unavailable_features() + list(self.available_features())
                },
                "brian_version": self.brian_version,
                "unavailable_features": self.unavailable_features(),
            },
        )


def _ema(values: Sequence[float], length: int) -> list[float]:
    alpha = 2.0 / (length + 1.0)
    out: list[float] = []
    for value in values:
        out.append(value if not out else out[-1] + alpha * (value - out[-1]))
    return out


def _rsi(values: Sequence[float], length: int = 14) -> float | None:
    if len(values) < length + 1:
        return None
    changes = [values[i] - values[i - 1] for i in range(len(values) - length, len(values))]
    gain = sum(max(change, 0.0) for change in changes) / length
    loss = sum(max(-change, 0.0) for change in changes) / length
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


def from_closed_candles(
    *, symbol: str, price: float, regime: str,
    candles: Iterable[Sequence[float]], timeframe: str = "1m",
    order_book: Mapping[str, float | None] | None = None,
    legacy_predictor_confidence: float | None = None,
    legacy_signal_fired: bool | None = None,
    legacy_slot: str | None = None,
    timestamp: float | None = None,
    dataset_id: str | None = None,
    source_timestamps: Mapping[str, float] | None = None,
) -> FeatureSnapshot:
    """Build features from oldest-to-newest *completed* OHLCV candles."""
    rows = [tuple(row[:6]) for row in candles if len(row) >= 6]
    snapshot = FeatureSnapshot(
        symbol=symbol, timestamp=float(timestamp or time.time()), price=float(price),
        regime=str(regime), timeframe=timeframe,
        legacy_predictor_confidence=legacy_predictor_confidence,
        legacy_signal_fired=legacy_signal_fired, legacy_slot=legacy_slot,
        sources={"candles": "binance_closed", "price": "binance_ticker"},
        dataset_id=dataset_id, source_timestamps=dict(source_timestamps or {}),
    )
    if order_book:
        for name in ("spread_bps", "book_imbalance", "wall_score"):
            value = order_book.get(name)
            if value is not None:
                setattr(snapshot, name, float(value))
        snapshot.sources["order_book"] = "binance_depth"
    if not rows:
        return snapshot

    closes = [float(row[4]) for row in rows]
    highs = [float(row[2]) for row in rows]
    lows = [float(row[3]) for row in rows]
    volumes = [float(row[5]) for row in rows]
    snapshot.candle_timestamp = float(
        snapshot.source_timestamps.get("closed_candle", float(rows[-1][0]) / 1000.0)
    )
    if snapshot.candle_timestamp > snapshot.timestamp:
        raise ValueError("future candle cannot enter a feature snapshot")
    snapshot.sources["technical_features"] = "completed_candles_only"

    if len(closes) >= 21:
        fast, slow = _ema(closes, 9), _ema(closes, 21)
        snapshot.ema_fast, snapshot.ema_slow = fast[-1], slow[-1]
        if len(fast) >= 4 and fast[-4] != 0:
            snapshot.ema_slope_pct = (fast[-1] / fast[-4] - 1.0) * 100.0
        sample = closes[-20:]
        mean, sd = statistics.fmean(sample), statistics.pstdev(sample)
        snapshot.zscore = (closes[-1] - mean) / sd if sd > 0 else 0.0
        lower, upper = mean - 2.0 * sd, mean + 2.0 * sd
        snapshot.bb_position = ((closes[-1] - lower) / (upper - lower)) if upper > lower else 0.5
        prior_volumes = volumes[-21:-1]
        vol_sd = statistics.pstdev(prior_volumes)
        snapshot.volume_z = ((volumes[-1] - statistics.fmean(prior_volumes)) / vol_sd
                             if vol_sd > 0 else 0.0)
        prior_highs = highs[-21:-1]
        snapshot.recent_high = max(prior_highs)
        trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                   abs(lows[i] - closes[i - 1])) for i in range(1, len(closes))]
        atr = statistics.fmean(trs[-14:])
        snapshot.atr_pct = atr / closes[-1] * 100.0 if closes[-1] else None
        snapshot.breakout_score = max(-1.0, min(1.0, (closes[-1] - snapshot.recent_high) / max(atr, 1e-12)))
    snapshot.rsi = _rsi(closes)
    if len(closes) >= 6 and closes[-6] != 0:
        snapshot.return_5 = (closes[-1] / closes[-6] - 1.0) * 100.0
    if len(closes) >= 3 and closes[-2] and closes[-3]:
        last_return = (closes[-1] / closes[-2] - 1.0) * 100.0
        prior_return = (closes[-2] / closes[-3] - 1.0) * 100.0
        snapshot.acceleration = last_return - prior_return
    return snapshot
