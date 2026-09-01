from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json

MARKET_EVENT_SCHEMA_VERSION = "brian.market-event.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ClosedCandle:
    open_timestamp: float
    close_timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.close_timestamp < self.open_timestamp:
            raise ValueError("candle closes before it opens")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("invalid OHLC range")
        if self.high < self.low or self.volume < 0:
            raise ValueError("invalid candle values")


@dataclass(frozen=True, slots=True)
class MarketEvent:
    symbol: str
    timeframe: str
    event_timestamp: float
    ingestion_timestamp: float
    candle: ClosedCandle
    bid: float | None = None
    ask: float | None = None
    regime: str = "UNKNOWN"
    order_book_sequence: int | None = None
    order_book_depth: int | None = None
    sources: tuple[tuple[str, str], ...] = ()
    schema_version: str = MARKET_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.symbol or not self.timeframe:
            raise ValueError("symbol and timeframe are required")
        if self.candle.close_timestamp > self.event_timestamp:
            raise ValueError("incomplete/future candle in market event")
        if self.ingestion_timestamp < self.event_timestamp:
            raise ValueError("ingestion precedes event availability")
        if (self.bid is None) != (self.ask is None):
            raise ValueError("bid and ask must be provided together")
        if self.bid is not None and (self.bid <= 0 or self.ask <= 0 or self.ask < self.bid):
            raise ValueError("invalid bid/ask")
        if self.schema_version != MARKET_EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported market event schema")

    @property
    def spread(self) -> float | None:
        return None if self.bid is None else self.ask - self.bid

    @property
    def spread_bps(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        mid = (self.bid + self.ask) / 2.0
        return (self.ask - self.bid) / mid * 10_000.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = dict(self.sources)
        data["spread"] = self.spread
        data["spread_bps"] = self.spread_bps
        return data

    @property
    def event_id(self) -> str:
        return sha256(_canonical(self.to_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class MarketDataset:
    events: tuple[MarketEvent, ...]
    schema_version: str = MARKET_EVENT_SCHEMA_VERSION
    metadata: tuple[tuple[str, str], ...] = ()
    dataset_id: str = field(init=False)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.events, key=lambda e: (e.event_timestamp, e.symbol, e.timeframe)))
        if ordered != self.events:
            raise ValueError("events must be deterministically ordered")
        payload = {
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
            "events": [event.to_dict() for event in self.events],
        }
        object.__setattr__(self, "dataset_id", sha256(_canonical(payload)).hexdigest())

    @classmethod
    def from_events(cls, events: Iterable[MarketEvent], **kwargs: Any) -> "MarketDataset":
        return cls(tuple(sorted(events, key=lambda e: (e.event_timestamp, e.symbol, e.timeframe))), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "metadata": dict(self.metadata),
            "events": [{"event_id": event.event_id, **event.to_dict()} for event in self.events],
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        content = _canonical(self.to_dict()) + b"\n"
        if target.exists():
            if target.read_bytes() != content:
                raise FileExistsError(f"immutable dataset already exists: {target}")
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target


def _timeframe_seconds(timeframe: str) -> int:
    unit = timeframe[-1:]
    value = int(timeframe[:-1])
    factors = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    if unit not in factors or value <= 0:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return value * factors[unit]


def from_legacy_closed_candles(
    *, symbol: str, timeframe: str, candles: Iterable[tuple],
    ingestion_timestamp: float, regime: str = "UNKNOWN",
    bid: float | None = None, ask: float | None = None,
) -> MarketDataset:
    """Create a hashed dataset from the exact closed-candle window observed."""
    rows = [tuple(row[:6]) for row in candles if len(row) >= 6]
    duration = _timeframe_seconds(timeframe)
    events: list[MarketEvent] = []
    for index, row in enumerate(rows):
        open_ts = float(row[0]) / 1000.0
        close_ts = open_ts + duration
        if close_ts > ingestion_timestamp:
            raise ValueError("incomplete candle supplied by legacy source")
        last = index == len(rows) - 1
        candle = ClosedCandle(open_ts, close_ts, float(row[1]), float(row[2]),
                              float(row[3]), float(row[4]), float(row[5]))
        events.append(MarketEvent(
            symbol=symbol, timeframe=timeframe, event_timestamp=close_ts,
            ingestion_timestamp=ingestion_timestamp, candle=candle,
            bid=bid if last else None, ask=ask if last else None,
            regime=regime if last else "UNKNOWN",
            sources=(("candles", "binance_spot_klines"),) +
                    ((("order_book", "binance_spot_depth"),) if last and bid is not None else ()),
        ))
    return MarketDataset.from_events(events, metadata=(("purpose", "brian_shadow_decision"),))
