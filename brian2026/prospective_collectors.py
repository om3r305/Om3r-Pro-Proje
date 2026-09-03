from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping
import time

import requests

from .intelligence_store import IntelligenceCapture, IntelligenceStore
from .universe_radar import (
    MarketUniverseRow,
    UniverseConfig,
    UniverseDelta,
    UniverseSnapshot,
    build_universe_snapshot,
    compare_universe,
)


JsonGetter = Callable[[str, float], Any]
Clock = Callable[[], float]


def requests_json_getter(url: str, timeout: float) -> Any:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


@dataclass(frozen=True, slots=True)
class CollectorCycle:
    snapshot: UniverseSnapshot
    delta: UniverseDelta
    capture_ids: tuple[str, ...]
    degraded_sources: tuple[str, ...]
    shadow_only: bool = True
    schema_version: str = "brian.prospective-collector-cycle.v1"


class BinancePublicUniverseCollector:
    """One prospective, unauthenticated Binance Spot universe observation.

    The collector has no API-key parameters and no order endpoints. Every raw
    provider response is timestamped *after* the response is received and can be
    written immediately to IntelligenceStore before any ranking logic is applied.
    Top-of-book is optional; its failure degrades spread quality rather than
    fabricating prices.
    """

    EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
    TICKER_24H = "https://api.binance.com/api/v3/ticker/24hr"
    BOOK_TICKER = "https://api.binance.com/api/v3/ticker/bookTicker"

    def __init__(
        self,
        *,
        store: IntelligenceStore | None = None,
        config: UniverseConfig = UniverseConfig(),
        getter: JsonGetter = requests_json_getter,
        clock: Clock = time.time,
        timeout: float = 8.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("collector timeout must be positive")
        self.store = store
        self.config = config
        self.getter = getter
        self.clock = clock
        self.timeout = float(timeout)

    def _capture(self, record_type: str, payload: Any, observed_at: float, source_url: str) -> str | None:
        if self.store is None:
            return None
        capture = IntelligenceCapture(
            provider="binance_public",
            record_type=record_type,
            observed_at=float(observed_at),
            captured_at=float(observed_at),
            payload={"response": payload},
            provenance_uri=source_url,
        )
        self.store.put(capture)
        return capture.capture_id

    @staticmethod
    def _index_rows(payload: Any, *, key: str = "symbol") -> Mapping[str, Mapping[str, Any]]:
        if not isinstance(payload, list):
            raise ValueError("expected Binance array response")
        out: dict[str, Mapping[str, Any]] = {}
        for row in payload:
            if isinstance(row, Mapping) and str(row.get(key, "")).strip():
                out[str(row[key])] = row
        return out

    def collect(self, previous: UniverseSnapshot | None = None) -> CollectorCycle:
        captures: list[str] = []

        exchange_payload = self.getter(self.EXCHANGE_INFO, self.timeout)
        exchange_observed_at = float(self.clock())
        capture_id = self._capture(
            "exchange_info", exchange_payload, exchange_observed_at, self.EXCHANGE_INFO
        )
        if capture_id is not None:
            captures.append(capture_id)

        ticker_payload = self.getter(self.TICKER_24H, self.timeout)
        ticker_observed_at = float(self.clock())
        capture_id = self._capture("ticker_24h", ticker_payload, ticker_observed_at, self.TICKER_24H)
        if capture_id is not None:
            captures.append(capture_id)

        degraded: list[str] = []
        book_payload: Any = []
        book_observed_at: float | None = None
        try:
            book_payload = self.getter(self.BOOK_TICKER, self.timeout)
            book_observed_at = float(self.clock())
            capture_id = self._capture(
                "book_ticker", book_payload, book_observed_at, self.BOOK_TICKER
            )
            if capture_id is not None:
                captures.append(capture_id)
        except Exception:
            degraded.append("book_ticker")
            book_payload = []
            # Timestamp the completed degraded cycle after the optional call failed.
            book_observed_at = float(self.clock())

        observed_at = max(exchange_observed_at, ticker_observed_at, float(book_observed_at))

        if not isinstance(exchange_payload, Mapping) or not isinstance(exchange_payload.get("symbols"), list):
            raise ValueError("invalid Binance exchangeInfo response")
        ticker_by_symbol = self._index_rows(ticker_payload)
        book_by_symbol = self._index_rows(book_payload) if book_payload else {}

        rows: list[MarketUniverseRow] = []
        for metadata in exchange_payload["symbols"]:
            if not isinstance(metadata, Mapping):
                continue
            symbol = str(metadata.get("symbol", ""))
            ticker = ticker_by_symbol.get(symbol)
            if ticker is None:
                continue
            try:
                book = book_by_symbol.get(symbol, {})
                row = MarketUniverseRow(
                    symbol=symbol,
                    base_asset=str(metadata.get("baseAsset", "")),
                    quote_asset=str(metadata.get("quoteAsset", "")),
                    last_price=float(ticker.get("lastPrice", 0.0)),
                    quote_volume=float(ticker.get("quoteVolume", 0.0)),
                    trades_24h=int(ticker.get("count", 0)),
                    price_change_pct=float(ticker.get("priceChangePercent", 0.0)),
                    high_price=float(ticker.get("highPrice", 0.0)),
                    low_price=float(ticker.get("lowPrice", 0.0)),
                    bid_price=(
                        float(book["bidPrice"])
                        if book and float(book.get("bidPrice", 0.0)) > 0 else None
                    ),
                    ask_price=(
                        float(book["askPrice"])
                        if book and float(book.get("askPrice", 0.0)) > 0 else None
                    ),
                    spot_trading_allowed=(
                        str(metadata.get("status", "")) == "TRADING" and
                        metadata.get("isSpotTradingAllowed", True) is not False
                    ),
                )
            except (TypeError, ValueError, KeyError):
                continue
            rows.append(row)

        snapshot = build_universe_snapshot(rows, observed_at=observed_at, config=self.config)
        delta = compare_universe(previous, snapshot)
        return CollectorCycle(snapshot, delta, tuple(captures), tuple(degraded), True)
