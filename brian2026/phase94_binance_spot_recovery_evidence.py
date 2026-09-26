from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence

import httpx

from .phase46_execution_simulator import LiquidityLevel, OrderBookSnapshot
from .phase56_pretrade_risk_engine import InstrumentRiskLimits
from .phase57_shadow_execution_cycle import ExecutionMarketInput

PHASE94_SCHEMA_VERSION = "brian.phase94-binance-spot-recovery-evidence.v1"

PUBLIC_MARKET_HOSTS = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api3.binance.com",
)
SAFE_ROUTES = frozenset({
    "/api/v3/depth",
    "/api/v3/exchangeInfo",
})
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}USDT$")


class BinanceSpotRecoveryEvidenceError(RuntimeError):
    pass


class BinanceSpotRecoveryRateLimitError(BinanceSpotRecoveryEvidenceError):
    pass


class BinanceSpotRecoveryTransportError(BinanceSpotRecoveryEvidenceError):
    pass


@dataclass(frozen=True, slots=True)
class BinanceSpotRecoveryAssetEvidence:
    asset_id: str
    market: ExecutionMarketInput
    risk_limits: InstrumentRiskLimits
    mark: float
    observed_at: float
    depth_last_update_id: str
    source_host: str
    exchange_status: str
    schema_version: str = PHASE94_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not _SYMBOL_RE.fullmatch(self.asset_id):
            raise ValueError("Phase94 supports uppercase Binance Spot *USDT symbols only")
        if not math.isfinite(self.mark) or self.mark <= 0:
            raise ValueError("recovery mark must be finite and positive")
        if not math.isfinite(self.observed_at):
            raise ValueError("observed_at must be finite")
        if not self.depth_last_update_id.isdigit():
            raise ValueError("depth_last_update_id must be an unsigned integer string")
        if self.exchange_status != "TRADING":
            raise ValueError("recovery symbol must be TRADING")
        if self.source_host not in PUBLIC_MARKET_HOSTS:
            raise ValueError("recovery evidence host is not allowlisted")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase94 recovery evidence must remain shadow-only")


@dataclass(frozen=True, slots=True)
class BinanceSpotRecoveryEvidenceBundle:
    assets: tuple[BinanceSpotRecoveryAssetEvidence, ...]
    schema_version: str = PHASE94_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        ids = [row.asset_id for row in self.assets]
        if ids != sorted(ids):
            raise ValueError("Phase94 evidence assets must be deterministically sorted")
        if len(ids) != len(set(ids)):
            raise ValueError("Phase94 evidence cannot contain duplicate assets")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase94 evidence bundle must remain shadow-only")

    @property
    def markets(self) -> dict[str, ExecutionMarketInput]:
        return {row.asset_id: row.market for row in self.assets}

    @property
    def risk_limits_by_asset(self) -> dict[str, InstrumentRiskLimits]:
        return {row.asset_id: row.risk_limits for row in self.assets}

    @property
    def marks(self) -> dict[str, float]:
        return {row.asset_id: row.mark for row in self.assets}


Fetcher = Callable[[str, str, Mapping[str, str], float], tuple[object, str, float]]


def _float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise BinanceSpotRecoveryEvidenceError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BinanceSpotRecoveryEvidenceError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise BinanceSpotRecoveryEvidenceError(f"{label} must be finite")
    if positive and result <= 0:
        raise BinanceSpotRecoveryEvidenceError(f"{label} must be positive")
    return result


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise BinanceSpotRecoveryEvidenceError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _levels(value: object, label: str, *, descending: bool) -> tuple[LiquidityLevel, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise BinanceSpotRecoveryEvidenceError(f"{label} must be a non-empty array")
    rows: list[LiquidityLevel] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise BinanceSpotRecoveryEvidenceError(
                f"{label}[{index}] must contain price and quantity"
            )
        price = _float(item[0], f"{label}[{index}].price", positive=True)
        quantity = _float(item[1], f"{label}[{index}].quantity", positive=True)
        rows.append(LiquidityLevel(price=price, quantity=quantity))

    for left, right in zip(rows, rows[1:]):
        if descending and left.price <= right.price:
            raise BinanceSpotRecoveryEvidenceError(
                f"{label} must be strictly descending"
            )
        if not descending and left.price >= right.price:
            raise BinanceSpotRecoveryEvidenceError(
                f"{label} must be strictly ascending"
            )
    return tuple(rows)


def _symbol_row(exchange: object, symbol: str) -> dict[str, object]:
    root = _mapping(exchange, "exchangeInfo")
    rows = root.get("symbols")
    if not isinstance(rows, list):
        raise BinanceSpotRecoveryEvidenceError("exchangeInfo.symbols must be an array")
    matches = [
        _mapping(row, f"exchangeInfo.symbols[{index}]")
        for index, row in enumerate(rows)
        if isinstance(row, Mapping) and str(row.get("symbol", "")) == symbol
    ]
    if len(matches) != 1:
        raise BinanceSpotRecoveryEvidenceError(
            f"exchangeInfo must contain exactly one requested symbol: {symbol}"
        )
    row = matches[0]
    if str(row.get("status", "")) != "TRADING":
        raise BinanceSpotRecoveryEvidenceError(f"{symbol} is not TRADING")
    if row.get("isSpotTradingAllowed") is False:
        raise BinanceSpotRecoveryEvidenceError(f"{symbol} Spot trading is disabled")
    return row


def _filter(row: Mapping[str, object], filter_type: str) -> dict[str, object] | None:
    filters = row.get("filters")
    if not isinstance(filters, list):
        raise BinanceSpotRecoveryEvidenceError("exchangeInfo filters must be an array")
    matches = [
        _mapping(value, f"filter:{filter_type}")
        for value in filters
        if isinstance(value, Mapping) and str(value.get("filterType", "")) == filter_type
    ]
    if len(matches) > 1:
        raise BinanceSpotRecoveryEvidenceError(
            f"exchangeInfo contains duplicate {filter_type}"
        )
    return None if not matches else matches[0]


def _rules(exchange: object, symbol: str) -> tuple[float, float, str]:
    row = _symbol_row(exchange, symbol)
    price_filter = _filter(row, "PRICE_FILTER")
    if price_filter is None:
        raise BinanceSpotRecoveryEvidenceError(f"{symbol} missing PRICE_FILTER")
    tick_size = _float(
        price_filter.get("tickSize"),
        f"{symbol}.PRICE_FILTER.tickSize",
        positive=True,
    )

    notional = _filter(row, "NOTIONAL")
    min_notional_filter = _filter(row, "MIN_NOTIONAL")
    chosen = notional or min_notional_filter
    if chosen is None:
        raise BinanceSpotRecoveryEvidenceError(
            f"{symbol} missing NOTIONAL/MIN_NOTIONAL"
        )
    min_notional = _float(
        chosen.get("minNotional"),
        f"{symbol}.{chosen.get('filterType')}.minNotional",
        positive=True,
    )
    return tick_size, min_notional, str(row.get("status"))


def _depth(
    payload: object,
    *,
    symbol: str,
    tick_size: float,
    received_at: float,
    max_spread_bps: float,
) -> tuple[ExecutionMarketInput, float, str]:
    row = _mapping(payload, f"{symbol}.depth")
    update_id = row.get("lastUpdateId")
    if isinstance(update_id, bool):
        raise BinanceSpotRecoveryEvidenceError("depth lastUpdateId must be integer/string")
    update_text = str(update_id)
    if not update_text.isdigit():
        raise BinanceSpotRecoveryEvidenceError("depth lastUpdateId must be unsigned integer")

    bids = _levels(row.get("bids"), f"{symbol}.bids", descending=True)
    asks = _levels(row.get("asks"), f"{symbol}.asks", descending=False)
    best_bid = bids[0].price
    best_ask = asks[0].price
    if best_bid >= best_ask:
        raise BinanceSpotRecoveryEvidenceError(f"{symbol} order book is crossed/locked")
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 10_000.0
    if spread_bps > max_spread_bps:
        raise BinanceSpotRecoveryEvidenceError(
            f"{symbol} spread {spread_bps:.4f} bps exceeds {max_spread_bps:.4f}"
        )

    snapshot = OrderBookSnapshot(
        timestamp=float(received_at),
        bids=bids,
        asks=asks,
    )
    return (
        ExecutionMarketInput(
            reference_price=mid,
            tick_size=tick_size,
            snapshots=(snapshot,),
        ),
        mid,
        update_text,
    )


class BinanceSpotRecoveryEvidenceProvider:
    """Public, read-only Binance Spot evidence adapter for shadow recovery.

    Only the market-data-only depth/exchangeInfo routes are allowed. No API key,
    account route, signed endpoint, order route or private execution surface is
    accepted or constructed here.
    """

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = 5.5,
        max_assets: int = 8,
        depth_limit: int = 100,
        max_spread_bps: float = 30.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_assets <= 0 or max_assets > 32:
            raise ValueError("max_assets must be in [1,32]")
        if depth_limit not in {5, 10, 20, 50, 100, 500, 1000, 5000}:
            raise ValueError("unsupported Binance depth limit")
        if not math.isfinite(max_spread_bps) or max_spread_bps <= 0:
            raise ValueError("max_spread_bps must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.max_assets = int(max_assets)
        self.depth_limit = int(depth_limit)
        self.max_spread_bps = float(max_spread_bps)
        self.clock = clock
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            trust_env=False,
        )

    def _get(self, path: str, params: Mapping[str, str]) -> tuple[object, str, float]:
        if path not in SAFE_ROUTES:
            raise BinanceSpotRecoveryEvidenceError(
                f"forbidden Binance recovery market route: {path}"
            )
        last_error: str | None = None
        for host in PUBLIC_MARKET_HOSTS:
            try:
                response = self._client.get(
                    host + path,
                    params=dict(params),
                    headers={
                        "accept": "application/json",
                        "user-agent": "Brian-Phase94-Shadow-Recovery/1",
                    },
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = type(exc).__name__
                continue

            if response.status_code in {418, 429}:
                retry_after = response.headers.get("retry-after")
                detail = f" retry_after={retry_after}" if retry_after else ""
                raise BinanceSpotRecoveryRateLimitError(
                    f"Binance rate limit HTTP {response.status_code}{detail}"
                )
            if 500 <= response.status_code < 600:
                last_error = f"HTTP_{response.status_code}"
                continue
            if response.status_code < 200 or response.status_code >= 300:
                raise BinanceSpotRecoveryTransportError(
                    f"Binance market data HTTP {response.status_code} for {path}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise BinanceSpotRecoveryTransportError(
                    f"Binance market data returned invalid JSON for {path}"
                ) from exc
            return payload, host, float(self.clock())

        raise BinanceSpotRecoveryTransportError(
            f"Binance market data unavailable for {path}: {last_error or 'unknown'}"
        )

    def collect(self, asset_ids: Sequence[str]) -> BinanceSpotRecoveryEvidenceBundle:
        normalized = sorted({str(asset).strip().upper() for asset in asset_ids})
        if not normalized:
            return BinanceSpotRecoveryEvidenceBundle(())
        if len(normalized) > self.max_assets:
            raise BinanceSpotRecoveryEvidenceError(
                f"requested {len(normalized)} assets exceeds Phase94 max {self.max_assets}"
            )
        for symbol in normalized:
            if not _SYMBOL_RE.fullmatch(symbol):
                raise BinanceSpotRecoveryEvidenceError(
                    f"unsupported recovery asset for Binance Spot evidence: {symbol}"
                )

        rows: list[BinanceSpotRecoveryAssetEvidence] = []
        for symbol in normalized:
            exchange, exchange_host, _ = self._get(
                "/api/v3/exchangeInfo",
                {"symbol": symbol},
            )
            tick_size, min_notional, status = _rules(exchange, symbol)

            depth, depth_host, received_at = self._get(
                "/api/v3/depth",
                {"symbol": symbol, "limit": str(self.depth_limit)},
            )
            market, mark, update_id = _depth(
                depth,
                symbol=symbol,
                tick_size=tick_size,
                received_at=received_at,
                max_spread_bps=self.max_spread_bps,
            )

            # Both calls are public-market data. Record the depth host because
            # it is the point-in-time book used for the recovery simulation.
            # Exchange-info may fail over independently without changing the
            # source identity of the actual execution snapshot.
            _ = exchange_host
            rows.append(BinanceSpotRecoveryAssetEvidence(
                asset_id=symbol,
                market=market,
                risk_limits=InstrumentRiskLimits(min_notional=min_notional),
                mark=mark,
                observed_at=received_at,
                depth_last_update_id=update_id,
                source_host=depth_host,
                exchange_status=status,
            ))
        return BinanceSpotRecoveryEvidenceBundle(tuple(rows))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "BinanceSpotRecoveryEvidenceProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
