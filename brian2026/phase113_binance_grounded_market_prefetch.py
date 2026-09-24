from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence

import httpx

from .phase94_binance_spot_recovery_evidence import PUBLIC_MARKET_HOSTS
from .phase110_supabase_grounded_market_prefetch import (
    GroundedMarketPrefetch,
    GroundedPricePoint,
    SupabaseGroundedMarketPrefetchReader,
)

PHASE113_SCHEMA_VERSION = "brian.phase113-binance-grounded-market-prefetch.v1"
_SAFE_ROUTE = "/api/v3/klines"
_CRYPTO_ASSET = re.compile(r"^crypto:([A-Z0-9]{2,20}USDT)$")


class BinanceGroundedMarketPrefetchError(RuntimeError):
    pass


class BinanceGroundedMarketPrefetchRateLimitError(
    BinanceGroundedMarketPrefetchError
):
    pass


class BinanceGroundedMarketPrefetchTransportError(
    BinanceGroundedMarketPrefetchError
):
    pass


def _symbol(asset_id: str) -> str:
    text = str(asset_id).strip()
    match = _CRYPTO_ASSET.fullmatch(text)
    if match is None:
        raise BinanceGroundedMarketPrefetchError(
            f"Phase113 supports canonical crypto:*USDT assets only: {text!r}"
        )
    return match.group(1)


def _float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise BinanceGroundedMarketPrefetchError(
            f"{label} must be numeric"
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BinanceGroundedMarketPrefetchError(
            f"{label} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise BinanceGroundedMarketPrefetchError(
            f"{label} must be finite"
        )
    if positive and result <= 0:
        raise BinanceGroundedMarketPrefetchError(
            f"{label} must be positive"
        )
    return result


def _source_id(
    *,
    host: str,
    symbol: str,
    open_time_ms: int,
    close_time_ms: int,
    close: float,
) -> str:
    payload = (
        f"{PHASE113_SCHEMA_VERSION}|{host}|{symbol}|"
        f"{open_time_ms}|{close_time_ms}|{close:.16g}"
    )
    return "binance-kline:" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


class BinanceGroundedMarketPrefetchReader:
    """Public Binance completed-klines + Supabase grounded-sensor composition.

    This replaces sparse signal-triggered micro-book prices for crypto
    covariance history. Only public 5m klines are fetched, bounded by the
    decision timestamp. Supabase Phase110 still owns observation parsing,
    expert snapshot construction, bucket alignment and causal return checks.
    No account, signed, order or private Binance route exists here.
    """

    def __init__(
        self,
        *,
        sensor_reader: SupabaseGroundedMarketPrefetchReader,
        client: httpx.Client | None = None,
        timeout_seconds: float = 5.5,
        max_assets: int = 12,
        clock: Callable[[], float] = time.time,
        owns_sensor_reader: bool = False,
    ) -> None:
        if not callable(getattr(sensor_reader, "load_with_price_points", None)):
            raise TypeError(
                "sensor_reader must expose callable load_with_price_points"
            )
        if sensor_reader.config.bucket_seconds != 300:
            raise ValueError(
                "Phase113 requires Phase110 bucket_seconds=300 for 5m klines"
            )
        if sensor_reader.config.return_observations + 1 > 1000:
            raise ValueError(
                "Phase113 requires return_observations <= 999"
            )
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_assets <= 0 or max_assets > 32:
            raise ValueError("max_assets must be in [1,32]")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.sensor_reader = sensor_reader
        self.timeout_seconds = float(timeout_seconds)
        self.max_assets = int(max_assets)
        self.clock = clock
        self._owns_sensor_reader = bool(owns_sensor_reader)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            trust_env=False,
        )
        self._closed = False

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
        sensor_reader_factory=SupabaseGroundedMarketPrefetchReader.from_env,
        clock: Callable[[], float] = time.time,
    ) -> "BinanceGroundedMarketPrefetchReader":
        sensor_reader = None
        try:
            sensor_reader = sensor_reader_factory(env=env)
            return cls(
                sensor_reader=sensor_reader,
                client=client,
                clock=clock,
                owns_sensor_reader=True,
            )
        except Exception:
            close = getattr(sensor_reader, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def _get_klines(
        self,
        *,
        symbol: str,
        decision_timestamp: float,
    ) -> tuple[object, str]:
        if self._closed:
            raise BinanceGroundedMarketPrefetchError(
                "Phase113 reader is closed"
            )
        end_time_ms = max(0, int(decision_timestamp * 1000) - 1)
        limit = min(
            1000,
            self.sensor_reader.config.return_observations + 8,
        )
        params = {
            "symbol": symbol,
            "interval": "5m",
            "limit": str(limit),
            "endTime": str(end_time_ms),
        }
        last_error: str | None = None
        for host in PUBLIC_MARKET_HOSTS:
            try:
                response = self._client.get(
                    host + _SAFE_ROUTE,
                    params=params,
                    headers={
                        "accept": "application/json",
                        "user-agent":
                            "Brian-Phase113-Grounded-Market-Prefetch/1",
                    },
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = type(exc).__name__
                continue
            if response.status_code in {418, 429}:
                retry_after = response.headers.get("retry-after")
                detail = (
                    f" retry_after={retry_after}"
                    if retry_after
                    else ""
                )
                raise BinanceGroundedMarketPrefetchRateLimitError(
                    f"Binance rate limit HTTP "
                    f"{response.status_code}{detail}"
                )
            if 500 <= response.status_code < 600:
                last_error = f"HTTP_{response.status_code}"
                continue
            if response.status_code < 200 or response.status_code >= 300:
                raise BinanceGroundedMarketPrefetchTransportError(
                    f"Binance public kline HTTP "
                    f"{response.status_code}"
                )
            try:
                return response.json(), host
            except ValueError as exc:
                raise BinanceGroundedMarketPrefetchTransportError(
                    "Binance public kline response is invalid JSON"
                ) from exc
        raise BinanceGroundedMarketPrefetchTransportError(
            "Binance public kline data unavailable: "
            f"{last_error or 'unknown'}"
        )

    def _points(
        self,
        *,
        asset_id: str,
        decision_timestamp: float,
    ) -> tuple[GroundedPricePoint, ...]:
        symbol = _symbol(asset_id)
        payload, host = self._get_klines(
            symbol=symbol,
            decision_timestamp=decision_timestamp,
        )
        if not isinstance(payload, list):
            raise BinanceGroundedMarketPrefetchError(
                "Binance kline payload must be an array"
            )
        decision_ms = decision_timestamp * 1000.0
        points: list[GroundedPricePoint] = []
        seen_close_times: set[int] = set()
        for index, value in enumerate(payload):
            if not isinstance(value, (list, tuple)) or len(value) < 7:
                raise BinanceGroundedMarketPrefetchError(
                    f"kline[{index}] must contain Binance OHLC fields"
                )
            open_time = _float(
                value[0],
                f"kline[{index}].open_time",
            )
            close_time = _float(
                value[6],
                f"kline[{index}].close_time",
            )
            if not open_time.is_integer() or not close_time.is_integer():
                raise BinanceGroundedMarketPrefetchError(
                    "Binance kline timestamps must be integer milliseconds"
                )
            open_ms = int(open_time)
            close_ms = int(close_time)
            if close_ms <= open_ms:
                raise BinanceGroundedMarketPrefetchError(
                    "Binance kline close time must follow open time"
                )
            # API endTime can still return a currently-open candle depending on
            # venue semantics. It never enters the point-in-time history.
            if close_ms > decision_ms + 1e-6:
                continue
            close = _float(
                value[4],
                f"kline[{index}].close",
                positive=True,
            )
            high = _float(
                value[2],
                f"kline[{index}].high",
                positive=True,
            )
            low = _float(
                value[3],
                f"kline[{index}].low",
                positive=True,
            )
            opened = _float(
                value[1],
                f"kline[{index}].open",
                positive=True,
            )
            if high < max(opened, close, low) or low > min(opened, close, high):
                raise BinanceGroundedMarketPrefetchError(
                    "Binance kline OHLC ordering is invalid"
                )
            if close_ms in seen_close_times:
                raise BinanceGroundedMarketPrefetchError(
                    "Binance kline payload contains duplicate close time"
                )
            seen_close_times.add(close_ms)
            points.append(GroundedPricePoint(
                asset_id=asset_id,
                observed_at=close_ms / 1000.0,
                price=close,
                source_id=_source_id(
                    host=host,
                    symbol=symbol,
                    open_time_ms=open_ms,
                    close_time_ms=close_ms,
                    close=close,
                ),
            ))
        points.sort(key=lambda row: row.observed_at)
        required = self.sensor_reader.config.return_observations + 1
        if len(points) < required:
            raise BinanceGroundedMarketPrefetchError(
                f"{asset_id} has insufficient completed Binance klines: "
                f"{len(points)} < {required}"
            )
        return tuple(points)

    def load(
        self,
        *,
        asset_ids: Sequence[str],
        decision_timestamp: float,
    ) -> GroundedMarketPrefetch:
        if self._closed:
            raise BinanceGroundedMarketPrefetchError(
                "Phase113 reader is closed"
            )
        timestamp = float(decision_timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("decision_timestamp must be finite")
        assets = tuple(sorted({str(value).strip() for value in asset_ids}))
        if not assets:
            raise ValueError("asset_ids are required")
        if len(assets) > self.max_assets:
            raise BinanceGroundedMarketPrefetchError(
                f"requested {len(assets)} assets exceeds "
                f"Phase113 max {self.max_assets}"
            )
        for asset in assets:
            _symbol(asset)

        price_points = {
            asset: self._points(
                asset_id=asset,
                decision_timestamp=timestamp,
            )
            for asset in assets
        }
        result = self.sensor_reader.load_with_price_points(
            asset_ids=assets,
            decision_timestamp=timestamp,
            price_points_by_asset=price_points,
        )
        if not isinstance(result, GroundedMarketPrefetch):
            raise BinanceGroundedMarketPrefetchError(
                "Phase110 sensor composition returned invalid type"
            )
        if set(result.asset_inputs) != set(assets):
            raise BinanceGroundedMarketPrefetchError(
                "Phase110 sensor composition changed asset coverage"
            )
        return result

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._owns_client:
                self._client.close()
            if self._owns_sensor_reader:
                self.sensor_reader.close()
        finally:
            self._closed = True

    def __enter__(self) -> "BinanceGroundedMarketPrefetchReader":
        if self._closed:
            raise BinanceGroundedMarketPrefetchError(
                "Phase113 reader is closed"
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
