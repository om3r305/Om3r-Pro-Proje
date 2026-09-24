from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from .global_sensor_mesh import SensorObservation
from .phase54_integrated_shadow_decision import AssetDecisionInput
from .phase91_supabase_rpc_transport import (
    SupabaseRecoveryRpcConfigurationError,
    _is_secure_project_url,
    _read_secret_key_from_env,
    _sanitize_error_payload,
    _validate_server_key,
)
from .phase109_pit_edge_prefetch_builder import PointInTimeReturnSeries
from .portfolio import DEVELOPMENT_CUTOFF

PHASE110_SCHEMA_VERSION = "brian.phase110-supabase-grounded-market-prefetch.v1"

_ALLOWED_TABLES = frozenset({
    "brian_sensor_observations",
    "brian_micro_book_ticks",
    "brian_multiasset_market_marks",
})
_ASSET_ID = re.compile(
    r"^(crypto:[A-Z0-9]{2,20}USDT|fx:[A-Z0-9]{6,12}|"
    r"index:[A-Z0-9]{2,24}|commodity:[A-Z0-9]{2,24}|"
    r"equity:[A-Z0-9.-]{1,16}|etf:[A-Z0-9.-]{1,16})$"
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")

_SOURCE_KIND_BY_FAMILY: Mapping[str, str] = {
    "price_structure": "market_snapshot",
    "orderbook": "orderbook",
    "derivatives": "derivatives",
    "onchain": "onchain",
    "news": "verified_news",
    "social_psychology": "social_psychology",
    "cross_asset": "cross_asset",
    "macro": "macro",
}

_SUPPORTED_HORIZONS = frozenset({
    "MICRO_1_5M",
    "FAST_5_30M",
    "INTRADAY_30M_6H",
    "SWING_6H_7D",
    "MACRO_1D_PLUS",
})


class SupabaseGroundedMarketPrefetchError(RuntimeError):
    pass


class SupabaseGroundedMarketPrefetchResponseError(
    SupabaseGroundedMarketPrefetchError
):
    pass


@dataclass(frozen=True, slots=True)
class GroundedMarketPrefetch:
    decision_timestamp: float
    asset_inputs: Mapping[str, AssetDecisionInput]
    return_series_by_asset: Mapping[str, PointInTimeReturnSeries]
    marks: Mapping[str, float]
    cost_asset_id_by_asset: Mapping[str, str]
    common_return_buckets: tuple[float, ...]
    schema_version: str = PHASE110_SCHEMA_VERSION
    read_only: bool = True
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        assets = set(self.asset_inputs)
        if not assets:
            raise ValueError("Phase110 prefetch requires assets")
        if set(self.return_series_by_asset) != assets:
            raise ValueError("Phase110 return assets must match decision assets")
        if set(self.marks) != assets:
            raise ValueError("Phase110 marks must match decision assets")
        if set(self.cost_asset_id_by_asset) != assets:
            raise ValueError("Phase110 cost ids must match decision assets")
        if not self.common_return_buckets:
            raise ValueError("Phase110 requires aligned return buckets")
        if not self.read_only or not self.shadow_only or self.live_execution:
            raise ValueError("Phase110 must remain read-only shadow-only")


@dataclass(frozen=True, slots=True)
class SupabaseGroundedMarketPrefetchConfig:
    project_url: str
    key_source: str
    timeout_seconds: float = 10.0
    sensor_lookback_seconds: float = 7 * 24 * 60 * 60.0
    price_lookback_seconds: float = 7 * 24 * 60 * 60.0
    mark_max_age_seconds: float = 15 * 60.0
    bucket_seconds: int = 300
    return_observations: int = 60
    max_sensor_rows: int = 4000
    max_price_rows: int = 12000
    schema_version: str = PHASE110_SCHEMA_VERSION
    read_only: bool = True
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not _is_secure_project_url(self.project_url):
            raise ValueError(
                "project_url must use https (http allowed only for localhost)"
            )
        if not self.key_source.strip():
            raise ValueError("key_source is required")
        for label, value in (
            ("timeout_seconds", self.timeout_seconds),
            ("sensor_lookback_seconds", self.sensor_lookback_seconds),
            ("price_lookback_seconds", self.price_lookback_seconds),
            ("mark_max_age_seconds", self.mark_max_age_seconds),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{label} must be positive")
        if self.bucket_seconds < 60 or self.bucket_seconds > 24 * 60 * 60:
            raise ValueError("bucket_seconds must be in [60,86400]")
        if self.return_observations < 30 or self.return_observations > 1000:
            raise ValueError("return_observations must be in [30,1000]")
        if self.max_sensor_rows < 1 or self.max_price_rows < 1:
            raise ValueError("row limits must be positive")
        if not self.read_only or not self.shadow_only or self.live_execution:
            raise ValueError("Phase110 must remain read-only shadow-only")


@dataclass(frozen=True, slots=True)
class _PricePoint:
    asset_id: str
    observed_at: float
    price: float
    source_id: str


@dataclass(frozen=True, slots=True)
class _SensorRow:
    observation_id: str
    eye_id: str
    asset_id: str
    observed_at: float
    direction: int
    strength: float
    confidence: float
    reliability: float
    available: bool
    independent_group: str
    source_ids: tuple[str, ...]
    horizon: str
    reason: str
    sensor_family: str


def _iso_utc(timestamp: float) -> str:
    value = float(timestamp)
    if not math.isfinite(value):
        raise ValueError("timestamp must be finite")
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _timestamp(value: object, label: str) -> float:
    text = str(value or "").strip()
    if not text:
        raise SupabaseGroundedMarketPrefetchResponseError(
            f"{label} is missing"
        )
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SupabaseGroundedMarketPrefetchResponseError(
            f"{label} is not ISO-8601"
        ) from exc
    if parsed.tzinfo is None:
        raise SupabaseGroundedMarketPrefetchResponseError(
            f"{label} must include timezone"
        )
    result = parsed.timestamp()
    if not math.isfinite(result):
        raise SupabaseGroundedMarketPrefetchResponseError(
            f"{label} is not finite"
        )
    return result


def _number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SupabaseGroundedMarketPrefetchResponseError(
            f"{label} must be numeric"
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SupabaseGroundedMarketPrefetchResponseError(
            f"{label} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise SupabaseGroundedMarketPrefetchResponseError(
            f"{label} must be finite"
        )
    return result


def _probability(value: object, label: str) -> float:
    result = _number(value, label)
    if not 0 <= result <= 1:
        raise SupabaseGroundedMarketPrefetchResponseError(
            f"{label} must be in [0,1]"
        )
    return result


def _identifier(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(text):
        raise SupabaseGroundedMarketPrefetchResponseError(
            f"{label} has invalid identifier characters"
        )
    return text


def _asset_id(value: object) -> str:
    text = str(value or "").strip()
    if not _ASSET_ID.fullmatch(text):
        raise ValueError(f"unsupported Phase110 asset id: {text!r}")
    return text


def _ema(values: Sequence[float], span: int) -> tuple[float, ...]:
    if not values:
        return ()
    if span <= 1:
        return tuple(float(value) for value in values)
    alpha = 2.0 / (span + 1.0)
    rows = [float(values[0])]
    for value in values[1:]:
        rows.append(float(value) * alpha + rows[-1] * (1.0 - alpha))
    return tuple(rows)


def _expert_snapshot(
    sensors: Sequence[_SensorRow],
    prices: Sequence[float],
) -> dict[str, float]:
    """Build only features directly measured or deterministically price-derived.

    The EMA recurrence matches the repository's existing candles.py helper.
    Mean-reversion z-score mirrors the Phase38 price-series approach. No RSI,
    volume, divergence, support/resistance, or multi-timeframe state is invented
    when the persisted source does not contain it.
    """
    snapshot: dict[str, float] = {}
    structure = [
        row
        for row in sensors
        if row.independent_group == "price_structure" and row.available
    ]
    if structure:
        newest_at = max(row.observed_at for row in structure)
        newest = [
            row for row in structure
            if abs(row.observed_at - newest_at) <= 1e-9
        ]
        directions = {row.direction for row in newest}
        if len(directions) == 1:
            snapshot["structure_state"] = float(next(iter(directions)))

    if len(prices) >= 2:
        last_return = math.log(prices[-1] / prices[-2])
        snapshot["return_1"] = last_return
    if len(prices) >= 3:
        previous_return = math.log(prices[-2] / prices[-3])
        snapshot["acceleration"] = snapshot["return_1"] - previous_return

    if len(prices) >= 10:
        ema = _ema(prices, 10)
        denominator = max(abs(prices[-2]), 1e-12)
        snapshot["ema_slope"] = (ema[-1] - ema[-2]) / denominator

    if len(prices) >= 12:
        window = tuple(float(value) for value in prices[-12:])
        mean = sum(window) / len(window)
        variance = sum((value - mean) ** 2 for value in window) / len(window)
        std = math.sqrt(max(0.0, variance))
        if std > 1e-12:
            snapshot["zscore"] = (window[-1] - mean) / std

    return snapshot


class SupabaseGroundedMarketPrefetchReader:
    """GET-only Phase110 adapter for current grounded observations and PIT prices.

    Crypto prices follow the existing public marks path through
    brian_micro_book_ticks. Non-crypto prices use brian_multiasset_market_marks.
    Return histories are aligned on a common closed time-bucket intersection,
    then converted to log returns. If enough common buckets do not exist, the
    reader fails closed rather than padding, forward-filling or fabricating zero
    returns.
    """

    def __init__(
        self,
        *,
        config: SupabaseGroundedMarketPrefetchConfig,
        api_key: str,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise SupabaseRecoveryRpcConfigurationError(
                "Supabase API key is required"
            )
        _validate_server_key(api_key, config.key_source)
        self.config = config
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=config.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> "SupabaseGroundedMarketPrefetchReader":
        source = os.environ if env is None else env
        project_url = source.get("SUPABASE_URL", "").strip().rstrip("/")
        if not project_url:
            raise SupabaseRecoveryRpcConfigurationError(
                "SUPABASE_URL is required"
            )
        key, key_source = _read_secret_key_from_env(source)
        _validate_server_key(key, key_source)

        def _float(name: str, default: float) -> float:
            raw = source.get(name)
            if raw is None or not str(raw).strip():
                return default
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise SupabaseRecoveryRpcConfigurationError(
                    f"{name} must be numeric"
                ) from exc
            if not math.isfinite(value) or value <= 0:
                raise SupabaseRecoveryRpcConfigurationError(
                    f"{name} must be positive"
                )
            return value

        def _int(name: str, default: int) -> int:
            raw = source.get(name)
            if raw is None or not str(raw).strip():
                return default
            try:
                return int(raw)
            except (TypeError, ValueError) as exc:
                raise SupabaseRecoveryRpcConfigurationError(
                    f"{name} must be integer"
                ) from exc

        return cls(
            config=SupabaseGroundedMarketPrefetchConfig(
                project_url=project_url,
                key_source=key_source,
                timeout_seconds=_float(
                    "BRIAN_PREFETCH_READER_TIMEOUT_SECONDS",
                    10.0,
                ),
                sensor_lookback_seconds=_float(
                    "BRIAN_PREFETCH_SENSOR_LOOKBACK_SECONDS",
                    7 * 24 * 60 * 60.0,
                ),
                price_lookback_seconds=_float(
                    "BRIAN_PREFETCH_PRICE_LOOKBACK_SECONDS",
                    7 * 24 * 60 * 60.0,
                ),
                mark_max_age_seconds=_float(
                    "BRIAN_PREFETCH_MARK_MAX_AGE_SECONDS",
                    15 * 60.0,
                ),
                bucket_seconds=_int(
                    "BRIAN_PREFETCH_BUCKET_SECONDS",
                    300,
                ),
                return_observations=_int(
                    "BRIAN_PREFETCH_RETURN_OBSERVATIONS",
                    60,
                ),
            ),
            api_key=key,
            client=client,
        )

    def _get(
        self,
        table: str,
        *,
        params: Mapping[str, str],
    ) -> list[Mapping[str, Any]]:
        if table not in _ALLOWED_TABLES:
            raise SupabaseGroundedMarketPrefetchError(
                f"table is not allowed by Phase110: {table}"
            )
        response: httpx.Response
        try:
            response = self._client.get(
                f"{self.config.project_url}/rest/v1/{table}",
                headers={
                    "apikey": self._api_key,
                    "accept": "application/json",
                    "user-agent": "brian-phase110-grounded-prefetch/1",
                },
                params=dict(params),
            )
        except httpx.TimeoutException as exc:
            raise SupabaseGroundedMarketPrefetchError(
                f"Supabase read timeout for {table}"
            ) from exc
        except httpx.TransportError as exc:
            raise SupabaseGroundedMarketPrefetchError(
                f"Supabase read transport failure for {table}"
            ) from exc
        if response.status_code < 200 or response.status_code >= 300:
            detail = _sanitize_error_payload(response)
            raise SupabaseGroundedMarketPrefetchResponseError(
                f"Supabase read {table} returned HTTP "
                f"{response.status_code}: {detail}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise SupabaseGroundedMarketPrefetchResponseError(
                f"Supabase read {table} returned invalid JSON"
            ) from exc
        if not isinstance(payload, list):
            raise SupabaseGroundedMarketPrefetchResponseError(
                f"Supabase read {table} must return a JSON array"
            )
        rows: list[Mapping[str, Any]] = []
        for value in payload:
            if not isinstance(value, Mapping):
                raise SupabaseGroundedMarketPrefetchResponseError(
                    f"Supabase read {table} returned non-object row"
                )
            rows.append(value)
        return rows

    def _sensor_rows(
        self,
        assets: Sequence[str],
        *,
        decision_timestamp: float,
    ) -> dict[str, tuple[_SensorRow, ...]]:
        decision_iso = _iso_utc(decision_timestamp)
        lower_iso = _iso_utc(
            decision_timestamp - self.config.sensor_lookback_seconds
        )
        rows = self._get(
            "brian_sensor_observations",
            params={
                "select": (
                    "observation_id,eye_id,asset_id,sensor_family,horizon,"
                    "independent_group,observed_at,direction,strength,"
                    "confidence,reliability,available,source_ids,reason,"
                    "evidence_class,shadow_only,live_execution"
                ),
                "asset_id": "in.(" + ",".join(assets) + ")",
                "observed_at": f"lte.{decision_iso}",
                "and": f"(observed_at.gte.{lower_iso})",
                "order": "observed_at.desc,observation_id.asc",
                "limit": str(self.config.max_sensor_rows),
            },
        )
        latest_by_eye: dict[tuple[str, str], _SensorRow] = {}
        for row in rows:
            asset = _asset_id(row.get("asset_id"))
            if asset not in assets:
                raise SupabaseGroundedMarketPrefetchResponseError(
                    f"sensor response contains unrequested asset {asset}"
                )
            observed_at = _timestamp(row.get("observed_at"), "observed_at")
            if observed_at > decision_timestamp:
                raise SupabaseGroundedMarketPrefetchResponseError(
                    "sensor response contains post-decision observation"
                )
            if observed_at < DEVELOPMENT_CUTOFF:
                raise SupabaseGroundedMarketPrefetchResponseError(
                    "sensor response reused pre-cutoff development evidence"
                )
            if row.get("shadow_only") is not True:
                raise SupabaseGroundedMarketPrefetchResponseError(
                    "sensor row is not shadow_only"
                )
            if row.get("live_execution") is not False:
                raise SupabaseGroundedMarketPrefetchResponseError(
                    "sensor row crossed live boundary"
                )
            if (
                str(row.get("evidence_class") or "")
                != "PROSPECTIVE_DEVELOPMENT_SHADOW"
            ):
                raise SupabaseGroundedMarketPrefetchResponseError(
                    "sensor row has wrong evidence class"
                )
            horizon = str(row.get("horizon") or "")
            if horizon not in _SUPPORTED_HORIZONS:
                raise SupabaseGroundedMarketPrefetchResponseError(
                    f"unsupported sensor horizon: {horizon}"
                )
            direction = int(_number(row.get("direction"), "direction"))
            if direction not in (-1, 0, 1):
                raise SupabaseGroundedMarketPrefetchResponseError(
                    "sensor direction must be -1,0,1"
                )
            available = row.get("available")
            if not isinstance(available, bool):
                raise SupabaseGroundedMarketPrefetchResponseError(
                    "sensor available must be boolean"
                )
            source_value = row.get("source_ids")
            if not isinstance(source_value, list):
                raise SupabaseGroundedMarketPrefetchResponseError(
                    "sensor source_ids must be an array"
                )
            observation_id = _identifier(
                row.get("observation_id"),
                "observation_id",
            )
            source_ids = tuple(sorted({
                observation_id,
                *(
                    _identifier(value, "source_id")
                    for value in source_value
                ),
            }))
            sensor = _SensorRow(
                observation_id=observation_id,
                eye_id=_identifier(row.get("eye_id"), "eye_id"),
                asset_id=asset,
                observed_at=observed_at,
                direction=direction,
                strength=_probability(row.get("strength"), "strength"),
                confidence=_probability(
                    row.get("confidence"),
                    "confidence",
                ),
                reliability=_probability(
                    row.get("reliability"),
                    "reliability",
                ),
                available=available,
                independent_group=_identifier(
                    row.get("independent_group"),
                    "independent_group",
                ),
                source_ids=source_ids,
                horizon=horizon,
                reason=str(row.get("reason") or "").strip(),
                sensor_family=_identifier(
                    row.get("sensor_family"),
                    "sensor_family",
                ),
            )
            if not sensor.reason:
                raise SupabaseGroundedMarketPrefetchResponseError(
                    "sensor reason is required"
                )
            if sensor.available and not sensor.source_ids:
                raise SupabaseGroundedMarketPrefetchResponseError(
                    "available sensor row has no provenance"
                )
            key = (asset, sensor.eye_id)
            prior = latest_by_eye.get(key)
            if prior is None:
                latest_by_eye[key] = sensor
            elif abs(prior.observed_at - sensor.observed_at) <= 1e-9:
                if prior != sensor:
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        f"ambiguous latest sensor state for eye {sensor.eye_id}"
                    )
            # rows are ordered newest first; older rows for the same eye are
            # intentionally ignored so stale direction cannot outvote current.
        result: dict[str, list[_SensorRow]] = {
            asset: [] for asset in assets
        }
        for (asset, _), row in latest_by_eye.items():
            result[asset].append(row)
        for asset in assets:
            if not result[asset]:
                raise SupabaseGroundedMarketPrefetchError(
                    f"no prospective sensor observations for {asset}"
                )
            result[asset].sort(
                key=lambda row: (
                    row.sensor_family,
                    row.independent_group,
                    row.eye_id,
                )
            )
        return {
            asset: tuple(rows)
            for asset, rows in result.items()
        }

    def _price_rows(
        self,
        assets: Sequence[str],
        *,
        decision_timestamp: float,
    ) -> dict[str, tuple[_PricePoint, ...]]:
        decision_iso = _iso_utc(decision_timestamp)
        lower_iso = _iso_utc(
            decision_timestamp - self.config.price_lookback_seconds
        )
        crypto = tuple(asset for asset in assets if asset.startswith("crypto:"))
        other = tuple(asset for asset in assets if not asset.startswith("crypto:"))
        points: dict[str, list[_PricePoint]] = {
            asset: [] for asset in assets
        }

        if crypto:
            rows = self._get(
                "brian_micro_book_ticks",
                params={
                    "select": (
                        "tick_id,asset_id,observed_at,observed_mid_price,"
                        "evidence_class,shadow_only,live_execution"
                    ),
                    "asset_id": "in.(" + ",".join(crypto) + ")",
                    "observed_at": f"lte.{decision_iso}",
                    "and": f"(observed_at.gte.{lower_iso})",
                    "order": "observed_at.desc,tick_id.asc",
                    "limit": str(self.config.max_price_rows),
                },
            )
            for row in rows:
                asset = _asset_id(row.get("asset_id"))
                if asset not in crypto:
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        f"crypto price response contains unrequested asset {asset}"
                    )
                if row.get("shadow_only") is not True or row.get(
                    "live_execution"
                ) is not False:
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        "crypto price row crossed shadow-only boundary"
                    )
                if (
                    str(row.get("evidence_class") or "")
                    != "PROSPECTIVE_DEVELOPMENT_SHADOW"
                ):
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        "crypto price row has wrong evidence class"
                    )
                observed = _timestamp(
                    row.get("observed_at"),
                    "crypto observed_at",
                )
                if observed > decision_timestamp:
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        "crypto price response contains future row"
                    )
                price = _number(
                    row.get("observed_mid_price"),
                    "observed_mid_price",
                )
                if price <= 0:
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        "observed_mid_price must be positive"
                    )
                points[asset].append(_PricePoint(
                    asset_id=asset,
                    observed_at=observed,
                    price=price,
                    source_id=_identifier(
                        row.get("tick_id"),
                        "tick_id",
                    ),
                ))

        if other:
            rows = self._get(
                "brian_multiasset_market_marks",
                params={
                    "select": (
                        "mark_id,asset_id,provider_time,price,"
                        "provider_quality,evidence_class,"
                        "shadow_only,live_execution"
                    ),
                    "asset_id": "in.(" + ",".join(other) + ")",
                    "provider_time": f"lte.{decision_iso}",
                    "and": f"(provider_time.gte.{lower_iso})",
                    "order": "provider_time.desc,mark_id.asc",
                    "limit": str(self.config.max_price_rows),
                },
            )
            for row in rows:
                asset = _asset_id(row.get("asset_id"))
                if asset not in other:
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        f"market mark response contains unrequested asset {asset}"
                    )
                if row.get("shadow_only") is not True or row.get(
                    "live_execution"
                ) is not False:
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        "market mark row crossed shadow-only boundary"
                    )
                if (
                    str(row.get("evidence_class") or "")
                    != "PROSPECTIVE_DEVELOPMENT_SHADOW"
                ):
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        "market mark row has wrong evidence class"
                    )
                quality = str(row.get("provider_quality") or "").strip()
                if not quality or quality.upper() == "UNAVAILABLE":
                    continue
                observed = _timestamp(
                    row.get("provider_time"),
                    "provider_time",
                )
                if observed > decision_timestamp:
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        "market mark response contains future row"
                    )
                price = _number(row.get("price"), "price")
                if price <= 0:
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        "market mark price must be positive"
                    )
                points[asset].append(_PricePoint(
                    asset_id=asset,
                    observed_at=observed,
                    price=price,
                    source_id=_identifier(
                        row.get("mark_id"),
                        "mark_id",
                    ),
                ))

        result: dict[str, tuple[_PricePoint, ...]] = {}
        for asset, rows in points.items():
            if not rows:
                raise SupabaseGroundedMarketPrefetchError(
                    f"no PIT market prices for {asset}"
                )
            rows.sort(key=lambda row: (row.observed_at, row.source_id))
            result[asset] = tuple(rows)
        return result

    def _bucket_prices(
        self,
        rows: Sequence[_PricePoint],
    ) -> dict[int, _PricePoint]:
        buckets: dict[int, _PricePoint] = {}
        for row in rows:
            bucket = int(row.observed_at // self.config.bucket_seconds)
            prior = buckets.get(bucket)
            if prior is None or row.observed_at > prior.observed_at:
                buckets[bucket] = row
                continue
            if abs(row.observed_at - prior.observed_at) <= 1e-9:
                relative = abs(row.price - prior.price) / max(
                    abs(row.price),
                    abs(prior.price),
                    1e-12,
                )
                if relative > 1e-9:
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        f"conflicting prices in bucket {bucket}"
                    )
                if row.source_id < prior.source_id:
                    buckets[bucket] = row
        return buckets

    def load(
        self,
        *,
        asset_ids: Sequence[str],
        decision_timestamp: float,
    ) -> GroundedMarketPrefetch:
        timestamp = float(decision_timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("decision_timestamp must be finite")
        if timestamp < DEVELOPMENT_CUTOFF:
            raise SupabaseGroundedMarketPrefetchError(
                "Phase110 requires post-cutoff prospective decision time"
            )
        assets = tuple(sorted({_asset_id(value) for value in asset_ids}))
        if not assets:
            raise ValueError("asset_ids are required")

        sensor_rows = self._sensor_rows(
            assets,
            decision_timestamp=timestamp,
        )
        price_rows = self._price_rows(
            assets,
            decision_timestamp=timestamp,
        )

        latest_marks: dict[str, float] = {}
        for asset, rows in price_rows.items():
            latest = max(rows, key=lambda row: row.observed_at)
            age = timestamp - latest.observed_at
            if age < -1e-9:
                raise SupabaseGroundedMarketPrefetchResponseError(
                    f"{asset} latest mark is from the future"
                )
            if age > self.config.mark_max_age_seconds:
                raise SupabaseGroundedMarketPrefetchError(
                    f"{asset} latest mark is stale by {age:.3f}s"
                )
            latest_marks[asset] = latest.price

        buckets_by_asset = {
            asset: self._bucket_prices(rows)
            for asset, rows in price_rows.items()
        }
        common = set.intersection(
            *(set(rows) for rows in buckets_by_asset.values())
        )
        # Covariance uses completed buckets only. A point captured inside the
        # still-open decision-time bucket may serve as the current mark, but it
        # cannot enter the historical return matrix.
        common = {
            bucket
            for bucket in common
            if (bucket + 1) * self.config.bucket_seconds
            <= timestamp + 1e-9
        }
        required_buckets = self.config.return_observations + 1
        if len(common) < required_buckets:
            raise SupabaseGroundedMarketPrefetchError(
                "insufficient aligned PIT price buckets: "
                f"{len(common)} < {required_buckets}"
            )
        chosen = tuple(sorted(common)[-required_buckets:])
        bucket_times = tuple(
            float(bucket * self.config.bucket_seconds)
            for bucket in chosen
        )

        return_series: dict[str, PointInTimeReturnSeries] = {}
        asset_inputs: dict[str, AssetDecisionInput] = {}
        for asset in assets:
            selected = tuple(
                buckets_by_asset[asset][bucket]
                for bucket in chosen
            )
            prices = tuple(row.price for row in selected)
            returns = tuple(
                math.log(prices[index] / prices[index - 1])
                for index in range(1, len(prices))
            )
            return_series[asset] = PointInTimeReturnSeries(
                asset_id=asset,
                values=returns,
                observed_from=selected[0].observed_at,
                observed_until=selected[-1].observed_at,
                source_ids=tuple(row.source_id for row in selected),
            )

            rows = sensor_rows[asset]
            observations: list[SensorObservation] = []
            source_kind_by_eye: dict[str, str] = {}
            for row in rows:
                source_kind = _SOURCE_KIND_BY_FAMILY.get(row.sensor_family)
                if source_kind is None:
                    raise SupabaseGroundedMarketPrefetchResponseError(
                        f"unsupported sensor family: {row.sensor_family}"
                    )
                observations.append(SensorObservation(
                    eye_id=row.eye_id,
                    asset_id=row.asset_id,
                    observed_at=row.observed_at,
                    direction=row.direction,
                    strength=row.strength,
                    confidence=row.confidence,
                    reliability=row.reliability,
                    available=row.available,
                    independent_group=row.independent_group,
                    source_ids=row.source_ids,
                    horizon=row.horizon,  # validated above
                    reason=row.reason,
                ))
                source_kind_by_eye[row.eye_id] = source_kind
            asset_inputs[asset] = AssetDecisionInput(
                snapshot=_expert_snapshot(rows, prices),
                observations=tuple(observations),
                source_kind_by_eye=source_kind_by_eye,
            )

        return GroundedMarketPrefetch(
            decision_timestamp=timestamp,
            asset_inputs=asset_inputs,
            return_series_by_asset=return_series,
            marks=latest_marks,
            cost_asset_id_by_asset={
                asset: asset for asset in assets
            },
            common_return_buckets=bucket_times,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SupabaseGroundedMarketPrefetchReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
