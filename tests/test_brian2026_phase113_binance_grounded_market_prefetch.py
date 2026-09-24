from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from brian2026.global_sensor_mesh import SensorObservation
from brian2026.phase54_integrated_shadow_decision import AssetDecisionInput
from brian2026.phase109_pit_edge_prefetch_builder import PointInTimeReturnSeries
from brian2026.phase110_supabase_grounded_market_prefetch import (
    GroundedMarketPrefetch,
)
from brian2026.phase113_binance_grounded_market_prefetch import (
    BinanceGroundedMarketPrefetchError,
    BinanceGroundedMarketPrefetchRateLimitError,
    BinanceGroundedMarketPrefetchReader,
)


TS = 1_789_999_950.0
BUCKET = 300
CURRENT_BUCKET = int(TS // BUCKET)
ASSET = "crypto:BTCUSDT"
SYMBOL = "BTCUSDT"


def _kline_rows(*, count=31, include_open=True):
    rows = []
    first_bucket = CURRENT_BUCKET - count
    for index in range(count):
        bucket = first_bucket + index
        open_ms = bucket * BUCKET * 1000
        close_ms = (bucket + 1) * BUCKET * 1000 - 1
        opened = 100.0 + index
        close = opened + 0.5
        rows.append([
            open_ms,
            f"{opened:.8f}",
            f"{close + 0.2:.8f}",
            f"{opened - 0.2:.8f}",
            f"{close:.8f}",
            "1000.0",
            close_ms,
            "0",
            100,
            "0",
            "0",
            "0",
        ])
    if include_open:
        bucket = CURRENT_BUCKET
        open_ms = bucket * BUCKET * 1000
        close_ms = (bucket + 1) * BUCKET * 1000 - 1
        rows.append([
            open_ms,
            "131.0",
            "132.0",
            "130.0",
            "131.5",
            "1000.0",
            close_ms,
            "0",
            100,
            "0",
            "0",
            "0",
        ])
    return rows


def _result(asset=ASSET):
    observation = SensorObservation(
        eye_id="eye-113",
        asset_id=asset,
        observed_at=TS - 10,
        direction=1,
        strength=0.8,
        confidence=0.8,
        reliability=0.7,
        available=True,
        independent_group="price_structure",
        source_ids=("obs-113",),
        horizon="FAST_5_30M",
        reason="phase113 fixture",
    )
    item = AssetDecisionInput(
        snapshot={"structure_state": 1.0},
        observations=(observation,),
        source_kind_by_eye={"eye-113": "market_snapshot"},
    )
    returns = PointInTimeReturnSeries(
        asset_id=asset,
        values=tuple(0.001 for _ in range(30)),
        observed_from=TS - 31 * BUCKET,
        observed_until=TS - BUCKET,
        source_ids=tuple(f"k-{i}" for i in range(31)),
    )
    return GroundedMarketPrefetch(
        decision_timestamp=TS,
        asset_inputs={asset: item},
        return_series_by_asset={asset: returns},
        marks={asset: 130.5},
        cost_asset_id_by_asset={asset: asset},
        common_return_buckets=tuple(
            TS - (31 - index) * BUCKET
            for index in range(31)
        ),
    )


class _SensorReader:
    def __init__(self, *, bucket_seconds=300, return_observations=30):
        self.config = SimpleNamespace(
            bucket_seconds=bucket_seconds,
            return_observations=return_observations,
        )
        self.calls = []
        self.close_calls = 0

    def load_with_price_points(self, **kwargs):
        self.calls.append(kwargs)
        return _result(next(iter(kwargs["asset_ids"])))

    def close(self):
        self.close_calls += 1


def _reader(handler, *, sensor=None):
    sensor_reader = sensor or _SensorReader()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    reader = BinanceGroundedMarketPrefetchReader(
        sensor_reader=sensor_reader,
        client=client,
        clock=lambda: TS,
    )
    return reader, sensor_reader, client


def test_completed_5m_klines_feed_phase110_and_open_candle_is_excluded() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        assert request.url.path == "/api/v3/klines"
        assert request.url.params["symbol"] == SYMBOL
        assert request.url.params["interval"] == "5m"
        assert request.url.params["endTime"] == str(int(TS * 1000) - 1)
        assert "authorization" not in request.headers
        assert "apikey" not in request.headers
        return httpx.Response(200, json=_kline_rows())

    reader, sensor, client = _reader(handler)
    try:
        result = reader.load(
            asset_ids=(ASSET,),
            decision_timestamp=TS,
        )
    finally:
        client.close()

    assert result.asset_inputs.keys() == {ASSET}
    assert len(sensor.calls) == 1
    call = sensor.calls[0]
    points = call["price_points_by_asset"][ASSET]
    assert len(points) == 31
    assert all(point.observed_at <= TS for point in points)
    assert points[-1].observed_at < CURRENT_BUCKET * BUCKET
    assert all(
        point.source_id.startswith("binance-kline:")
        for point in points
    )
    assert call["decision_timestamp"] == TS
    assert call["asset_ids"] == (ASSET,)
    assert len(requests) == 1


def test_server_current_candle_never_enters_pit_history() -> None:
    rows = _kline_rows(count=31, include_open=True)
    # Make the current/open candle absurd so accidental inclusion is obvious.
    rows[-1][4] = "999999.0"

    reader, sensor, client = _reader(
        lambda request: httpx.Response(200, json=rows)
    )
    try:
        reader.load(asset_ids=(ASSET,), decision_timestamp=TS)
    finally:
        client.close()

    points = sensor.calls[0]["price_points_by_asset"][ASSET]
    assert all(point.price < 1000 for point in points)


def test_noncrypto_asset_is_rejected_before_network_or_sensor_read() -> None:
    network_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network must not be reached")

    reader, sensor, client = _reader(handler)
    try:
        with pytest.raises(
            BinanceGroundedMarketPrefetchError,
            match="crypto:\\*USDT",
        ):
            reader.load(
                asset_ids=("fx:EURUSD",),
                decision_timestamp=TS,
            )
    finally:
        client.close()

    assert network_calls == 0
    assert sensor.calls == []


def test_insufficient_completed_klines_fail_before_sensor_composition() -> None:
    reader, sensor, client = _reader(
        lambda request: httpx.Response(
            200,
            json=_kline_rows(count=20, include_open=False),
        )
    )
    try:
        with pytest.raises(
            BinanceGroundedMarketPrefetchError,
            match="insufficient completed Binance klines",
        ):
            reader.load(asset_ids=(ASSET,), decision_timestamp=TS)
    finally:
        client.close()

    assert sensor.calls == []


def test_duplicate_kline_close_time_fails_closed() -> None:
    rows = _kline_rows(include_open=False)
    rows.append(list(rows[-1]))
    reader, sensor, client = _reader(
        lambda request: httpx.Response(200, json=rows)
    )
    try:
        with pytest.raises(
            BinanceGroundedMarketPrefetchError,
            match="duplicate close time",
        ):
            reader.load(asset_ids=(ASSET,), decision_timestamp=TS)
    finally:
        client.close()

    assert sensor.calls == []


def test_invalid_ohlc_ordering_fails_closed() -> None:
    rows = _kline_rows(include_open=False)
    rows[0][2] = "1.0"
    reader, sensor, client = _reader(
        lambda request: httpx.Response(200, json=rows)
    )
    try:
        with pytest.raises(
            BinanceGroundedMarketPrefetchError,
            match="OHLC ordering",
        ):
            reader.load(asset_ids=(ASSET,), decision_timestamp=TS)
    finally:
        client.close()

    assert sensor.calls == []


def test_rate_limit_is_machine_visible_and_does_not_fail_over() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            headers={"retry-after": "2"},
            json={"code": -1003},
        )

    reader, sensor, client = _reader(handler)
    try:
        with pytest.raises(
            BinanceGroundedMarketPrefetchRateLimitError,
            match="retry_after=2",
        ):
            reader.load(asset_ids=(ASSET,), decision_timestamp=TS)
    finally:
        client.close()

    assert calls == 1
    assert sensor.calls == []


def test_5xx_fails_over_to_next_public_host() -> None:
    hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if len(hosts) == 1:
            return httpx.Response(503, json={"error": "temporary"})
        return httpx.Response(
            200,
            json=_kline_rows(include_open=False),
        )

    reader, sensor, client = _reader(handler)
    try:
        reader.load(asset_ids=(ASSET,), decision_timestamp=TS)
    finally:
        client.close()

    assert len(hosts) == 2
    assert hosts[0] != hosts[1]
    assert len(sensor.calls) == 1


def test_phase113_requires_5m_phase110_bucket_contract() -> None:
    sensor = _SensorReader(bucket_seconds=60)
    with pytest.raises(ValueError, match="bucket_seconds=300"):
        BinanceGroundedMarketPrefetchReader(
            sensor_reader=sensor,
            client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=[])
                )
            ),
        )


def test_phase113_refuses_return_window_beyond_single_bounded_kline_read() -> None:
    sensor = _SensorReader(return_observations=1000)
    with pytest.raises(ValueError, match="<= 999"):
        BinanceGroundedMarketPrefetchReader(
            sensor_reader=sensor,
            client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=[])
                )
            ),
        )


def test_owned_sensor_reader_is_closed_but_external_reader_is_not() -> None:
    handler = lambda request: httpx.Response(
        200,
        json=_kline_rows(include_open=False),
    )
    external = _SensorReader()
    reader, _, client = _reader(handler, sensor=external)
    reader.close()
    client.close()
    assert external.close_calls == 0

    owned = _SensorReader()
    client2 = httpx.Client(transport=httpx.MockTransport(handler))
    reader2 = BinanceGroundedMarketPrefetchReader(
        sensor_reader=owned,
        client=client2,
        owns_sensor_reader=True,
    )
    reader2.close()
    client2.close()
    assert owned.close_calls == 1


def test_from_env_closes_sensor_reader_if_constructor_fails() -> None:
    sensor = _SensorReader(bucket_seconds=60)

    with pytest.raises(ValueError, match="bucket_seconds=300"):
        BinanceGroundedMarketPrefetchReader.from_env(
            env={"SUPABASE_URL": "https://example.supabase.co"},
            sensor_reader_factory=lambda **kwargs: sensor,
        )

    assert sensor.close_calls == 1
