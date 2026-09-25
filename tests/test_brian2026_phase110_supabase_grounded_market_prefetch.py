from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from brian2026.phase91_supabase_rpc_transport import (
    SupabaseRecoveryRpcConfigurationError,
)
from brian2026.phase110_supabase_grounded_market_prefetch import (
    GroundedPricePoint,
    SupabaseGroundedMarketPrefetchConfig,
    SupabaseGroundedMarketPrefetchError,
    SupabaseGroundedMarketPrefetchReader,
    SupabaseGroundedMarketPrefetchResponseError,
)


TS = 1_789_999_950.0
BUCKET = 300
CURRENT_BUCKET = int(TS // BUCKET)


def _iso(value: float) -> str:
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _sensor(
    asset: str,
    eye: str,
    family: str,
    group: str,
    *,
    direction: int = 1,
    observed_at: float = TS - 30,
    observation_id: str | None = None,
):
    return {
        "observation_id": observation_id or f"obs-{eye}",
        "eye_id": eye,
        "asset_id": asset,
        "sensor_family": family,
        "horizon": "FAST_5_30M",
        "independent_group": group,
        "observed_at": _iso(observed_at),
        "direction": direction,
        "strength": 0.8,
        "confidence": 0.75,
        "reliability": 0.6,
        "available": True,
        "source_ids": [f"raw-{eye}"],
        "reason": "phase110 fixture",
        "evidence_class": "PROSPECTIVE_DEVELOPMENT_SHADOW",
        "shadow_only": True,
        "live_execution": False,
    }


def _closed_price_points(
    asset: str,
    *,
    base: float,
    count: int = 31,
    crypto: bool,
):
    rows = []
    first_bucket = CURRENT_BUCKET - count
    for index in range(count):
        bucket = first_bucket + index
        observed = bucket * BUCKET + (BUCKET - 1)
        price = base * (1.0 + 0.001 * index)
        if crypto:
            rows.append({
                "tick_id": f"tick-{asset.replace(':', '-')}-{index:03d}",
                "asset_id": asset,
                "observed_at": _iso(observed),
                "observed_mid_price": price,
                "evidence_class": "PROSPECTIVE_DEVELOPMENT_SHADOW",
                "shadow_only": True,
                "live_execution": False,
            })
        else:
            rows.append({
                "mark_id": f"mark-{asset.replace(':', '-')}-{index:03d}",
                "asset_id": asset,
                "provider_time": _iso(observed),
                "price": price,
                "provider_quality": "REALTIME",
                "evidence_class": "PROSPECTIVE_DEVELOPMENT_SHADOW",
                "shadow_only": True,
                "live_execution": False,
            })
    return rows


def _current_mark(asset: str, *, price: float, crypto: bool):
    if crypto:
        return {
            "tick_id": f"tick-{asset.replace(':', '-')}-current",
            "asset_id": asset,
            "observed_at": _iso(TS - 10),
            "observed_mid_price": price,
            "evidence_class": "PROSPECTIVE_DEVELOPMENT_SHADOW",
            "shadow_only": True,
            "live_execution": False,
        }
    return {
        "mark_id": f"mark-{asset.replace(':', '-')}-current",
        "asset_id": asset,
        "provider_time": _iso(TS - 10),
        "price": price,
        "provider_quality": "REALTIME",
        "evidence_class": "PROSPECTIVE_DEVELOPMENT_SHADOW",
        "shadow_only": True,
        "live_execution": False,
    }


def _reader(handler, **overrides):
    values = dict(
        project_url="https://example.supabase.co",
        key_source="SUPABASE_SECRET_KEY",
        mark_max_age_seconds=600.0,
        bucket_seconds=BUCKET,
        return_observations=30,
        max_sensor_rows=500,
        max_price_rows=5000,
    )
    values.update(overrides)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    reader = SupabaseGroundedMarketPrefetchReader(
        config=SupabaseGroundedMarketPrefetchConfig(**values),
        api_key="sb_secret_phase110",
        client=client,
    )
    return reader, client


def _happy_handler(requests):
    crypto = "crypto:BTCUSDT"
    fx = "fx:EURUSD"
    sensors = [
        _sensor(
            crypto,
            "eye-btc-structure",
            "price_structure",
            "price_structure",
            direction=1,
        ),
        _sensor(
            crypto,
            "eye-btc-deriv",
            "derivatives",
            "derivatives",
            direction=1,
        ),
        _sensor(
            fx,
            "eye-fx-structure",
            "price_structure",
            "price_structure",
            direction=-1,
        ),
        _sensor(
            fx,
            "eye-fx-macro",
            "macro",
            "macro",
            direction=-1,
        ),
    ]
    crypto_prices = _closed_price_points(
        crypto,
        base=100.0,
        crypto=True,
    ) + [_current_mark(crypto, price=104.0, crypto=True)]
    fx_prices = _closed_price_points(
        fx,
        base=1.05,
        crypto=False,
    ) + [_current_mark(fx, price=1.08, crypto=False)]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        assert request.headers["apikey"] == "sb_secret_phase110"
        assert "authorization" not in request.headers
        assert "sb_secret_phase110" not in str(request.url)
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_observations"
        ):
            return httpx.Response(200, json=sensors)
        if request.url.path.endswith(
            "/rest/v1/brian_micro_book_ticks"
        ):
            return httpx.Response(200, json=list(reversed(crypto_prices)))
        if request.url.path.endswith(
            "/rest/v1/brian_multiasset_market_marks"
        ):
            return httpx.Response(200, json=list(reversed(fx_prices)))
        raise AssertionError(str(request.url))

    return handler


def test_loads_mixed_assets_with_aligned_closed_returns_and_current_marks() -> None:
    requests = []
    reader, client = _reader(_happy_handler(requests))
    try:
        result = reader.load(
            asset_ids=("crypto:BTCUSDT", "fx:EURUSD"),
            decision_timestamp=TS,
        )
    finally:
        client.close()

    assert set(result.asset_inputs) == {"crypto:BTCUSDT", "fx:EURUSD"}
    assert set(result.return_series_by_asset) == {
        "crypto:BTCUSDT",
        "fx:EURUSD",
    }
    assert len(result.common_return_buckets) == 31
    assert len(result.return_series_by_asset["crypto:BTCUSDT"].values) == 30
    assert len(result.return_series_by_asset["fx:EURUSD"].values) == 30
    assert result.marks["crypto:BTCUSDT"] == pytest.approx(104.0)
    assert result.marks["fx:EURUSD"] == pytest.approx(1.08)

    # Current incomplete bucket may provide the current mark, but never the
    # covariance return history.
    assert result.common_return_buckets[-1] + BUCKET <= TS
    assert (
        result.return_series_by_asset["crypto:BTCUSDT"].observed_until
        < CURRENT_BUCKET * BUCKET
    )
    assert (
        result.return_series_by_asset["fx:EURUSD"].observed_until
        < CURRENT_BUCKET * BUCKET
    )

    btc_input = result.asset_inputs["crypto:BTCUSDT"]
    assert btc_input.source_kind_by_eye == {
        "eye-btc-deriv": "derivatives",
        "eye-btc-structure": "market_snapshot",
    }
    assert btc_input.snapshot["structure_state"] == pytest.approx(1.0)
    assert btc_input.snapshot["return_1"] > 0
    assert btc_input.snapshot["ema_slope"] > 0
    assert "rsi" not in btc_input.snapshot
    assert "relative_volume" not in btc_input.snapshot

    fx_input = result.asset_inputs["fx:EURUSD"]
    assert fx_input.source_kind_by_eye == {
        "eye-fx-macro": "macro",
        "eye-fx-structure": "market_snapshot",
    }
    assert fx_input.snapshot["structure_state"] == pytest.approx(-1.0)

    # DB observation id is retained inside provenance even though the immutable
    # SensorObservation derives its own content identity.
    btc_structure = next(
        row
        for row in btc_input.observations
        if row.eye_id == "eye-btc-structure"
    )
    assert "obs-eye-btc-structure" in btc_structure.source_ids
    assert "raw-eye-btc-structure" in btc_structure.source_ids

    assert len(requests) == 3
    assert all(request.method == "GET" for request in requests)
    assert result.read_only is True
    assert result.shadow_only is True
    assert result.live_execution is False


def test_latest_sensor_state_per_eye_prevents_old_direction_from_outvoting_current() -> None:
    asset = "crypto:BTCUSDT"
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_observations"
        ):
            return httpx.Response(200, json=[
                _sensor(
                    asset,
                    "eye-structure",
                    "price_structure",
                    "price_structure",
                    direction=1,
                    observed_at=TS - 10,
                    observation_id="obs-new",
                ),
                _sensor(
                    asset,
                    "eye-structure",
                    "price_structure",
                    "price_structure",
                    direction=-1,
                    observed_at=TS - 100,
                    observation_id="obs-old",
                ),
            ])
        if request.url.path.endswith("/rest/v1/brian_micro_book_ticks"):
            rows = _closed_price_points(
                asset,
                base=100.0,
                crypto=True,
            ) + [_current_mark(asset, price=104.0, crypto=True)]
            return httpx.Response(200, json=list(reversed(rows)))
        raise AssertionError(str(request.url))

    reader, client = _reader(handler)
    try:
        result = reader.load(
            asset_ids=(asset,),
            decision_timestamp=TS,
        )
    finally:
        client.close()

    observations = result.asset_inputs[asset].observations
    assert len(observations) == 1
    assert observations[0].direction == 1
    assert "obs-new" in observations[0].source_ids
    assert "obs-old" not in observations[0].source_ids


def test_ambiguous_same_timestamp_sensor_state_fails_closed() -> None:
    asset = "crypto:BTCUSDT"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_observations"
        ):
            return httpx.Response(200, json=[
                _sensor(
                    asset,
                    "eye-structure",
                    "price_structure",
                    "price_structure",
                    direction=1,
                    observation_id="obs-a",
                ),
                _sensor(
                    asset,
                    "eye-structure",
                    "price_structure",
                    "price_structure",
                    direction=-1,
                    observation_id="obs-b",
                ),
            ])
        raise AssertionError("price read must not occur")

    reader, client = _reader(handler)
    try:
        with pytest.raises(
            SupabaseGroundedMarketPrefetchResponseError,
            match="ambiguous latest sensor state",
        ):
            reader.load(
                asset_ids=(asset,),
                decision_timestamp=TS,
            )
    finally:
        client.close()


def test_fractional_sensor_direction_is_rejected_not_truncated() -> None:
    asset = "crypto:BTCUSDT"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_observations"
        ):
            row = _sensor(
                asset,
                "eye-structure",
                "price_structure",
                "price_structure",
            )
            row["direction"] = 0.5
            return httpx.Response(200, json=[row])
        raise AssertionError("price read must not occur")

    reader, client = _reader(handler)
    try:
        with pytest.raises(
            SupabaseGroundedMarketPrefetchResponseError,
            match="must be an integer",
        ):
            reader.load(
                asset_ids=(asset,),
                decision_timestamp=TS,
            )
    finally:
        client.close()


def test_future_sensor_response_is_rejected_even_if_server_filter_is_bypassed() -> None:
    asset = "crypto:BTCUSDT"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_observations"
        ):
            return httpx.Response(200, json=[
                _sensor(
                    asset,
                    "eye-future",
                    "price_structure",
                    "price_structure",
                    observed_at=TS + 1,
                )
            ])
        raise AssertionError("price read must not occur")

    reader, client = _reader(handler)
    try:
        with pytest.raises(
            SupabaseGroundedMarketPrefetchResponseError,
            match="post-decision observation",
        ):
            reader.load(
                asset_ids=(asset,),
                decision_timestamp=TS,
            )
    finally:
        client.close()


def test_stale_latest_mark_fails_closed() -> None:
    asset = "crypto:BTCUSDT"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_observations"
        ):
            return httpx.Response(200, json=[
                _sensor(
                    asset,
                    "eye-structure",
                    "price_structure",
                    "price_structure",
                )
            ])
        if request.url.path.endswith("/rest/v1/brian_micro_book_ticks"):
            return httpx.Response(
                200,
                json=_closed_price_points(
                    asset,
                    base=100.0,
                    crypto=True,
                ),
            )
        raise AssertionError(str(request.url))

    reader, client = _reader(handler, mark_max_age_seconds=60.0)
    try:
        with pytest.raises(
            SupabaseGroundedMarketPrefetchError,
            match="latest mark is stale",
        ):
            reader.load(
                asset_ids=(asset,),
                decision_timestamp=TS,
            )
    finally:
        client.close()


def test_insufficient_common_buckets_fails_without_padding_or_zero_fill() -> None:
    requests = []
    crypto = "crypto:BTCUSDT"
    fx = "fx:EURUSD"
    sensors = [
        _sensor(
            crypto,
            "eye-btc",
            "price_structure",
            "price_structure",
        ),
        _sensor(
            fx,
            "eye-fx",
            "price_structure",
            "price_structure",
        ),
    ]
    crypto_rows = _closed_price_points(
        crypto,
        base=100.0,
        count=31,
        crypto=True,
    ) + [_current_mark(crypto, price=104.0, crypto=True)]
    # Shift every FX bucket by one entire bucket so common history is short.
    fx_rows = _closed_price_points(
        fx,
        base=1.0,
        count=20,
        crypto=False,
    ) + [_current_mark(fx, price=1.05, crypto=False)]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_observations"
        ):
            return httpx.Response(200, json=sensors)
        if request.url.path.endswith("/rest/v1/brian_micro_book_ticks"):
            return httpx.Response(200, json=list(reversed(crypto_rows)))
        if request.url.path.endswith(
            "/rest/v1/brian_multiasset_market_marks"
        ):
            return httpx.Response(200, json=list(reversed(fx_rows)))
        raise AssertionError(str(request.url))

    reader, client = _reader(handler)
    try:
        with pytest.raises(
            SupabaseGroundedMarketPrefetchError,
            match="insufficient aligned PIT price buckets",
        ):
            reader.load(
                asset_ids=(crypto, fx),
                decision_timestamp=TS,
            )
    finally:
        client.close()


def test_conflicting_same_timestamp_prices_fail_closed() -> None:
    asset = "crypto:BTCUSDT"
    rows = _closed_price_points(
        asset,
        base=100.0,
        crypto=True,
    )
    conflict = dict(rows[-1])
    conflict["tick_id"] = "tick-conflict"
    conflict["observed_mid_price"] = float(rows[-1]["observed_mid_price"]) * 1.01
    rows.extend([
        conflict,
        _current_mark(asset, price=104.0, crypto=True),
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_observations"
        ):
            return httpx.Response(200, json=[
                _sensor(
                    asset,
                    "eye-structure",
                    "price_structure",
                    "price_structure",
                )
            ])
        if request.url.path.endswith("/rest/v1/brian_micro_book_ticks"):
            return httpx.Response(200, json=list(reversed(rows)))
        raise AssertionError(str(request.url))

    reader, client = _reader(handler)
    try:
        with pytest.raises(
            SupabaseGroundedMarketPrefetchResponseError,
            match="conflicting prices",
        ):
            reader.load(
                asset_ids=(asset,),
                decision_timestamp=TS,
            )
    finally:
        client.close()


def test_disallowed_table_has_no_generic_read_surface() -> None:
    reader, client = _reader(
        lambda request: (_ for _ in ()).throw(
            AssertionError("network must not be reached")
        )
    )
    try:
        with pytest.raises(
            SupabaseGroundedMarketPrefetchError,
            match="not allowed",
        ):
            reader._get(
                "brian_alpha_decisions",
                params={"select": "*"},
            )
    finally:
        client.close()


def test_http_error_is_sanitized_without_secret_echo() -> None:
    secret = "sb_secret_do_not_echo_phase110"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "code": "XX110",
                "message": "prefetch read failed",
                "hint": "retry",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reader = SupabaseGroundedMarketPrefetchReader(
        config=SupabaseGroundedMarketPrefetchConfig(
            project_url="https://example.supabase.co",
            key_source="SUPABASE_SECRET_KEY",
            return_observations=30,
        ),
        api_key=secret,
        client=client,
    )
    try:
        with pytest.raises(
            SupabaseGroundedMarketPrefetchResponseError,
        ) as exc:
            reader.load(
                asset_ids=("crypto:BTCUSDT",),
                decision_timestamp=TS,
            )
    finally:
        client.close()

    assert secret not in str(exc.value)
    assert "XX110" in str(exc.value)


def test_from_env_rejects_publishable_key() -> None:
    with pytest.raises(
        SupabaseRecoveryRpcConfigurationError,
        match="publishable",
    ):
        SupabaseGroundedMarketPrefetchReader.from_env(
            env={
                "SUPABASE_URL": "https://example.supabase.co",
                "SUPABASE_SECRET_KEY": "sb_publishable_not_server",
            }
        )

def test_external_grounded_price_points_reuse_phase110_sensor_and_alignment_logic() -> None:
    asset = "crypto:BTCUSDT"
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_observations"
        ):
            return httpx.Response(200, json=[
                _sensor(
                    asset,
                    "eye-external",
                    "price_structure",
                    "price_structure",
                )
            ])
        raise AssertionError(
            "external price composition must not query a Phase110 price table"
        )

    points = tuple(
        GroundedPricePoint(
            asset_id=asset,
            observed_at=(CURRENT_BUCKET - 31 + index) * BUCKET
                + (BUCKET - 1),
            price=100.0 + index,
            source_id=f"external-kline-{index:03d}",
        )
        for index in range(31)
    ) + (
        GroundedPricePoint(
            asset_id=asset,
            observed_at=TS - 10,
            price=140.0,
            source_id="external-current-mark",
        ),
    )

    reader, client = _reader(handler)
    try:
        result = reader.load_with_price_points(
            asset_ids=(asset,),
            decision_timestamp=TS,
            price_points_by_asset={asset: points},
        )
    finally:
        client.close()

    assert len(requests) == 1
    assert requests[0].url.path.endswith(
        "/rest/v1/brian_sensor_observations"
    )
    assert len(result.return_series_by_asset[asset].values) == 30
    assert result.return_series_by_asset[asset].observed_until < (
        CURRENT_BUCKET * BUCKET
    )
    assert result.marks[asset] == pytest.approx(140.0)
    assert result.asset_inputs[asset].snapshot["structure_state"] == pytest.approx(1.0)


def test_external_grounded_price_points_reject_future_row_before_sensor_read() -> None:
    asset = "crypto:BTCUSDT"
    network_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        return httpx.Response(200, json=[])

    reader, client = _reader(handler)
    try:
        with pytest.raises(
            SupabaseGroundedMarketPrefetchError,
            match="after decision time",
        ):
            reader.load_with_price_points(
                asset_ids=(asset,),
                decision_timestamp=TS,
                price_points_by_asset={
                    asset: (
                        GroundedPricePoint(
                            asset_id=asset,
                            observed_at=TS + 1,
                            price=100.0,
                            source_id="future-price",
                        ),
                    )
                },
            )
    finally:
        client.close()

    assert network_calls == 0

def test_sensor_query_filters_to_supported_live_horizons_and_families() -> None:
    asset = "crypto:BTCUSDT"
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_observations"
        ):
            seen["horizon"] = request.url.params.get("horizon")
            seen["sensor_family"] = request.url.params.get("sensor_family")
            return httpx.Response(200, json=[
                _sensor(
                    asset,
                    "eye-structure",
                    "price_structure",
                    "price_structure",
                )
            ])
        if request.url.path.endswith("/rest/v1/brian_micro_book_ticks"):
            rows = _closed_price_points(
                asset,
                base=100.0,
                crypto=True,
            ) + [_current_mark(asset, price=104.0, crypto=True)]
            return httpx.Response(200, json=list(reversed(rows)))
        raise AssertionError(str(request.url))

    reader, client = _reader(handler)
    try:
        reader.load(
            asset_ids=(asset,),
            decision_timestamp=TS,
        )
    finally:
        client.close()

    assert seen["horizon"] is not None
    assert "EVENT_DRIVEN" not in seen["horizon"]
    assert "DAILY" not in seen["horizon"]
    assert "FAST_5_30M" in seen["horizon"]
    assert "MICRO_1_5M" in seen["horizon"]
    assert "taker_flow" in seen["sensor_family"]
    assert "open_interest" in seen["sensor_family"]
    assert "funding_crowding" in seen["sensor_family"]


@pytest.mark.parametrize(
    ("family", "group"),
    [
        ("taker_flow", "derivatives_taker"),
        ("open_interest", "derivatives_oi"),
        ("funding_crowding", "derivatives_funding"),
    ],
)
def test_live_derivative_sensor_families_map_to_derivatives_source_kind(
    family,
    group,
) -> None:
    asset = "crypto:BTCUSDT"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_observations"
        ):
            return httpx.Response(200, json=[
                _sensor(
                    asset,
                    f"eye-{family}",
                    family,
                    group,
                )
            ])
        if request.url.path.endswith("/rest/v1/brian_micro_book_ticks"):
            rows = _closed_price_points(
                asset,
                base=100.0,
                crypto=True,
            ) + [_current_mark(asset, price=104.0, crypto=True)]
            return httpx.Response(200, json=list(reversed(rows)))
        raise AssertionError(str(request.url))

    reader, client = _reader(handler)
    try:
        result = reader.load(
            asset_ids=(asset,),
            decision_timestamp=TS,
        )
    finally:
        client.close()

    assert result.asset_inputs[asset].source_kind_by_eye == {
        f"eye-{family}": "derivatives",
    }


def test_from_env_prefers_scoped_sensor_supabase_credentials() -> None:
    scoped_url = "https://sensor-project.supabase.co"
    scoped_key = "sb_secret_sensor_phase110_abcdefghijklmnopqrstuvwxyz"
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=[])
        )
    )
    reader = SupabaseGroundedMarketPrefetchReader.from_env(
        env={
            "SUPABASE_URL": "https://generic.supabase.co",
            "SUPABASE_SECRET_KEY":
                "sb_secret_generic_phase110_abcdefghijklmnopqrstuvwxyz",
            "BRIAN_SENSOR_SUPABASE_URL": scoped_url,
            "BRIAN_SENSOR_SUPABASE_SECRET_KEY": scoped_key,
        },
        client=client,
    )
    try:
        assert reader.config.project_url == scoped_url
        assert reader.config.key_source == "BRIAN_SENSOR_SUPABASE_SECRET_KEY"
        assert reader._api_key == scoped_key
    finally:
        client.close()
