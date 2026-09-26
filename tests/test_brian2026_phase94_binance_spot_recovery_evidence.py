from __future__ import annotations

import httpx
import pytest

from brian2026.phase94_binance_spot_recovery_evidence import (
    PUBLIC_MARKET_HOSTS,
    BinanceSpotRecoveryEvidenceError,
    BinanceSpotRecoveryEvidenceProvider,
    BinanceSpotRecoveryRateLimitError,
    BinanceSpotRecoveryTransportError,
)


def _exchange(symbol="BTCUSDT", *, status="TRADING", spot=True, with_notional=True):
    filters = [
        {
            "filterType": "PRICE_FILTER",
            "minPrice": "0.01",
            "maxPrice": "1000000",
            "tickSize": "0.10",
        },
        {
            "filterType": "MIN_NOTIONAL",
            "minNotional": "5.00",
        },
    ]
    if with_notional:
        filters.append({
            "filterType": "NOTIONAL",
            "minNotional": "10.00",
            "maxNotional": "1000000.00",
        })
    return {
        "timezone": "UTC",
        "symbols": [{
            "symbol": symbol,
            "status": status,
            "isSpotTradingAllowed": spot,
            "filters": filters,
        }],
    }


def _depth(*, bid="99.9", ask="100.1", update=123):
    return {
        "lastUpdateId": update,
        "bids": [[bid, "2.0"], ["99.8", "3.0"]],
        "asks": [[ask, "2.5"], ["100.2", "4.0"]],
    }


def _client(handler):
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )


def test_valid_public_evidence_builds_phase57_market_and_phase56_limits() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/v3/exchangeInfo":
            return httpx.Response(200, json=_exchange())
        if request.url.path == "/api/v3/depth":
            return httpx.Response(200, json=_depth())
        raise AssertionError(request.url)

    provider = BinanceSpotRecoveryEvidenceProvider(
        client=_client(handler),
        clock=lambda: 123.5,
    )
    bundle = provider.collect(["BTCUSDT"])
    assert len(bundle.assets) == 1
    row = bundle.assets[0]
    assert row.asset_id == "BTCUSDT"
    assert row.market.reference_price == pytest.approx(100.0)
    assert row.market.tick_size == pytest.approx(0.10)
    assert row.market.snapshots[0].timestamp == 123.5
    assert row.market.snapshots[0].bids[0].price == pytest.approx(99.9)
    assert row.market.snapshots[0].asks[0].quantity == pytest.approx(2.5)
    # NOTIONAL takes precedence over legacy MIN_NOTIONAL when both are present.
    assert row.risk_limits.min_notional == pytest.approx(10.0)
    assert row.mark == pytest.approx(100.0)
    assert row.depth_last_update_id == "123"
    assert row.source_host == PUBLIC_MARKET_HOSTS[0]
    assert bundle.markets["BTCUSDT"] is row.market
    assert bundle.risk_limits_by_asset["BTCUSDT"] is row.risk_limits
    assert bundle.marks == {"BTCUSDT": pytest.approx(100.0)}

    for request in seen:
        headers = {k.lower(): v for k, v in request.headers.items()}
        assert "authorization" not in headers
        assert "x-mbx-apikey" not in headers
        assert request.method == "GET"
        assert request.url.path in {"/api/v3/exchangeInfo", "/api/v3/depth"}


def test_legacy_min_notional_is_used_when_notional_filter_is_absent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("exchangeInfo"):
            return httpx.Response(200, json=_exchange(with_notional=False))
        return httpx.Response(200, json=_depth())

    bundle = BinanceSpotRecoveryEvidenceProvider(
        client=_client(handler),
        clock=lambda: 1.0,
    ).collect(["BTCUSDT"])
    assert bundle.assets[0].risk_limits.min_notional == pytest.approx(5.0)


def test_assets_are_deduplicated_sorted_and_empty_input_is_valid() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        symbol = request.url.params["symbol"]
        if request.url.path.endswith("exchangeInfo"):
            return httpx.Response(200, json=_exchange(symbol))
        return httpx.Response(200, json=_depth())

    provider = BinanceSpotRecoveryEvidenceProvider(
        client=_client(handler),
        clock=lambda: 1.0,
    )
    assert provider.collect([]).assets == ()
    bundle = provider.collect(["ETHUSDT", "btcusdt", "BTCUSDT"])
    assert [row.asset_id for row in bundle.assets] == ["BTCUSDT", "ETHUSDT"]
    assert len(calls) == 4


def test_only_spot_usdt_assets_and_exact_public_routes_are_allowed() -> None:
    provider = BinanceSpotRecoveryEvidenceProvider(
        client=_client(lambda request: httpx.Response(200, json={})),
    )
    with pytest.raises(BinanceSpotRecoveryEvidenceError, match="unsupported recovery asset"):
        provider.collect(["fx:EURUSD"])
    with pytest.raises(BinanceSpotRecoveryEvidenceError, match="forbidden"):
        provider._get("/api/v3/order", {"symbol": "BTCUSDT"})


def test_429_or_418_stops_immediately_without_host_hopping() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            headers={"Retry-After": "5"},
            json={"code": -1003, "msg": "rate limit"},
        )

    provider = BinanceSpotRecoveryEvidenceProvider(client=_client(handler))
    with pytest.raises(BinanceSpotRecoveryRateLimitError, match="retry_after=5"):
        provider.collect(["BTCUSDT"])
    assert calls == 1


def test_5xx_can_fail_over_to_next_public_market_host() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.host == "data-api.binance.vision":
            return httpx.Response(503, json={"msg": "unavailable"})
        if request.url.path.endswith("exchangeInfo"):
            return httpx.Response(200, json=_exchange())
        return httpx.Response(200, json=_depth())

    bundle = BinanceSpotRecoveryEvidenceProvider(
        client=_client(handler),
        clock=lambda: 1.0,
    ).collect(["BTCUSDT"])
    assert bundle.assets[0].source_host == "https://api.binance.com"
    assert any("data-api.binance.vision" in url for url in calls)
    assert any("api.binance.com" in url for url in calls)


def test_redirect_is_not_followed() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            307,
            headers={"location": "https://attacker.invalid/api/v3/depth"},
        )

    provider = BinanceSpotRecoveryEvidenceProvider(client=_client(handler))
    with pytest.raises(BinanceSpotRecoveryTransportError, match="HTTP 307"):
        provider.collect(["BTCUSDT"])
    assert calls == 1


@pytest.mark.parametrize(
    ("exchange", "match"),
    [
        (_exchange(status="BREAK"), "not TRADING"),
        (_exchange(spot=False), "Spot trading is disabled"),
        ({"symbols": []}, "exactly one requested symbol"),
    ],
)
def test_exchange_rule_failures_are_closed(exchange, match) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("exchangeInfo"):
            return httpx.Response(200, json=exchange)
        return httpx.Response(200, json=_depth())

    provider = BinanceSpotRecoveryEvidenceProvider(client=_client(handler))
    with pytest.raises(BinanceSpotRecoveryEvidenceError, match=match):
        provider.collect(["BTCUSDT"])


def test_missing_price_or_notional_filter_is_rejected() -> None:
    missing_price = _exchange()
    missing_price["symbols"][0]["filters"] = [
        x for x in missing_price["symbols"][0]["filters"]
        if x["filterType"] != "PRICE_FILTER"
    ]
    missing_notional = _exchange()
    missing_notional["symbols"][0]["filters"] = [
        x for x in missing_notional["symbols"][0]["filters"]
        if x["filterType"] not in {"NOTIONAL", "MIN_NOTIONAL"}
    ]

    for exchange, match in (
        (missing_price, "PRICE_FILTER"),
        (missing_notional, "NOTIONAL/MIN_NOTIONAL"),
    ):
        def handler(request: httpx.Request, exchange=exchange) -> httpx.Response:
            if request.url.path.endswith("exchangeInfo"):
                return httpx.Response(200, json=exchange)
            return httpx.Response(200, json=_depth())

        with pytest.raises(BinanceSpotRecoveryEvidenceError, match=match):
            BinanceSpotRecoveryEvidenceProvider(
                client=_client(handler)
            ).collect(["BTCUSDT"])


@pytest.mark.parametrize(
    ("depth", "match"),
    [
        (_depth(bid="100.1", ask="100.1"), "crossed/locked"),
        (
            {
                "lastUpdateId": 1,
                "bids": [["99.8", "1"], ["99.9", "1"]],
                "asks": [["100.1", "1"], ["100.2", "1"]],
            },
            "strictly descending",
        ),
        (
            {
                "lastUpdateId": 1,
                "bids": [["99.9", "1"], ["99.8", "1"]],
                "asks": [["100.2", "1"], ["100.1", "1"]],
            },
            "strictly ascending",
        ),
        (
            {
                "lastUpdateId": "bad",
                "bids": [["99.9", "1"]],
                "asks": [["100.1", "1"]],
            },
            "lastUpdateId",
        ),
        (
            {
                "lastUpdateId": 1,
                "bids": [["99.9", "0"]],
                "asks": [["100.1", "1"]],
            },
            "positive",
        ),
    ],
)
def test_depth_shape_is_fail_closed(depth, match) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("exchangeInfo"):
            return httpx.Response(200, json=_exchange())
        return httpx.Response(200, json=depth)

    with pytest.raises(BinanceSpotRecoveryEvidenceError, match=match):
        BinanceSpotRecoveryEvidenceProvider(
            client=_client(handler)
        ).collect(["BTCUSDT"])


def test_wide_spread_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("exchangeInfo"):
            return httpx.Response(200, json=_exchange())
        return httpx.Response(200, json={
            "lastUpdateId": 1,
            "bids": [["99.0", "1"]],
            "asks": [["101.0", "1"]],
        })

    with pytest.raises(BinanceSpotRecoveryEvidenceError, match="spread"):
        BinanceSpotRecoveryEvidenceProvider(
            client=_client(handler),
            max_spread_bps=30.0,
        ).collect(["BTCUSDT"])


def test_asset_budget_and_depth_limit_configuration_are_bounded() -> None:
    provider = BinanceSpotRecoveryEvidenceProvider(
        client=_client(lambda request: httpx.Response(500)),
        max_assets=1,
    )
    with pytest.raises(BinanceSpotRecoveryEvidenceError, match="exceeds"):
        provider.collect(["BTCUSDT", "ETHUSDT"])

    with pytest.raises(ValueError, match="depth limit"):
        BinanceSpotRecoveryEvidenceProvider(
            client=_client(lambda request: httpx.Response(500)),
            depth_limit=123,
        )


def test_all_hosts_unavailable_returns_bounded_transport_failure() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    provider = BinanceSpotRecoveryEvidenceProvider(client=_client(handler))
    with pytest.raises(BinanceSpotRecoveryTransportError, match="unavailable"):
        provider.collect(["BTCUSDT"])
    # One exchangeInfo attempt per allowlisted public host, then fail closed.
    assert calls == len(PUBLIC_MARKET_HOSTS)
