from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from brian2026.phase91_supabase_rpc_transport import (
    SupabaseRecoveryRpcConfigurationError,
)
from brian2026.phase108_supabase_lagged_edge_reader import (
    READINESS_COST_COMPILER_VERSION,
    SupabaseLaggedEdgeReader,
    SupabaseLaggedEdgeReaderConfig,
    SupabaseLaggedEdgeReaderError,
    SupabaseLaggedEdgeReaderResponseError,
)


TS = 1_790_000_000.0


def _iso(value: float) -> str:
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _reliability_row(group: str, *, generated=TS - 1800, window=TS - 3600):
    return {
        "independent_group": group,
        "sample_count": 250,
        "bayesian_hit_rate_beta10_10": 0.62,
        "avg_signed_bps": 30.0,
        "avg_cost_adjusted_signed_bps": 25.0,
        "outcome_horizon_seconds": 900,
        "window_end": _iso(window),
        "generated_at": _iso(generated),
        "evidence_class": "PROSPECTIVE_DEVELOPMENT_SHADOW",
        "shadow_only": True,
        "live_execution": False,
    }


def _cost_row(asset: str, *, observed=TS - 10, cost=4.0):
    return {
        "asset_id": asset,
        "observed_at": _iso(observed),
        "estimated_round_trip_cost_bps": cost,
        "fillable": True,
        "quality": "L2_OBSERVED",
        "shadow_only": True,
        "live_execution": False,
    }


def _reader(handler, *, cost_max_age=300.0):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    reader = SupabaseLaggedEdgeReader(
        config=SupabaseLaggedEdgeReaderConfig(
            project_url="https://example.supabase.co",
            key_source="SUPABASE_SECRET_KEY",
            cost_max_age_seconds=cost_max_age,
            outcome_horizon_seconds=900,
        ),
        api_key="sb_secret_phase108",
        client=client,
    )
    return reader, client


def test_load_contexts_uses_one_pit_window_and_latest_fresh_costs_read_only() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        assert request.headers["apikey"] == "sb_secret_phase108"
        assert "authorization" not in request.headers
        assert "sb_secret_phase108" not in str(request.url)

        if request.url.path.endswith(
            "/rest/v1/brian_sensor_reliability_shadow_snapshots"
        ):
            select = request.url.params.get("select", "")
            if select == "window_end,generated_at":
                return httpx.Response(
                    200,
                    json=[{
                        "window_end": _iso(TS - 3600),
                        "generated_at": _iso(TS - 1800),
                    }],
                )
            assert request.url.params["window_end"] == (
                f"eq.{_iso(TS - 3600)}"
            )
            assert request.url.params["generated_at"] == (
                f"eq.{_iso(TS - 1800)}"
            )
            assert request.url.params["outcome_horizon_seconds"] == "eq.900"
            return httpx.Response(
                200,
                json=[
                    _reliability_row("derivatives"),
                    _reliability_row("price_structure"),
                ],
            )

        if request.url.path.endswith(
            "/rest/v1/brian_dynamic_cost_quotes"
        ):
            assert request.url.params["fillable"] == "eq.true"
            assert request.url.params["quality"] == "neq.UNAVAILABLE"
            assert request.url.params["compiler_version"] == (
                f"eq.{READINESS_COST_COMPILER_VERSION}"
            )
            return httpx.Response(
                200,
                json=[
                    _cost_row("BTCUSDT", observed=TS - 10, cost=4.5),
                    _cost_row("ETHUSDT", observed=TS - 1000, cost=5.0),
                ],
            )
        raise AssertionError(str(request.url))

    reader, client = _reader(handler, cost_max_age=300)
    try:
        contexts = reader.load_contexts(
            groups_by_asset={
                "BTCUSDT": ("price_structure", "derivatives"),
                "ETHUSDT": ("price_structure",),
            },
            decision_timestamp=TS,
        )
    finally:
        client.close()

    btc = contexts["BTCUSDT"]
    assert {row.group for row in btc.reliability} == {
        "price_structure",
        "derivatives",
    }
    assert btc.round_trip_cost_bps == pytest.approx(4.5)
    assert btc.cost_observed_at == pytest.approx(TS - 10)

    eth = contexts["ETHUSDT"]
    assert {row.group for row in eth.reliability} == {"price_structure"}
    assert eth.round_trip_cost_bps is None
    assert eth.cost_observed_at is None
    assert len(requests) == 3
    assert all(request.method == "GET" for request in requests)


def test_no_reliability_window_returns_empty_reliability_without_fabrication() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_reliability_shadow_snapshots"
        ):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=[_cost_row("BTCUSDT")])

    reader, client = _reader(handler)
    try:
        contexts = reader.load_contexts(
            groups_by_asset={"BTCUSDT": ("price_structure",)},
            decision_timestamp=TS,
        )
    finally:
        client.close()

    assert contexts["BTCUSDT"].reliability == ()
    assert contexts["BTCUSDT"].round_trip_cost_bps == pytest.approx(4.0)


def test_future_reliability_window_from_server_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_reliability_shadow_snapshots"
        ):
            return httpx.Response(
                200,
                json=[{
                    "window_end": _iso(TS + 1),
                    "generated_at": _iso(TS - 10),
                }],
            )
        raise AssertionError("cost must not be read")

    reader, client = _reader(handler)
    try:
        with pytest.raises(
            SupabaseLaggedEdgeReaderResponseError,
            match="after decision time",
        ):
            reader.load_contexts(
                groups_by_asset={"BTCUSDT": ("price_structure",)},
                decision_timestamp=TS,
            )
    finally:
        client.close()


def test_second_query_cannot_escape_selected_pit_window() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json=[{
                    "window_end": _iso(TS - 3600),
                    "generated_at": _iso(TS - 1800),
                }],
            )
        if calls == 2:
            return httpx.Response(
                200,
                json=[
                    _reliability_row(
                        "price_structure",
                        generated=TS + 1,
                    )
                ],
            )
        raise AssertionError("cost must not be read")

    reader, client = _reader(handler)
    try:
        with pytest.raises(
            SupabaseLaggedEdgeReaderResponseError,
            match="escaped selected PIT window",
        ):
            reader.load_contexts(
                groups_by_asset={"BTCUSDT": ("price_structure",)},
                decision_timestamp=TS,
            )
    finally:
        client.close()


def test_ambiguous_same_group_reliability_rows_fail_closed() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json=[{
                    "window_end": _iso(TS - 3600),
                    "generated_at": _iso(TS - 1800),
                }],
            )
        if calls == 2:
            return httpx.Response(
                200,
                json=[
                    _reliability_row("price_structure"),
                    _reliability_row("price_structure"),
                ],
            )
        raise AssertionError("cost must not be read")

    reader, client = _reader(handler)
    try:
        with pytest.raises(
            SupabaseLaggedEdgeReaderResponseError,
            match="ambiguous reliability rows",
        ):
            reader.load_contexts(
                groups_by_asset={"BTCUSDT": ("price_structure",)},
                decision_timestamp=TS,
            )
    finally:
        client.close()


def test_future_cost_response_fails_closed_even_if_server_filter_was_bypassed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_reliability_shadow_snapshots"
        ):
            if request.url.params.get("select") == "window_end,generated_at":
                return httpx.Response(200, json=[])
            raise AssertionError("no second reliability query expected")
        return httpx.Response(
            200,
            json=[_cost_row("BTCUSDT", observed=TS + 1)],
        )

    reader, client = _reader(handler)
    try:
        with pytest.raises(
            SupabaseLaggedEdgeReaderResponseError,
            match="after decision time",
        ):
            reader.load_contexts(
                groups_by_asset={"BTCUSDT": ()},
                decision_timestamp=TS,
            )
    finally:
        client.close()


def test_http_error_is_sanitized_and_secret_is_not_echoed() -> None:
    secret = "sb_secret_do_not_echo_108"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "code": "XX001",
                "message": "database read failed",
                "hint": "retry later",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reader = SupabaseLaggedEdgeReader(
        config=SupabaseLaggedEdgeReaderConfig(
            project_url="https://example.supabase.co",
            key_source="SUPABASE_SECRET_KEY",
        ),
        api_key=secret,
        client=client,
    )
    try:
        with pytest.raises(SupabaseLaggedEdgeReaderResponseError) as exc:
            reader.load_contexts(
                groups_by_asset={"BTCUSDT": ("price_structure",)},
                decision_timestamp=TS,
            )
    finally:
        client.close()

    assert secret not in str(exc.value)
    assert "XX001" in str(exc.value)


def test_disallowed_table_has_no_generic_read_surface() -> None:
    reader, client = _reader(
        lambda request: (_ for _ in ()).throw(
            AssertionError("network must not be reached")
        )
    )
    try:
        with pytest.raises(
            SupabaseLaggedEdgeReaderError,
            match="not allowed",
        ):
            reader._get(
                "brian_alpha_decisions",
                params={"select": "*"},
            )
    finally:
        client.close()


def test_from_env_rejects_publishable_key() -> None:
    with pytest.raises(
        SupabaseRecoveryRpcConfigurationError,
        match="publishable",
    ):
        SupabaseLaggedEdgeReader.from_env(
            env={
                "SUPABASE_URL": "https://example.supabase.co",
                "SUPABASE_SECRET_KEY": "sb_publishable_not_server",
            }
        )

def test_reliability_and_cost_can_use_distinct_supabase_projects_and_keys() -> None:
    reliability_url = "https://edge-project.supabase.co"
    cost_url = "https://cost-project.supabase.co"
    edge_key = "sb_secret_edge_phase108_abcdefghijklmnopqrstuvwxyz"
    cost_key = "sb_secret_cost_phase108_abcdefghijklmnopqrstuvwxyz"
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((
            request.url.host,
            request.url.path,
            request.headers.get("apikey"),
        ))
        if request.url.path.endswith(
            "/rest/v1/brian_sensor_reliability_shadow_snapshots"
        ):
            if request.url.params.get("select") == "window_end,generated_at":
                return httpx.Response(
                    200,
                    json=[{
                        "window_end": _iso(TS - 3600),
                        "generated_at": _iso(TS - 1800),
                    }],
                )
            return httpx.Response(
                200,
                json=[_reliability_row("price_structure")],
            )
        if request.url.path.endswith("/rest/v1/brian_dynamic_cost_quotes"):
            return httpx.Response(
                200,
                json=[_cost_row("crypto:BTCUSDT", cost=3.5)],
            )
        raise AssertionError(str(request.url))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reader = SupabaseLaggedEdgeReader(
        config=SupabaseLaggedEdgeReaderConfig(
            project_url=reliability_url,
            key_source="SUPABASE_SECRET_KEY",
            cost_project_url=cost_url,
            cost_key_source="SUPABASE_SECRET_KEY",
            outcome_horizon_seconds=900,
        ),
        api_key=edge_key,
        cost_api_key=cost_key,
        client=client,
    )
    try:
        contexts = reader.load_contexts(
            groups_by_asset={
                "crypto:BTCUSDT": ("price_structure",),
            },
            cost_asset_id_by_asset={
                "crypto:BTCUSDT": "crypto:BTCUSDT",
            },
            decision_timestamp=TS,
        )
    finally:
        client.close()

    assert contexts["crypto:BTCUSDT"].round_trip_cost_bps == pytest.approx(3.5)
    reliability_hosts = {
        host for host, path, _ in seen
        if path.endswith("/brian_sensor_reliability_shadow_snapshots")
    }
    cost_hosts = {
        host for host, path, _ in seen
        if path.endswith("/brian_dynamic_cost_quotes")
    }
    assert reliability_hosts == {"edge-project.supabase.co"}
    assert cost_hosts == {"cost-project.supabase.co"}
    assert all(
        key == edge_key
        for _, path, key in seen
        if path.endswith("/brian_sensor_reliability_shadow_snapshots")
    )
    assert all(
        key == cost_key
        for _, path, key in seen
        if path.endswith("/brian_dynamic_cost_quotes")
    )


def test_from_env_supports_scoped_edge_and_cost_supabase_sources() -> None:
    edge_url = "https://edge-project.supabase.co"
    cost_url = "https://cost-project.supabase.co"
    edge_key = "sb_secret_edge_env_phase108_abcdefghijklmnopqrstuvwxyz"
    cost_key = "sb_secret_cost_env_phase108_abcdefghijklmnopqrstuvwxyz"
    reader = SupabaseLaggedEdgeReader.from_env(
        env={
            "SUPABASE_URL": "https://generic.supabase.co",
            "SUPABASE_SECRET_KEY": "sb_secret_generic_abcdefghijklmnopqrstuvwxyz",
            "BRIAN_EDGE_SUPABASE_URL": edge_url,
            "BRIAN_EDGE_SUPABASE_SECRET_KEY": edge_key,
            "BRIAN_COST_SUPABASE_URL": cost_url,
            "BRIAN_COST_SUPABASE_SECRET_KEY": cost_key,
        },
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=[])
            )
        ),
    )
    try:
        assert reader.config.project_url == edge_url
        assert reader.config.key_source == "BRIAN_EDGE_SUPABASE_SECRET_KEY"
        assert reader.config.cost_project_url == cost_url
        assert reader.config.cost_key_source == "BRIAN_COST_SUPABASE_SECRET_KEY"
        assert reader._api_key == edge_key
        assert reader._cost_api_key == cost_key
    finally:
        reader._client.close()


def test_partial_cost_scope_configuration_fails_closed() -> None:
    with pytest.raises(
        SupabaseRecoveryRpcConfigurationError,
        match="server key",
    ):
        SupabaseLaggedEdgeReader.from_env(
            env={
                "BRIAN_EDGE_SUPABASE_URL": "https://edge.supabase.co",
                "BRIAN_EDGE_SUPABASE_SECRET_KEY":
                    "sb_secret_edge_only_abcdefghijklmnopqrstuvwxyz",
                "BRIAN_COST_SUPABASE_URL": "https://cost.supabase.co",
            }
        )

    with pytest.raises(
        SupabaseRecoveryRpcConfigurationError,
        match="BRIAN_COST_SUPABASE_URL",
    ):
        SupabaseLaggedEdgeReader.from_env(
            env={
                "BRIAN_EDGE_SUPABASE_URL": "https://edge.supabase.co",
                "BRIAN_EDGE_SUPABASE_SECRET_KEY":
                    "sb_secret_edge_only_abcdefghijklmnopqrstuvwxyz",
                "BRIAN_COST_SUPABASE_SECRET_KEY":
                    "sb_secret_cost_only_abcdefghijklmnopqrstuvwxyz",
            }
        )
