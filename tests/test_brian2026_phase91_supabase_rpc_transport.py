from __future__ import annotations

import json

import httpx
import pytest

from brian2026.phase91_supabase_rpc_transport import (
    SupabaseRecoveryRpcConfigurationError,
    SupabaseRecoveryRpcConfig,
    SupabaseRecoveryRpcResponseError,
    SupabaseRecoveryRpcTransport,
    SupabaseRecoveryRpcTransportError,
)


SECRET = "sb_secret_abcdefghijklmnopqrstuvwxyz"
LEGACY = "eyJlegacy-service-role-test"
URL = "https://example-project.supabase.co"
RPC = "brian_read_shadow_recovery_admission"


def _client(handler):
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )


def test_modern_secret_key_uses_apikey_header_without_bearer_authorization() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "runtime_id": "runtime-91",
                "status": "OPEN",
                "blocked": False,
            },
        )

    transport = SupabaseRecoveryRpcTransport(
        config=SupabaseRecoveryRpcConfig(
            project_url=URL,
            key_source="SUPABASE_SECRET_KEY",
        ),
        api_key=SECRET,
        client=_client(handler),
    )
    row = transport(
        RPC,
        {"p_runtime_id": "runtime-91"},
    )

    assert row["status"] == "OPEN"
    assert seen["url"] == f"{URL}/rest/v1/rpc/{RPC}"
    assert seen["headers"]["apikey"] == SECRET
    assert "authorization" not in seen["headers"]
    assert seen["headers"]["content-type"] == "application/json"
    assert seen["body"] == {"p_runtime_id": "runtime-91"}


def test_legacy_service_role_key_remains_migration_compatible() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["apikey"] = request.headers.get("apikey")
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    transport = SupabaseRecoveryRpcTransport.from_env(
        env={
            "SUPABASE_URL": URL,
            "SUPABASE_SERVICE_ROLE_KEY": LEGACY,
        },
        client=_client(handler),
    )
    assert transport(RPC, {"p_runtime_id": "runtime-91"}) == {"ok": True}
    assert seen["apikey"] == LEGACY
    assert seen["authorization"] is None
    assert transport.config.key_source == "SUPABASE_SERVICE_ROLE_KEY"


def test_modern_secret_key_takes_precedence_over_legacy_key() -> None:
    transport = SupabaseRecoveryRpcTransport.from_env(
        env={
            "SUPABASE_URL": URL,
            "SUPABASE_SECRET_KEY": SECRET,
            "SUPABASE_SERVICE_ROLE_KEY": LEGACY,
        },
        client=_client(lambda request: httpx.Response(200, json={"ok": True})),
    )
    assert transport.config.key_source == "SUPABASE_SECRET_KEY"
    assert transport._api_key == SECRET


def test_edge_secret_keys_json_default_is_supported() -> None:
    transport = SupabaseRecoveryRpcTransport.from_env(
        env={
            "SUPABASE_URL": URL,
            "SUPABASE_SECRET_KEYS": json.dumps({"default": SECRET}),
        },
        client=_client(lambda request: httpx.Response(200, json={"ok": True})),
    )
    assert transport.config.key_source == "SUPABASE_SECRET_KEYS.default"
    assert transport._api_key == SECRET


@pytest.mark.parametrize(
    "env,match",
    [
        (
            {
                "SUPABASE_URL": URL,
                "SUPABASE_SECRET_KEY": "sb_publishable_public",
            },
            "publishable",
        ),
        (
            {
                "SUPABASE_URL": URL,
                "SUPABASE_SECRET_KEY": LEGACY,
            },
            "sb_secret_",
        ),
        (
            {
                "SUPABASE_URL": URL,
                "SUPABASE_SECRET_KEYS": "not-json",
            },
            "valid JSON",
        ),
        (
            {
                "SUPABASE_URL": URL,
                "SUPABASE_SECRET_KEYS": json.dumps({"other": SECRET}),
            },
            "default",
        ),
        (
            {"SUPABASE_URL": URL},
            "secret key",
        ),
        (
            {"SUPABASE_SECRET_KEY": SECRET},
            "SUPABASE_URL",
        ),
    ],
)
def test_invalid_environment_configuration_fails_closed(env, match) -> None:
    with pytest.raises(SupabaseRecoveryRpcConfigurationError, match=match):
        SupabaseRecoveryRpcTransport.from_env(
            env=env,
            client=_client(lambda request: httpx.Response(200, json={"ok": True})),
        )


def test_remote_plain_http_project_url_is_rejected_but_localhost_is_allowed() -> None:
    with pytest.raises(ValueError, match="https"):
        SupabaseRecoveryRpcConfig(
            project_url="http://example-project.supabase.co",
            key_source="SUPABASE_SERVICE_ROLE_KEY",
        )

    config = SupabaseRecoveryRpcConfig(
        project_url="http://127.0.0.1:54321",
        key_source="SUPABASE_SERVICE_ROLE_KEY",
    )
    assert config.project_url.startswith("http://127.0.0.1")


def test_only_exact_phase70_and_phase81_to_phase87_rpc_surface_is_allowed() -> None:
    transport = SupabaseRecoveryRpcTransport(
        config=SupabaseRecoveryRpcConfig(
            project_url=URL,
            key_source="SUPABASE_SECRET_KEY",
        ),
        api_key=SECRET,
        client=_client(lambda request: httpx.Response(200, json={"ok": True})),
    )
    assert transport(
        "brian_read_shadow_runtime_checkpoint",
        {"p_runtime_id": "runtime-91"},
    ) == {"ok": True}

    with pytest.raises(SupabaseRecoveryRpcConfigurationError, match="not allowed"):
        transport("dangerous_admin_rpc", {})


def test_http_error_is_sanitized_and_never_leaks_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "code": "42501",
                "message": "permission denied for function",
                "hint": "grant execute to service_role",
            },
        )

    transport = SupabaseRecoveryRpcTransport(
        config=SupabaseRecoveryRpcConfig(
            project_url=URL,
            key_source="SUPABASE_SECRET_KEY",
        ),
        api_key=SECRET,
        client=_client(handler),
    )
    with pytest.raises(SupabaseRecoveryRpcResponseError) as exc:
        transport(RPC, {"p_runtime_id": "runtime-91"})
    message = str(exc.value)
    assert "HTTP 403" in message
    assert "permission denied" in message
    assert SECRET not in message


def test_redirect_is_not_followed_and_is_treated_as_failure() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            307,
            headers={"location": "https://attacker.invalid/collect"},
        )

    transport = SupabaseRecoveryRpcTransport(
        config=SupabaseRecoveryRpcConfig(
            project_url=URL,
            key_source="SUPABASE_SECRET_KEY",
        ),
        api_key=SECRET,
        client=_client(handler),
    )
    with pytest.raises(SupabaseRecoveryRpcResponseError, match="HTTP 307"):
        transport(RPC, {"p_runtime_id": "runtime-91"})
    assert calls == 1


def test_timeout_is_not_blindly_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timeout", request=request)

    transport = SupabaseRecoveryRpcTransport(
        config=SupabaseRecoveryRpcConfig(
            project_url=URL,
            key_source="SUPABASE_SECRET_KEY",
        ),
        api_key=SECRET,
        client=_client(handler),
    )
    with pytest.raises(SupabaseRecoveryRpcTransportError, match="timeout"):
        transport(RPC, {"p_runtime_id": "runtime-91"})
    assert calls == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json=[{"ok": True}]),
        httpx.Response(204),
    ],
)
def test_success_status_requires_json_object_response(response) -> None:
    transport = SupabaseRecoveryRpcTransport(
        config=SupabaseRecoveryRpcConfig(
            project_url=URL,
            key_source="SUPABASE_SECRET_KEY",
        ),
        api_key=SECRET,
        client=_client(lambda request: response),
    )
    with pytest.raises(SupabaseRecoveryRpcResponseError):
        transport(RPC, {"p_runtime_id": "runtime-91"})


def test_timeout_environment_values_must_be_positive_numbers() -> None:
    with pytest.raises(SupabaseRecoveryRpcConfigurationError, match="numeric"):
        SupabaseRecoveryRpcTransport.from_env(
            env={
                "SUPABASE_URL": URL,
                "SUPABASE_SECRET_KEY": SECRET,
                "BRIAN_SUPABASE_READ_TIMEOUT": "abc",
            },
            client=_client(lambda request: httpx.Response(200, json={"ok": True})),
        )

    with pytest.raises(SupabaseRecoveryRpcConfigurationError, match="positive"):
        SupabaseRecoveryRpcTransport.from_env(
            env={
                "SUPABASE_URL": URL,
                "SUPABASE_SECRET_KEY": SECRET,
                "BRIAN_SUPABASE_READ_TIMEOUT": "0",
            },
            client=_client(lambda request: httpx.Response(200, json={"ok": True})),
        )
