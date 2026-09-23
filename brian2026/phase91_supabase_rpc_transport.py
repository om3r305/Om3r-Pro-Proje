from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

PHASE91_SCHEMA_VERSION = "brian.phase91-supabase-rpc-transport.v1"

RECOVERY_RPC_ALLOWLIST = frozenset({
    "brian_prepare_shadow_cancel_recovery",
    "brian_claim_shadow_cancel_recovery",
    "brian_renew_shadow_cancel_recovery_claim",
    "brian_mark_shadow_recovery_started",
    "brian_commit_shadow_recovery_checkpoint",
    "brian_certify_shadow_recovery_completion",
    "brian_read_shadow_recovery_admission",
    "brian_read_next_shadow_recovery_work",
})


class SupabaseRecoveryRpcError(RuntimeError):
    pass


class SupabaseRecoveryRpcConfigurationError(SupabaseRecoveryRpcError):
    pass


class SupabaseRecoveryRpcTransportError(SupabaseRecoveryRpcError):
    pass


class SupabaseRecoveryRpcResponseError(SupabaseRecoveryRpcError):
    pass


def _sanitize_error_payload(response: httpx.Response) -> str:
    try:
        raw = response.json()
    except (ValueError, json.JSONDecodeError):
        text = response.text.strip()
        return text[:300] if text else "non-json response"

    if not isinstance(raw, Mapping):
        return "unexpected error payload"
    parts: list[str] = []
    for key in ("code", "message", "hint"):
        value = raw.get(key)
        if value is not None:
            parts.append(f"{key}={str(value)[:180]}")
    return "; ".join(parts) if parts else "database api error"


def _is_secure_project_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.netloc:
        return True
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        return True
    return False


def _read_secret_key_from_env(env: Mapping[str, str]) -> tuple[str, str]:
    modern = env.get("SUPABASE_SECRET_KEY", "").strip()
    if modern:
        return modern, "SUPABASE_SECRET_KEY"

    secret_keys_raw = env.get("SUPABASE_SECRET_KEYS", "").strip()
    if secret_keys_raw:
        try:
            parsed = json.loads(secret_keys_raw)
        except json.JSONDecodeError as exc:
            raise SupabaseRecoveryRpcConfigurationError(
                "SUPABASE_SECRET_KEYS must be valid JSON"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise SupabaseRecoveryRpcConfigurationError(
                "SUPABASE_SECRET_KEYS must be a JSON object"
            )
        default = parsed.get("default")
        if isinstance(default, str) and default.strip():
            return default.strip(), "SUPABASE_SECRET_KEYS.default"
        raise SupabaseRecoveryRpcConfigurationError(
            "SUPABASE_SECRET_KEYS.default is required"
        )

    legacy = env.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if legacy:
        return legacy, "SUPABASE_SERVICE_ROLE_KEY"

    raise SupabaseRecoveryRpcConfigurationError(
        "server-side Supabase secret key is required"
    )


def _validate_server_key(key: str, source: str) -> None:
    if not key.strip():
        raise SupabaseRecoveryRpcConfigurationError("Supabase key is empty")
    if key.startswith("sb_publishable_"):
        raise SupabaseRecoveryRpcConfigurationError(
            f"{source} contains a publishable key; backend recovery requires a secret key"
        )
    if source == "SUPABASE_SECRET_KEY" and not key.startswith("sb_secret_"):
        raise SupabaseRecoveryRpcConfigurationError(
            "SUPABASE_SECRET_KEY must use the sb_secret_ format"
        )


@dataclass(frozen=True, slots=True)
class SupabaseRecoveryRpcConfig:
    project_url: str
    key_source: str
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 15.0
    write_timeout_seconds: float = 15.0
    pool_timeout_seconds: float = 5.0
    schema_version: str = PHASE91_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not _is_secure_project_url(self.project_url):
            raise ValueError(
                "Supabase project_url must use https (http allowed only for localhost)"
            )
        if not self.key_source:
            raise ValueError("key_source is required")
        for label, value in (
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("read_timeout_seconds", self.read_timeout_seconds),
            ("write_timeout_seconds", self.write_timeout_seconds),
            ("pool_timeout_seconds", self.pool_timeout_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase91 transport must remain shadow-only")


class SupabaseRecoveryRpcTransport:
    """Fail-closed PostgREST RPC transport for the Phase81-90 recovery stack.

    The modern Supabase secret key is sent only through the `apikey` header.
    Legacy service-role keys remain accepted during migration, but secrets are
    never placed in URLs, exception messages or response diagnostics.

    No automatic HTTP retry is performed. Recovery RPCs cross durable database
    boundaries; retries are intentionally delegated to Phase81-89 idempotency
    and restart semantics rather than replayed blindly at the HTTP layer.
    """

    def __init__(
        self,
        *,
        config: SupabaseRecoveryRpcConfig,
        api_key: str,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise SupabaseRecoveryRpcConfigurationError("Supabase API key is required")
        _validate_server_key(api_key, config.key_source)
        self.config = config
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(
                connect=config.connect_timeout_seconds,
                read=config.read_timeout_seconds,
                write=config.write_timeout_seconds,
                pool=config.pool_timeout_seconds,
            ),
            follow_redirects=False,
            trust_env=False,
        )

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> "SupabaseRecoveryRpcTransport":
        source = os.environ if env is None else env
        project_url = source.get("SUPABASE_URL", "").strip().rstrip("/")
        if not project_url:
            raise SupabaseRecoveryRpcConfigurationError("SUPABASE_URL is required")
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
            if value <= 0:
                raise SupabaseRecoveryRpcConfigurationError(
                    f"{name} must be positive"
                )
            return value

        config = SupabaseRecoveryRpcConfig(
            project_url=project_url,
            key_source=key_source,
            connect_timeout_seconds=_float("BRIAN_SUPABASE_CONNECT_TIMEOUT", 5.0),
            read_timeout_seconds=_float("BRIAN_SUPABASE_READ_TIMEOUT", 15.0),
            write_timeout_seconds=_float("BRIAN_SUPABASE_WRITE_TIMEOUT", 15.0),
            pool_timeout_seconds=_float("BRIAN_SUPABASE_POOL_TIMEOUT", 5.0),
        )
        return cls(config=config, api_key=key, client=client)

    def __call__(self, function_name: str, params: Mapping[str, object]) -> object:
        if function_name not in RECOVERY_RPC_ALLOWLIST:
            raise SupabaseRecoveryRpcConfigurationError(
                f"RPC function is not allowed by Phase91: {function_name}"
            )
        if not isinstance(params, Mapping):
            raise TypeError("RPC params must be a mapping")

        url = f"{self.config.project_url}/rest/v1/rpc/{function_name}"
        headers = {
            "apikey": self._api_key,
            "accept": "application/json",
            "content-type": "application/json",
            "user-agent": "brian-phase91-recovery-worker/1",
        }
        try:
            response = self._client.post(
                url,
                headers=headers,
                json=dict(params),
            )
        except httpx.TimeoutException as exc:
            raise SupabaseRecoveryRpcTransportError(
                f"Supabase RPC timeout for {function_name}"
            ) from exc
        except httpx.TransportError as exc:
            raise SupabaseRecoveryRpcTransportError(
                f"Supabase RPC transport failure for {function_name}"
            ) from exc

        if response.status_code < 200 or response.status_code >= 300:
            detail = _sanitize_error_payload(response)
            raise SupabaseRecoveryRpcResponseError(
                f"Supabase RPC {function_name} returned HTTP "
                f"{response.status_code}: {detail}"
            )

        try:
            payload: Any = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise SupabaseRecoveryRpcResponseError(
                f"Supabase RPC {function_name} returned invalid JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise SupabaseRecoveryRpcResponseError(
                f"Supabase RPC {function_name} must return a JSON object"
            )
        return {str(key): value for key, value in payload.items()}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SupabaseRecoveryRpcTransport":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
