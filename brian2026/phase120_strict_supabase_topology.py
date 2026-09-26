from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .phase91_supabase_rpc_transport import (
    SupabaseRecoveryRpcConfigurationError,
    _is_secure_project_url,
    _read_scoped_secret_key_from_env,
    _validate_server_key,
)

PHASE120_SCHEMA_VERSION = "brian.phase120-strict-supabase-topology.v1"

_SCOPES = (
    "BRIAN_SENSOR",
    "BRIAN_EDGE",
    "BRIAN_COST",
    "BRIAN_RUNTIME",
)


@dataclass(frozen=True, slots=True)
class StrictSupabaseBinding:
    scope: str
    project_url: str
    key_source: str
    project_host: str

    def __post_init__(self) -> None:
        if self.scope not in _SCOPES:
            raise ValueError("unsupported strict Supabase scope")
        if not _is_secure_project_url(self.project_url):
            raise ValueError("strict Supabase project_url must be secure")
        parsed = urlparse(self.project_url)
        if not parsed.hostname:
            raise ValueError("strict Supabase project URL requires hostname")
        if self.project_host != parsed.hostname:
            raise ValueError("project_host disagrees with project_url")
        if not self.key_source.startswith(f"{self.scope}_SUPABASE_"):
            raise ValueError("strict binding key must be scope-local")


@dataclass(frozen=True, slots=True)
class StrictSupabaseTopology:
    sensor: StrictSupabaseBinding
    edge: StrictSupabaseBinding
    cost: StrictSupabaseBinding
    runtime: StrictSupabaseBinding
    topology_id: str = field(init=False)
    schema_version: str = PHASE120_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        bindings = (self.sensor, self.edge, self.cost, self.runtime)
        if tuple(row.scope for row in bindings) != _SCOPES:
            raise ValueError("strict topology scopes are incomplete or misordered")
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase120 topology must remain shadow-only")
        object.__setattr__(
            self,
            "topology_id",
            hashlib.sha256(
                json.dumps(
                    self.identity_payload(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "bindings": [
                {
                    "scope": row.scope,
                    "project_url": row.project_url,
                    "key_source": row.key_source,
                    "project_host": row.project_host,
                }
                for row in (
                    self.sensor,
                    self.edge,
                    self.cost,
                    self.runtime,
                )
            ],
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }

    def public_summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "topology_id": self.topology_id,
            "bindings": {
                row.scope: {
                    "project_host": row.project_host,
                    "key_source": row.key_source,
                }
                for row in (
                    self.sensor,
                    self.edge,
                    self.cost,
                    self.runtime,
                )
            },
            "co_location": {
                "sensor_cost": (
                    self.sensor.project_url == self.cost.project_url
                ),
                "sensor_runtime": (
                    self.sensor.project_url == self.runtime.project_url
                ),
                "edge_runtime": (
                    self.edge.project_url == self.runtime.project_url
                ),
            },
            "shadow_only": True,
            "live_execution": False,
        }


def _has_scoped_key(env: Mapping[str, str], scope: str) -> bool:
    return any(
        str(env.get(name, "")).strip()
        for name in (
            f"{scope}_SUPABASE_SECRET_KEY",
            f"{scope}_SUPABASE_SECRET_KEYS",
            f"{scope}_SUPABASE_SERVICE_ROLE_KEY",
        )
    )


def _strict_binding(
    env: Mapping[str, str],
    scope: str,
) -> StrictSupabaseBinding:
    url_name = f"{scope}_SUPABASE_URL"
    project_url = str(env.get(url_name, "")).strip().rstrip("/")
    if not project_url:
        raise SupabaseRecoveryRpcConfigurationError(
            f"{url_name} is required for strict Phase120 topology; "
            "generic SUPABASE_URL fallback is not accepted"
        )
    if not _is_secure_project_url(project_url):
        raise SupabaseRecoveryRpcConfigurationError(
            f"{url_name} must use https outside localhost"
        )
    if not _has_scoped_key(env, scope):
        raise SupabaseRecoveryRpcConfigurationError(
            f"scoped server key is required for {scope}; "
            "generic Supabase key fallback is not accepted"
        )

    api_key, key_source = _read_scoped_secret_key_from_env(env, scope)
    if not key_source.startswith(f"{scope}_SUPABASE_"):
        raise SupabaseRecoveryRpcConfigurationError(
            f"{scope} resolved a non-scoped key source"
        )
    _validate_server_key(api_key, key_source)
    parsed = urlparse(project_url)
    if not parsed.hostname:
        raise SupabaseRecoveryRpcConfigurationError(
            f"{url_name} has no hostname"
        )
    return StrictSupabaseBinding(
        scope=scope,
        project_url=project_url,
        key_source=key_source,
        project_host=parsed.hostname,
    )


def load_strict_supabase_topology(
    env: Mapping[str, str],
) -> StrictSupabaseTopology:
    """Seal the four Brian data-authority scopes before scheduler entrypoints."""
    bindings = {
        scope: _strict_binding(env, scope)
        for scope in _SCOPES
    }
    return StrictSupabaseTopology(
        sensor=bindings["BRIAN_SENSOR"],
        edge=bindings["BRIAN_EDGE"],
        cost=bindings["BRIAN_COST"],
        runtime=bindings["BRIAN_RUNTIME"],
    )
