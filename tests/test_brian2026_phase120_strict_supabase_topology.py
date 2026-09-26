from __future__ import annotations

import json

import pytest

from brian2026.phase91_supabase_rpc_transport import (
    SupabaseRecoveryRpcConfigurationError,
)
from brian2026.phase120_strict_supabase_topology import (
    load_strict_supabase_topology,
)


def _env():
    return {
        "SUPABASE_URL": "https://generic.supabase.co",
        "SUPABASE_SECRET_KEY":
            "sb_secret_generic_phase120_abcdefghijklmnopqrstuvwxyz",
        "BRIAN_SENSOR_SUPABASE_URL":
            "https://realtime.supabase.co",
        "BRIAN_SENSOR_SUPABASE_SECRET_KEY":
            "sb_secret_sensor_phase120_abcdefghijklmnopqrstuvwxyz",
        "BRIAN_EDGE_SUPABASE_URL":
            "https://market.supabase.co",
        "BRIAN_EDGE_SUPABASE_SECRET_KEY":
            "sb_secret_edge_phase120_abcdefghijklmnopqrstuvwxyz",
        "BRIAN_COST_SUPABASE_URL":
            "https://realtime.supabase.co",
        "BRIAN_COST_SUPABASE_SECRET_KEY":
            "sb_secret_cost_phase120_abcdefghijklmnopqrstuvwxyz",
        "BRIAN_RUNTIME_SUPABASE_URL":
            "https://realtime.supabase.co",
        "BRIAN_RUNTIME_SUPABASE_SECRET_KEY":
            "sb_secret_runtime_phase120_abcdefghijklmnopqrstuvwxyz",
    }


def test_strict_topology_uses_only_scoped_bindings_and_exposes_no_secret() -> None:
    env = _env()

    topology = load_strict_supabase_topology(env)

    assert topology.sensor.project_host == "realtime.supabase.co"
    assert topology.edge.project_host == "market.supabase.co"
    assert topology.cost.project_host == "realtime.supabase.co"
    assert topology.runtime.project_host == "realtime.supabase.co"
    assert topology.sensor.key_source == "BRIAN_SENSOR_SUPABASE_SECRET_KEY"
    assert topology.edge.key_source == "BRIAN_EDGE_SUPABASE_SECRET_KEY"
    assert topology.cost.key_source == "BRIAN_COST_SUPABASE_SECRET_KEY"
    assert topology.runtime.key_source == "BRIAN_RUNTIME_SUPABASE_SECRET_KEY"
    assert len(topology.topology_id) == 64

    summary = topology.public_summary()
    serialized = json.dumps(summary, sort_keys=True)
    assert "sb_secret_" not in serialized
    assert summary["co_location"] == {
        "sensor_cost": True,
        "sensor_runtime": True,
        "edge_runtime": False,
    }
    assert summary["shadow_only"] is True
    assert summary["live_execution"] is False


@pytest.mark.parametrize(
    "missing_name",
    [
        "BRIAN_SENSOR_SUPABASE_URL",
        "BRIAN_EDGE_SUPABASE_URL",
        "BRIAN_COST_SUPABASE_URL",
        "BRIAN_RUNTIME_SUPABASE_URL",
    ],
)
def test_generic_url_never_substitutes_for_missing_scoped_url(
    missing_name,
) -> None:
    env = _env()
    env.pop(missing_name)

    with pytest.raises(
        SupabaseRecoveryRpcConfigurationError,
        match="generic SUPABASE_URL fallback is not accepted",
    ):
        load_strict_supabase_topology(env)


@pytest.mark.parametrize(
    "scope",
    [
        "BRIAN_SENSOR",
        "BRIAN_EDGE",
        "BRIAN_COST",
        "BRIAN_RUNTIME",
    ],
)
def test_generic_key_never_substitutes_for_missing_scoped_key(scope) -> None:
    env = _env()
    env.pop(f"{scope}_SUPABASE_SECRET_KEY")

    with pytest.raises(
        SupabaseRecoveryRpcConfigurationError,
        match="generic Supabase key fallback is not accepted",
    ):
        load_strict_supabase_topology(env)


def test_scoped_secret_keys_json_is_allowed_without_exposing_value() -> None:
    env = _env()
    env.pop("BRIAN_EDGE_SUPABASE_SECRET_KEY")
    env["BRIAN_EDGE_SUPABASE_SECRET_KEYS"] = json.dumps({
        "default": "sb_secret_edge_json_phase120_abcdefghijklmnopqrstuvwxyz",
    })

    topology = load_strict_supabase_topology(env)

    assert topology.edge.key_source == (
        "BRIAN_EDGE_SUPABASE_SECRET_KEYS.default"
    )
    assert "sb_secret_" not in json.dumps(topology.public_summary())


def test_scoped_legacy_service_role_key_remains_migration_compatible() -> None:
    env = _env()
    env.pop("BRIAN_RUNTIME_SUPABASE_SECRET_KEY")
    env["BRIAN_RUNTIME_SUPABASE_SERVICE_ROLE_KEY"] = (
        "eyJhbGciOiJIUzI1NiJ9."
        "eyJyb2xlIjoic2VydmljZV9yb2xlIn0."
        "phase120legacy"
    )

    topology = load_strict_supabase_topology(env)

    assert topology.runtime.key_source == (
        "BRIAN_RUNTIME_SUPABASE_SERVICE_ROLE_KEY"
    )


def test_publishable_scoped_key_is_rejected() -> None:
    env = _env()
    env["BRIAN_COST_SUPABASE_SECRET_KEY"] = "sb_publishable_not_backend"

    with pytest.raises(
        SupabaseRecoveryRpcConfigurationError,
        match="publishable",
    ):
        load_strict_supabase_topology(env)


def test_secret_key_field_rejects_legacy_format() -> None:
    env = _env()
    env["BRIAN_RUNTIME_SUPABASE_SECRET_KEY"] = (
        "eyJhbGciOiJIUzI1NiJ9.legacy.jwt"
    )

    with pytest.raises(
        SupabaseRecoveryRpcConfigurationError,
        match="sb_secret_",
    ):
        load_strict_supabase_topology(env)


def test_insecure_nonlocal_url_is_rejected() -> None:
    env = _env()
    env["BRIAN_SENSOR_SUPABASE_URL"] = "http://example.com"

    with pytest.raises(
        SupabaseRecoveryRpcConfigurationError,
        match="must use https",
    ):
        load_strict_supabase_topology(env)


def test_topology_identity_changes_when_project_binding_changes() -> None:
    first = load_strict_supabase_topology(_env())
    env = _env()
    env["BRIAN_RUNTIME_SUPABASE_URL"] = "https://runtime-other.supabase.co"
    second = load_strict_supabase_topology(env)

    assert first.topology_id != second.topology_id


def test_topology_identity_does_not_hash_or_depend_on_secret_value() -> None:
    first = load_strict_supabase_topology(_env())
    env = _env()
    env["BRIAN_RUNTIME_SUPABASE_SECRET_KEY"] = (
        "sb_secret_runtime_rotated_phase120_abcdefghijklmnopqrstuvwxyz"
    )
    second = load_strict_supabase_topology(env)

    assert first.topology_id == second.topology_id
