from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = ROOT / "config" / "brian-shadow-topology-v1.env.example"


def _lines() -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in TOPOLOGY.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


def test_topology_routes_each_authority_to_the_intended_project() -> None:
    values = _lines()
    realtime = "https://dliediwlldojkfjzlznm.supabase.co"
    market = "https://qbcjuxhvhwagvqbjyemo.supabase.co"

    assert values["BRIAN_SENSOR_SUPABASE_URL"] == realtime
    assert values["BRIAN_EDGE_SUPABASE_URL"] == market
    assert values["BRIAN_COST_SUPABASE_URL"] == realtime
    assert values["BRIAN_RUNTIME_SUPABASE_URL"] == realtime
    assert values["BRIAN_RUNTIME_ID"] == "brian-shadow-main"


def test_topology_example_contains_no_backend_secret_value() -> None:
    source = TOPOLOGY.read_text(encoding="utf-8")

    assert "sb_secret_" not in source
    assert "service_role" not in source.lower()
