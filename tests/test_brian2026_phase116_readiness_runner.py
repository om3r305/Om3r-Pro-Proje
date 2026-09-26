from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-brian-phase116-readiness.ps1"


def test_phase116_powershell_runner_uses_strict_scoped_topology_and_policy() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for name in (
        "BRIAN_SENSOR_SUPABASE_SECRET_KEY",
        "BRIAN_EDGE_SUPABASE_SECRET_KEY",
        "BRIAN_COST_SUPABASE_SECRET_KEY",
        "BRIAN_RUNTIME_SUPABASE_SECRET_KEY",
    ):
        assert name in source

    assert "brian2026.phase116_crypto_shadow_readiness_entrypoint" in source
    assert "config/brian-shadow-machine-policy-v1.json" in source
    assert "brian-shadow-main" in source


def test_phase116_powershell_runner_never_embeds_or_prints_secret_values() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()

    assert "sb_secret_" not in source
    assert "write-host $env:" not in source
    assert "write-output $env:" not in source
