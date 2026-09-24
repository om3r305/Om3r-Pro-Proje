from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-brian-phase117-shadow.ps1"


def test_phase117_runner_is_readiness_guarded_and_uses_strict_scoped_topology() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for name in (
        "BRIAN_SENSOR_SUPABASE_SECRET_KEY",
        "BRIAN_EDGE_SUPABASE_SECRET_KEY",
        "BRIAN_COST_SUPABASE_SECRET_KEY",
        "BRIAN_RUNTIME_SUPABASE_SECRET_KEY",
    ):
        assert name in source

    assert "brian2026.phase117_readiness_guarded_crypto_shadow" in source
    assert "config/brian-shadow-machine-policy-v1.json" in source
    assert "brian-shadow-main" in source


def test_phase117_runner_is_not_itself_a_scheduler_or_live_boundary() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()

    assert "register-scheduledtask" not in source
    assert "schtasks" not in source
    assert "live_execution" not in source
    assert "sb_secret_" not in source
