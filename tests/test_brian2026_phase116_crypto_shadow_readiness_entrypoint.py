from __future__ import annotations

import io
import json

import pytest

from brian2026.phase115_crypto_shadow_readiness_gate import (
    CryptoShadowReadinessReport,
    ReadinessCheck,
)
from brian2026.phase116_crypto_shadow_readiness_entrypoint import (
    EXIT_CHECK_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_SAFE_FAIL_CLOSED_ONLY,
    main,
)


TS = 1_790_000_000.0


def _strict_env():
    return {
        "BRIAN_SENSOR_SUPABASE_URL": "https://realtime.supabase.co",
        "BRIAN_SENSOR_SUPABASE_SECRET_KEY":
            "sb_secret_sensor_phase116_abcdefghijklmnopqrstuvwxyz",
        "BRIAN_EDGE_SUPABASE_URL": "https://market.supabase.co",
        "BRIAN_EDGE_SUPABASE_SECRET_KEY":
            "sb_secret_edge_phase116_abcdefghijklmnopqrstuvwxyz",
        "BRIAN_COST_SUPABASE_URL": "https://realtime.supabase.co",
        "BRIAN_COST_SUPABASE_SECRET_KEY":
            "sb_secret_cost_phase116_abcdefghijklmnopqrstuvwxyz",
        "BRIAN_RUNTIME_SUPABASE_URL": "https://realtime.supabase.co",
        "BRIAN_RUNTIME_SUPABASE_SECRET_KEY":
            "sb_secret_runtime_phase116_abcdefghijklmnopqrstuvwxyz",
    }


def _policy_payload():
    return {
        "asset_ids": ["crypto:BTCUSDT"],
        "model_weights": {"market_snapshot_analyst": 1.0},
        "decision": {
            "gross_target": 0.5,
            "position_limits": {
                "max_position_pct": 0.3,
                "max_gross_exposure": 0.5,
            },
            "covariance": {
                "alpha": "lw",
                "min_observations": 30,
                "max_period_volatility": 0.025,
                "nan_policy": "reject",
            },
            "turnover": {
                "max_l1_turnover": 0.25,
                "risk_reduction_bypass": True,
            },
            "market_neutral": False,
        },
        "execution": {
            "max_slippage_bps": 20.0,
            "ttl_seconds": 60,
            "minimum_net_margin_bps": 2.0,
        },
    }


def _report(status: str) -> CryptoShadowReadinessReport:
    if status == "READY_FOR_EDGE_BOUND_SHADOW":
        checks = (
            ReadinessCheck(
                code="CORE",
                scope="CORE",
                status="PASS",
                detail="core ready",
            ),
            ReadinessCheck(
                code="EDGE",
                scope="NEW_RISK",
                status="PASS",
                detail="edge ready",
            ),
        )
        safe = True
        new_risk = True
    elif status == "SAFE_FAIL_CLOSED_ONLY":
        checks = (
            ReadinessCheck(
                code="CORE",
                scope="CORE",
                status="PASS",
                detail="core ready",
            ),
            ReadinessCheck(
                code="EDGE",
                scope="NEW_RISK",
                status="FAIL",
                detail="edge not mature",
            ),
        )
        safe = True
        new_risk = False
    else:
        checks = (
            ReadinessCheck(
                code="CORE",
                scope="CORE",
                status="FAIL",
                detail="runtime not ready",
            ),
            ReadinessCheck(
                code="EDGE",
                scope="NEW_RISK",
                status="FAIL",
                detail="edge unavailable",
            ),
        )
        safe = False
        new_risk = False

    return CryptoShadowReadinessReport(
        runtime_id="runtime-116",
        observed_at=TS,
        status=status,
        safe_to_invoke_shadow_worker=safe,
        new_risk_ready=new_risk,
        checks=checks,
    )


class _Gate:
    def __init__(self, report):
        self.report = report
        self.calls = []
        self.closed = False

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.report

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("READY_FOR_EDGE_BOUND_SHADOW", EXIT_READY),
        ("SAFE_FAIL_CLOSED_ONLY", EXIT_SAFE_FAIL_CLOSED_ONLY),
        ("NOT_READY", EXIT_NOT_READY),
    ],
)
def test_entrypoint_exit_code_matches_readiness_status(status, expected) -> None:
    gate = _Gate(_report(status))
    factory_calls = []

    def gate_factory(**kwargs):
        factory_calls.append(kwargs)
        return gate

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        ["--runtime-id", "runtime-116"],
        env=_strict_env(),
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=stdout,
        stderr=stderr,
        gate_factory=gate_factory,
        clock=lambda: TS,
    )

    assert code == expected
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["status"] == status
    assert payload["runtime_id"] == "runtime-116"
    assert payload["entrypoint_schema_version"].startswith("brian.phase116")
    assert len(payload["topology_id"]) == 64
    assert payload["read_only"] is True
    assert payload["shadow_only"] is True
    assert payload["live_execution"] is False
    assert gate.closed is True
    assert len(factory_calls) == 1
    assert len(gate.calls) == 1
    assert gate.calls[0]["runtime_id"] == "runtime-116"
    assert gate.calls[0]["asset_ids"] == ("crypto:BTCUSDT",)
    assert gate.calls[0]["decision_timestamp"] == pytest.approx(TS)


def test_runtime_id_can_come_from_environment() -> None:
    gate = _Gate(_report("READY_FOR_EDGE_BOUND_SHADOW"))
    code = main(
        [],
        env={**_strict_env(), "BRIAN_RUNTIME_ID": "runtime-env-116"},
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        gate_factory=lambda **kwargs: gate,
        clock=lambda: TS,
    )

    assert code == EXIT_READY
    assert gate.calls[0]["runtime_id"] == "runtime-env-116"


def test_missing_runtime_or_policy_fails_before_gate_construction() -> None:
    calls = 0

    def gate_factory(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("gate must not be built")

    stderr = io.StringIO()
    code = main(
        [],
        env={},
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=io.StringIO(),
        stderr=stderr,
        gate_factory=gate_factory,
    )
    assert code == EXIT_INPUT_ERROR
    assert calls == 0
    assert "runtime_id is required" in json.loads(stderr.getvalue())["error"]

    stderr = io.StringIO()
    code = main(
        ["--runtime-id", "runtime-116"],
        env={},
        stdin=io.StringIO(""),
        stdout=io.StringIO(),
        stderr=stderr,
        gate_factory=gate_factory,
    )
    assert code == EXIT_INPUT_ERROR
    assert calls == 0
    assert "policy JSON cannot be empty" in json.loads(stderr.getvalue())["error"]


def test_check_error_redacts_all_scoped_supabase_secret_forms() -> None:
    secret = "sb_secret_phase116_never_echo"

    def gate_factory(**kwargs):
        raise RuntimeError(f"readiness source failed with {secret}")

    stderr = io.StringIO()
    code = main(
        ["--runtime-id", "runtime-116"],
        env={
            **_strict_env(),
            "BRIAN_RUNTIME_SUPABASE_SECRET_KEY": secret,
        },
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=io.StringIO(),
        stderr=stderr,
        gate_factory=gate_factory,
    )

    assert code == EXIT_CHECK_ERROR
    text = stderr.getvalue()
    assert secret not in text
    payload = json.loads(text)
    assert payload["status"] == "CHECK_ERROR"
    assert payload["read_only"] is True
    assert "<redacted>" in payload["error"]

def test_strict_topology_failure_blocks_gate_before_any_readiness_io() -> None:
    calls = 0

    def gate_factory(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("gate must not be built")

    env = _strict_env()
    env.pop("BRIAN_EDGE_SUPABASE_URL")
    stderr = io.StringIO()
    code = main(
        ["--runtime-id", "runtime-116"],
        env=env,
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=io.StringIO(),
        stderr=stderr,
        gate_factory=gate_factory,
    )

    assert code == EXIT_INPUT_ERROR
    assert calls == 0
    payload = json.loads(stderr.getvalue())
    assert payload["status"] == "INPUT_ERROR"
    assert "BRIAN_EDGE_SUPABASE_URL" in payload["error"]
