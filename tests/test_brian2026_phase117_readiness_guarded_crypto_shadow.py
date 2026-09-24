from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from brian2026.phase115_crypto_shadow_readiness_gate import (
    CryptoShadowReadinessReport,
    ReadinessCheck,
)
from brian2026.phase117_readiness_guarded_crypto_shadow import (
    EXIT_EXECUTED,
    EXIT_INPUT_ERROR,
    EXIT_NOT_READY,
    EXIT_SAFE_FAIL_CLOSED_ONLY,
    EXIT_WORKER_ERROR,
    main,
)


TS = 1_790_000_000.0


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


def _readiness(status: str) -> CryptoShadowReadinessReport:
    if status == "READY_FOR_EDGE_BOUND_SHADOW":
        checks = (
            ReadinessCheck("CORE", "CORE", "PASS", "core ready"),
            ReadinessCheck("EDGE", "NEW_RISK", "PASS", "edge ready"),
        )
        safe = True
        new_risk = True
    elif status == "SAFE_FAIL_CLOSED_ONLY":
        checks = (
            ReadinessCheck("CORE", "CORE", "PASS", "core ready"),
            ReadinessCheck("EDGE", "NEW_RISK", "FAIL", "edge blocked"),
        )
        safe = True
        new_risk = False
    else:
        checks = (
            ReadinessCheck("CORE", "CORE", "FAIL", "core blocked"),
            ReadinessCheck("EDGE", "NEW_RISK", "FAIL", "edge blocked"),
        )
        safe = False
        new_risk = False
    return CryptoShadowReadinessReport(
        runtime_id="runtime-117",
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


class _Service:
    def __init__(self, *, runtime_id="runtime-117", ready=True):
        self.runtime_id = runtime_id
        self.ready = ready
        self.calls = []
        self.closed = False

    def run_once(self, **kwargs):
        self.calls.append(kwargs)
        startup = SimpleNamespace(
            status=(
                "READY_FOR_NORMAL_SHADOW"
                if self.ready
                else "RECOVERY_BLOCKED"
            ),
            ready_for_normal_shadow=self.ready,
        )
        cycle = (
            SimpleNamespace(
                decision=SimpleNamespace(
                    pipeline_id="p" * 64,
                    status="REBALANCE_PLANNED",
                ),
                execution=SimpleNamespace(
                    status="SHADOW_EXECUTED",
                    executed=True,
                ),
            )
            if self.ready
            else None
        )
        return SimpleNamespace(
            runtime_id=self.runtime_id,
            status=(
                "SHADOW_EXECUTED"
                if self.ready
                else "RECOVERY_BLOCKED"
            ),
            startup=startup,
            prefetched=self.ready,
            bundle_ref="b" * 64 if self.ready else None,
            cycle=cycle,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True


@pytest.mark.parametrize(
    ("readiness_status", "expected_exit"),
    [
        ("SAFE_FAIL_CLOSED_ONLY", EXIT_SAFE_FAIL_CLOSED_ONLY),
        ("NOT_READY", EXIT_NOT_READY),
    ],
)
def test_worker_is_never_constructed_when_readiness_is_not_full(
    readiness_status,
    expected_exit,
) -> None:
    gate = _Gate(_readiness(readiness_status))
    service_calls = 0

    def service_factory(**kwargs):
        nonlocal service_calls
        service_calls += 1
        raise AssertionError("worker service must not be constructed")

    stdout = io.StringIO()
    code = main(
        ["--runtime-id", "runtime-117"],
        env={},
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=stdout,
        stderr=io.StringIO(),
        readiness_factory=lambda **kwargs: gate,
        service_factory=service_factory,
        clock=lambda: TS,
    )

    assert code == expected_exit
    assert service_calls == 0
    payload = json.loads(stdout.getvalue())
    assert payload["status"] == "READINESS_BLOCKED"
    assert payload["worker_invoked"] is False
    assert payload["readiness"]["status"] == readiness_status
    assert gate.closed is True


def test_fully_ready_report_constructs_worker_only_after_gate() -> None:
    events = []
    gate = _Gate(_readiness("READY_FOR_EDGE_BOUND_SHADOW"))
    service = _Service()

    def readiness_factory(**kwargs):
        events.append("readiness_factory")
        return gate

    original_gate_run = gate.run

    def gate_run(**kwargs):
        events.append("readiness_run")
        return original_gate_run(**kwargs)

    gate.run = gate_run

    def service_factory(**kwargs):
        events.append("service_factory")
        assert kwargs["env"]["BRIAN_RUNTIME_ID"] == "runtime-117"
        return service

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        ["--runtime-id", "runtime-117"],
        env={},
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=stdout,
        stderr=stderr,
        readiness_factory=readiness_factory,
        service_factory=service_factory,
        clock=lambda: TS,
    )

    assert code == EXIT_EXECUTED
    assert stderr.getvalue() == ""
    assert events == [
        "readiness_factory",
        "readiness_run",
        "service_factory",
    ]
    assert len(service.calls) == 1
    payload = json.loads(stdout.getvalue())
    assert payload["worker_invoked"] is True
    assert payload["status"] == "SHADOW_EXECUTED"
    assert payload["worker"]["runtime_id"] == "runtime-117"
    assert payload["worker"]["executed"] is True
    assert payload["readiness_report_id"] == gate.report.report_id
    assert service.closed is True


def test_runtime_id_conflict_fails_before_readiness_or_worker() -> None:
    readiness_calls = 0
    service_calls = 0

    def readiness_factory(**kwargs):
        nonlocal readiness_calls
        readiness_calls += 1
        raise AssertionError("readiness must not be constructed")

    def service_factory(**kwargs):
        nonlocal service_calls
        service_calls += 1
        raise AssertionError("service must not be constructed")

    stderr = io.StringIO()
    code = main(
        ["--runtime-id", "runtime-arg"],
        env={"BRIAN_RUNTIME_ID": "runtime-env"},
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=io.StringIO(),
        stderr=stderr,
        readiness_factory=readiness_factory,
        service_factory=service_factory,
    )

    assert code == EXIT_INPUT_ERROR
    assert readiness_calls == 0
    assert service_calls == 0
    assert "conflicts" in json.loads(stderr.getvalue())["error"]


def test_worker_runtime_identity_drift_is_machine_failure() -> None:
    gate = _Gate(_readiness("READY_FOR_EDGE_BOUND_SHADOW"))
    service = _Service(runtime_id="different-runtime")
    stderr = io.StringIO()

    code = main(
        ["--runtime-id", "runtime-117"],
        env={},
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=io.StringIO(),
        stderr=stderr,
        readiness_factory=lambda **kwargs: gate,
        service_factory=lambda **kwargs: service,
        clock=lambda: TS,
    )

    assert code == EXIT_WORKER_ERROR
    payload = json.loads(stderr.getvalue())
    assert payload["status"] == "WORKER_ERROR"
    assert "runtime identity changed" in payload["error"]


def test_worker_recovery_can_reblock_after_ready_preflight() -> None:
    gate = _Gate(_readiness("READY_FOR_EDGE_BOUND_SHADOW"))
    service = _Service(ready=False)
    stdout = io.StringIO()

    code = main(
        ["--runtime-id", "runtime-117"],
        env={},
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=stdout,
        stderr=io.StringIO(),
        readiness_factory=lambda **kwargs: gate,
        service_factory=lambda **kwargs: service,
        clock=lambda: TS,
    )

    assert code == EXIT_NOT_READY
    payload = json.loads(stdout.getvalue())
    assert payload["worker_invoked"] is True
    assert payload["status"] == "RECOVERY_BLOCKED"
    assert payload["worker"]["ready_for_normal_shadow"] is False


def test_readiness_error_redacts_scoped_secret_and_never_constructs_worker() -> None:
    secret = "sb_secret_phase117_never_echo"
    service_calls = 0

    class BrokenGate:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def run(self, **kwargs):
            raise RuntimeError(f"probe failed with {secret}")

    def service_factory(**kwargs):
        nonlocal service_calls
        service_calls += 1
        raise AssertionError("service must not be constructed")

    stderr = io.StringIO()
    code = main(
        ["--runtime-id", "runtime-117"],
        env={"BRIAN_RUNTIME_SUPABASE_SECRET_KEY": secret},
        stdin=io.StringIO(json.dumps(_policy_payload())),
        stdout=io.StringIO(),
        stderr=stderr,
        readiness_factory=lambda **kwargs: BrokenGate(),
        service_factory=service_factory,
        clock=lambda: TS,
    )

    assert code == EXIT_WORKER_ERROR
    assert service_calls == 0
    text = stderr.getvalue()
    assert secret not in text
    payload = json.loads(text)
    assert payload["status"] == "READINESS_ERROR"
    assert "<redacted>" in payload["error"]
