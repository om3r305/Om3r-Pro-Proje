from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from brian2026.phase86_recovery_admission_interlock import RecoveryAdmissionState
from brian2026.phase89_recovery_startup_gate import RecoveryStartupGateReceipt
from brian2026.phase95_auto_binance_recovery_worker import AutoRecoveryStartupReceipt
from brian2026.phase97_bounded_auto_recovery_drain import (
    BoundedAutoRecoveryDrainReceipt,
)
from brian2026.phase98_bounded_auto_recovery_entrypoint import (
    EXIT_BUDGET_EXHAUSTED,
    EXIT_INPUT_ERROR,
    EXIT_MANUAL_REVIEW,
    EXIT_READY,
    EXIT_RECOVERY_BLOCKED,
    EXIT_WORKER_ERROR,
    main,
)


RUNTIME = "runtime-98"
CYCLE = "c" * 64
RISK = "r" * 64


def _attempt(
    *,
    status="READY_FOR_NORMAL_WORK",
    ready=True,
    blocked=False,
    outcome="IDLE",
    processed=0,
    evidence_assets=(),
):
    admission = (
        RecoveryAdmissionState(
            runtime_id=RUNTIME,
            status="RECOVERY_BARRIER",
            blocked=True,
            original_cycle_id=CYCLE,
            cancel_risk_receipt_id=RISK,
            reason="AFTER_START_CANCEL",
        )
        if blocked
        else RecoveryAdmissionState(
            runtime_id=RUNTIME,
            status="OPEN",
            blocked=False,
        )
    )
    gate = RecoveryStartupGateReceipt(
        runtime_id=RUNTIME,
        steps=(SimpleNamespace(outcome=outcome),),
        admission=admission,
        status=status,
        ready_for_normal_work=ready,
        processed_items=processed,
        max_items=1,
    )
    return AutoRecoveryStartupReceipt(
        gate=gate,
        original_cycle_id=CYCLE if processed else None,
        cancel_risk_receipt_id=RISK if processed else None,
        preflight_work_state="RECOVERY_PROGRESS" if processed else "IDLE",
        directive_status="PREPARED" if evidence_assets else None,
        evidence_assets=tuple(evidence_assets),
        evidence_observed_at=tuple(
            (asset, 101.0 + index)
            for index, asset in enumerate(evidence_assets)
        ),
        decision_at=100.0,
    )


def _drain(
    *,
    status="READY_FOR_NORMAL_WORK",
    ready=True,
    blocked=False,
    attempts=None,
    max_items=8,
):
    rows = tuple(attempts or (_attempt(),))
    admission = (
        RecoveryAdmissionState(
            runtime_id=RUNTIME,
            status="RECOVERY_BARRIER",
            blocked=True,
            original_cycle_id=CYCLE,
            cancel_risk_receipt_id=RISK,
            reason="AFTER_START_CANCEL",
        )
        if blocked
        else RecoveryAdmissionState(
            runtime_id=RUNTIME,
            status="OPEN",
            blocked=False,
        )
    )
    return BoundedAutoRecoveryDrainReceipt(
        runtime_id=RUNTIME,
        attempts=rows,
        final_admission=admission,
        status=status,
        ready_for_normal_work=ready,
        processed_items=sum(row.gate.processed_items for row in rows),
        max_items=max_items,
    )


class _Session:
    def __init__(self):
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited += 1


def test_ready_entrypoint_uses_one_session_and_emits_one_json_line() -> None:
    session = _Session()
    seen = {}

    def session_factory(**kwargs):
        seen["session_kwargs"] = kwargs
        return session

    def drain_runner(session_arg, **kwargs):
        seen["session"] = session_arg
        seen["drain_kwargs"] = kwargs
        return _drain(max_items=4)

    stdout = io.StringIO()
    stderr = io.StringIO()
    env = {"BRIAN_RUNTIME_ID": RUNTIME}
    code = main(
        [
            "--max-items", "4",
            "--claim-seconds", "45",
            "--intent-ttl-seconds", "90",
            "--worker-token", "worker-98",
            "--source-ref", "phase98:test",
            "--depth-limit", "50",
            "--max-spread-bps", "25",
            "--market-timeout-seconds", "4",
            "--max-assets", "6",
        ],
        env=env,
        stdout=stdout,
        stderr=stderr,
        session_factory=session_factory,
        drain_runner=drain_runner,
        clock=lambda: 100.0,
    )

    assert code == EXIT_READY
    assert stderr.getvalue() == ""
    assert session.entered == 1
    assert session.exited == 1
    assert seen["session"] is session
    assert seen["session_kwargs"] == {"env": env}

    kwargs = seen["drain_kwargs"]
    assert kwargs["max_items"] == 4
    assert kwargs["recovery_worker_token"] == "worker-98"
    assert kwargs["recovery_claim_seconds"] == 45
    assert kwargs["recovery_ttl_seconds"] == 90
    assert kwargs["source_ref"] == "phase98:test"

    provider = kwargs["provider_factory"]()
    try:
        assert provider.depth_limit == 50
        assert provider.max_spread_bps == pytest.approx(25.0)
        assert provider.timeout_seconds == pytest.approx(4.0)
        assert provider.max_assets == 6
    finally:
        provider.close()

    lines = stdout.getvalue().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["schema_version"] == (
        "brian.phase98-bounded-auto-recovery-entrypoint.v1"
    )
    assert payload["status"] == "READY_FOR_NORMAL_WORK"
    assert payload["ready_for_normal_work"] is True
    assert payload["attempt_count"] == 1
    assert payload["shadow_only"] is True
    assert payload["live_execution"] is False


def test_env_defaults_configure_drain_and_provider() -> None:
    session = _Session()
    seen = {}

    def drain_runner(session_arg, **kwargs):
        seen.update(kwargs)
        return _drain(max_items=3)

    env = {
        "BRIAN_RUNTIME_ID": RUNTIME,
        "BRIAN_RECOVERY_MAX_ITEMS": "3",
        "BRIAN_RECOVERY_CLAIM_SECONDS": "55",
        "BRIAN_RECOVERY_INTENT_TTL_SECONDS": "120",
        "BRIAN_RECOVERY_WORKER_TOKEN": "env-worker",
        "BRIAN_RECOVERY_BINANCE_DEPTH_LIMIT": "20",
        "BRIAN_RECOVERY_BINANCE_MAX_SPREAD_BPS": "18.5",
        "BRIAN_RECOVERY_BINANCE_TIMEOUT_SECONDS": "3.5",
        "BRIAN_RECOVERY_BINANCE_MAX_ASSETS": "4",
    }
    code = main(
        [],
        env=env,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        session_factory=lambda **kwargs: session,
        drain_runner=drain_runner,
        clock=lambda: 100.0,
    )

    assert code == EXIT_READY
    assert seen["max_items"] == 3
    assert seen["recovery_worker_token"] == "env-worker"
    assert seen["recovery_claim_seconds"] == 55
    assert seen["recovery_ttl_seconds"] == 120
    provider = seen["provider_factory"]()
    try:
        assert provider.depth_limit == 20
        assert provider.max_spread_bps == pytest.approx(18.5)
        assert provider.timeout_seconds == pytest.approx(3.5)
        assert provider.max_assets == 4
    finally:
        provider.close()


@pytest.mark.parametrize(
    ("receipt", "expected"),
    [
        (
            _drain(
                status="RECOVERY_BLOCKED",
                ready=False,
                blocked=True,
                attempts=(
                    _attempt(
                        status="RECOVERY_BLOCKED",
                        ready=False,
                        blocked=True,
                        outcome="WAIT_RISK_RELEASE",
                        processed=1,
                    ),
                ),
            ),
            EXIT_RECOVERY_BLOCKED,
        ),
        (
            _drain(
                status="RECOVERY_BLOCKED",
                ready=False,
                blocked=True,
                attempts=(
                    _attempt(
                        status="RECOVERY_BLOCKED",
                        ready=False,
                        blocked=True,
                        outcome="MANUAL_REVIEW_REQUIRED",
                        processed=1,
                    ),
                ),
            ),
            EXIT_MANUAL_REVIEW,
        ),
        (
            _drain(
                status="RECOVERY_DRAIN_BUDGET_EXHAUSTED",
                ready=False,
                blocked=True,
                attempts=(
                    _attempt(
                        status="RECOVERY_BUDGET_EXHAUSTED",
                        ready=False,
                        blocked=True,
                        outcome="RECOVERY_COMPLETED",
                        processed=1,
                    ),
                ),
                max_items=1,
            ),
            EXIT_BUDGET_EXHAUSTED,
        ),
    ],
)
def test_exit_codes_preserve_block_manual_and_budget(receipt, expected) -> None:
    stdout = io.StringIO()
    code = main(
        [],
        env={"BRIAN_RUNTIME_ID": RUNTIME},
        stdout=stdout,
        stderr=io.StringIO(),
        session_factory=lambda **kwargs: _Session(),
        drain_runner=lambda *args, **kwargs: receipt,
    )
    assert code == expected
    assert json.loads(stdout.getvalue())["ready_for_normal_work"] is False


def test_machine_summary_exposes_bounded_evidence_metadata_not_orderbooks() -> None:
    receipt = _drain(
        attempts=(
            _attempt(
                evidence_assets=("BTCUSDT", "ETHUSDT"),
            ),
        ),
    )
    stdout = io.StringIO()
    code = main(
        [],
        env={"BRIAN_RUNTIME_ID": RUNTIME},
        stdout=stdout,
        stderr=io.StringIO(),
        session_factory=lambda **kwargs: _Session(),
        drain_runner=lambda *args, **kwargs: receipt,
    )

    assert code == EXIT_READY
    payload = json.loads(stdout.getvalue())
    assert payload["attempts"][0]["evidence_assets"] == [
        "BTCUSDT",
        "ETHUSDT",
    ]
    assert payload["attempts"][0]["evidence_observed_at"] == {
        "BTCUSDT": 101.0,
        "ETHUSDT": 102.0,
    }
    text = stdout.getvalue()
    assert "orderbook" not in text.lower()
    assert "snapshots" not in text.lower()


@pytest.mark.parametrize(
    "argv",
    [
        ["--max-items", "0"],
        ["--max-items", "33"],
        ["--claim-seconds", "9"],
        ["--claim-seconds", "301"],
        ["--intent-ttl-seconds", "9"],
        ["--intent-ttl-seconds", "901"],
        ["--depth-limit", "123"],
        ["--max-spread-bps", "0"],
        ["--max-spread-bps", "501"],
        ["--market-timeout-seconds", "0.1"],
        ["--market-timeout-seconds", "21"],
        ["--max-assets", "0"],
        ["--max-assets", "33"],
        ["--source-ref", ""],
        ["--unknown-arg"],
    ],
)
def test_invalid_configuration_is_json_input_error(argv) -> None:
    calls = 0

    def session_factory(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("session must not open")

    stderr = io.StringIO()
    code = main(
        argv,
        env={"BRIAN_RUNTIME_ID": RUNTIME},
        stdout=io.StringIO(),
        stderr=stderr,
        session_factory=session_factory,
    )

    assert code == EXIT_INPUT_ERROR
    assert calls == 0
    payload = json.loads(stderr.getvalue())
    assert payload["status"] == "INPUT_ERROR"
    assert payload["ready_for_normal_work"] is False


def test_worker_error_closes_session_and_redacts_supabase_secrets() -> None:
    modern = "sb_secret_phase98modern"
    legacy = "legacy-phase98-private"
    hosted = "sb_secret_phase98hosted"
    session = _Session()
    stderr = io.StringIO()

    def drain_runner(*args, **kwargs):
        raise RuntimeError(f"boom {modern} / {legacy} / {hosted}")

    code = main(
        [],
        env={
            "BRIAN_RUNTIME_ID": RUNTIME,
            "SUPABASE_SECRET_KEY": modern,
            "SUPABASE_SERVICE_ROLE_KEY": legacy,
            "SUPABASE_SECRET_KEYS": json.dumps({"default": hosted}),
        },
        stdout=io.StringIO(),
        stderr=stderr,
        session_factory=lambda **kwargs: session,
        drain_runner=drain_runner,
    )

    assert code == EXIT_WORKER_ERROR
    assert session.entered == 1
    assert session.exited == 1
    text = stderr.getvalue()
    assert modern not in text
    assert legacy not in text
    assert hosted not in text
    payload = json.loads(text)
    assert payload["status"] == "WORKER_ERROR"
    assert "<redacted>" in payload["error"]


def test_missing_worker_token_generates_phase98_process_identity() -> None:
    session = _Session()
    seen = {}

    def drain_runner(session_arg, **kwargs):
        seen.update(kwargs)
        return _drain()

    code = main(
        [],
        env={"BRIAN_RUNTIME_ID": RUNTIME},
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        session_factory=lambda **kwargs: session,
        drain_runner=drain_runner,
    )
    assert code == EXIT_READY
    assert seen["recovery_worker_token"].startswith("phase98-")
    assert len(seen["recovery_worker_token"]) > len("phase98-")


def test_cli_values_override_environment_values() -> None:
    session = _Session()
    seen = {}

    def drain_runner(session_arg, **kwargs):
        seen.update(kwargs)
        return _drain(max_items=2)

    code = main(
        [
            "--max-items", "2",
            "--claim-seconds", "40",
            "--intent-ttl-seconds", "80",
            "--worker-token", "cli-worker",
        ],
        env={
            "BRIAN_RUNTIME_ID": RUNTIME,
            "BRIAN_RECOVERY_MAX_ITEMS": "7",
            "BRIAN_RECOVERY_CLAIM_SECONDS": "70",
            "BRIAN_RECOVERY_INTENT_TTL_SECONDS": "140",
            "BRIAN_RECOVERY_WORKER_TOKEN": "env-worker",
        },
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        session_factory=lambda **kwargs: session,
        drain_runner=drain_runner,
    )

    assert code == EXIT_READY
    assert seen["max_items"] == 2
    assert seen["recovery_claim_seconds"] == 40
    assert seen["recovery_ttl_seconds"] == 80
    assert seen["recovery_worker_token"] == "cli-worker"
