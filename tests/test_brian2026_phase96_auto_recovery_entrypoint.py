from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from brian2026.phase86_recovery_admission_interlock import RecoveryAdmissionState
from brian2026.phase89_recovery_startup_gate import RecoveryStartupGateReceipt
from brian2026.phase95_auto_binance_recovery_worker import AutoRecoveryStartupReceipt
from brian2026.phase96_auto_recovery_entrypoint import (
    EXIT_BUDGET_EXHAUSTED,
    EXIT_INPUT_ERROR,
    EXIT_MANUAL_REVIEW,
    EXIT_READY,
    EXIT_RECOVERY_BLOCKED,
    EXIT_WORKER_ERROR,
    main,
)


RUNTIME = "runtime-96"


def _receipt(
    *,
    status="READY_FOR_NORMAL_WORK",
    ready=True,
    blocked=False,
    outcomes=(),
    processed=0,
    evidence_assets=(),
):
    admission = (
        RecoveryAdmissionState(
            runtime_id=RUNTIME,
            status="RECOVERY_BARRIER",
            blocked=True,
            original_cycle_id="o" * 64,
            cancel_risk_receipt_id="r" * 64,
            reason="REDUCING_NEW_RISK",
        )
        if blocked
        else RecoveryAdmissionState(
            runtime_id=RUNTIME,
            status="OPEN",
            blocked=False,
        )
    )
    steps = tuple(SimpleNamespace(outcome=value) for value in outcomes)
    gate = RecoveryStartupGateReceipt(
        runtime_id=RUNTIME,
        steps=steps,
        admission=admission,
        status=status,
        ready_for_normal_work=ready,
        processed_items=processed,
        max_items=1,
    )
    return AutoRecoveryStartupReceipt(
        gate=gate,
        original_cycle_id=("o" * 64) if processed else None,
        cancel_risk_receipt_id=("r" * 64) if processed else None,
        preflight_work_state="NEEDS_CLAIM" if processed else "IDLE",
        directive_status="DUPLICATE" if evidence_assets else None,
        evidence_assets=tuple(evidence_assets),
        evidence_observed_at=tuple(
            (asset, 101.0 + index)
            for index, asset in enumerate(evidence_assets)
        ),
        decision_at=100.0,
    )


def test_ready_entrypoint_emits_one_machine_json_line() -> None:
    seen = {}

    def runner(**kwargs):
        seen.update(kwargs)
        return _receipt()

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "--claim-seconds", "45",
            "--intent-ttl-seconds", "90",
            "--worker-token", "worker-a",
            "--source-ref", "phase96:test",
            "--depth-limit", "50",
            "--max-spread-bps", "25",
            "--market-timeout-seconds", "4",
            "--max-assets", "6",
        ],
        env={"BRIAN_RUNTIME_ID": RUNTIME},
        stdout=stdout,
        stderr=stderr,
        worker_runner=runner,
        clock=lambda: 100.0,
    )
    assert code == EXIT_READY
    assert stderr.getvalue() == ""
    lines = stdout.getvalue().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["schema_version"] == "brian.phase96-auto-recovery-entrypoint.v1"
    assert payload["status"] == "READY_FOR_NORMAL_WORK"
    assert payload["ready_for_normal_work"] is True
    assert payload["shadow_only"] is True
    assert payload["live_execution"] is False

    assert seen["recovery_worker_token"] == "worker-a"
    assert seen["recovery_claim_seconds"] == 45
    assert seen["recovery_ttl_seconds"] == 90
    assert seen["source_ref"] == "phase96:test"
    assert seen["env"] == {"BRIAN_RUNTIME_ID": RUNTIME}
    assert seen["clock"]() == 100.0

    provider = seen["provider_factory"]()
    try:
        assert provider.depth_limit == 50
        assert provider.max_spread_bps == pytest.approx(25.0)
        assert provider.timeout_seconds == pytest.approx(4.0)
        assert provider.max_assets == 6
    finally:
        provider.close()


def test_env_defaults_are_forwarded_to_provider_and_worker() -> None:
    seen = {}

    def runner(**kwargs):
        seen.update(kwargs)
        return _receipt()

    env = {
        "BRIAN_RUNTIME_ID": RUNTIME,
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
        worker_runner=runner,
        clock=lambda: 100.0,
    )
    assert code == EXIT_READY
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
            _receipt(
                status="RECOVERY_BLOCKED",
                ready=False,
                blocked=True,
                outcomes=("WAIT_RISK_RELEASE",),
                processed=1,
            ),
            EXIT_RECOVERY_BLOCKED,
        ),
        (
            _receipt(
                status="RECOVERY_BLOCKED",
                ready=False,
                blocked=True,
                outcomes=("MANUAL_REVIEW_REQUIRED",),
                processed=1,
            ),
            EXIT_MANUAL_REVIEW,
        ),
        (
            _receipt(
                status="RECOVERY_BUDGET_EXHAUSTED",
                ready=False,
                blocked=True,
                outcomes=("RECOVERY_COMPLETED",),
                processed=1,
            ),
            EXIT_BUDGET_EXHAUSTED,
        ),
    ],
)
def test_exit_codes_preserve_block_manual_and_single_item_budget(receipt, expected) -> None:
    stdout = io.StringIO()
    code = main(
        [],
        env={"BRIAN_RUNTIME_ID": RUNTIME},
        stdout=stdout,
        stderr=io.StringIO(),
        worker_runner=lambda **kwargs: receipt,
        clock=lambda: 100.0,
    )
    assert code == expected
    assert json.loads(stdout.getvalue())["ready_for_normal_work"] is False


def test_evidence_metadata_is_exposed_without_orderbook_payload() -> None:
    receipt = _receipt(
        status="RECOVERY_BLOCKED",
        ready=False,
        blocked=True,
        outcomes=("WAIT_RISK_RELEASE",),
        processed=1,
        evidence_assets=("BTCUSDT", "ETHUSDT"),
    )
    stdout = io.StringIO()
    code = main(
        [],
        env={"BRIAN_RUNTIME_ID": RUNTIME},
        stdout=stdout,
        stderr=io.StringIO(),
        worker_runner=lambda **kwargs: receipt,
        clock=lambda: 100.0,
    )
    assert code == EXIT_RECOVERY_BLOCKED
    payload = json.loads(stdout.getvalue())
    assert payload["evidence_assets"] == ["BTCUSDT", "ETHUSDT"]
    assert payload["evidence_observed_at"] == {
        "BTCUSDT": 101.0,
        "ETHUSDT": 102.0,
    }
    assert "markets" not in payload
    assert "snapshots" not in payload


@pytest.mark.parametrize(
    "argv",
    [
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
def test_invalid_cli_configuration_is_json_input_error_without_worker(argv) -> None:
    calls = 0

    def runner(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("worker must not execute")

    stderr = io.StringIO()
    code = main(
        argv,
        env={"BRIAN_RUNTIME_ID": RUNTIME},
        stdout=io.StringIO(),
        stderr=stderr,
        worker_runner=runner,
    )
    assert code == EXIT_INPUT_ERROR
    assert calls == 0
    payload = json.loads(stderr.getvalue())
    assert payload["status"] == "INPUT_ERROR"
    assert payload["ready_for_normal_work"] is False


def test_runtime_failure_is_worker_error_and_redacts_supabase_secrets() -> None:
    modern = "sb_secret_superprivate"
    legacy = "eyJlegacy-superprivate"
    hosted = "sb_secret_hostedprivate"
    stderr = io.StringIO()

    def runner(**kwargs):
        raise RuntimeError(
            f"transport failed {modern} / {legacy} / {hosted}"
        )

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
        worker_runner=runner,
    )
    assert code == EXIT_WORKER_ERROR
    text = stderr.getvalue()
    assert modern not in text
    assert legacy not in text
    assert hosted not in text
    payload = json.loads(text)
    assert payload["status"] == "WORKER_ERROR"
    assert "<redacted>" in payload["error"]


def test_missing_worker_token_generates_process_unique_phase96_token() -> None:
    seen = {}

    def runner(**kwargs):
        seen.update(kwargs)
        return _receipt()

    code = main(
        [],
        env={"BRIAN_RUNTIME_ID": RUNTIME},
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        worker_runner=runner,
        clock=lambda: 100.0,
    )
    assert code == EXIT_READY
    assert seen["recovery_worker_token"].startswith("phase96-")


def test_cli_overrides_env_for_market_bounds() -> None:
    seen = {}

    def runner(**kwargs):
        seen.update(kwargs)
        return _receipt()

    code = main(
        [
            "--depth-limit", "10",
            "--max-spread-bps", "12",
            "--market-timeout-seconds", "2",
            "--max-assets", "2",
        ],
        env={
            "BRIAN_RUNTIME_ID": RUNTIME,
            "BRIAN_RECOVERY_BINANCE_DEPTH_LIMIT": "500",
            "BRIAN_RECOVERY_BINANCE_MAX_SPREAD_BPS": "50",
            "BRIAN_RECOVERY_BINANCE_TIMEOUT_SECONDS": "8",
            "BRIAN_RECOVERY_BINANCE_MAX_ASSETS": "12",
        },
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        worker_runner=runner,
        clock=lambda: 100.0,
    )
    assert code == EXIT_READY
    provider = seen["provider_factory"]()
    try:
        assert provider.depth_limit == 10
        assert provider.max_spread_bps == pytest.approx(12.0)
        assert provider.timeout_seconds == pytest.approx(2.0)
        assert provider.max_assets == 2
    finally:
        provider.close()
