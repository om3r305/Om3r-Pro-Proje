from __future__ import annotations

import io
import json

import pytest

from brian2026.phase86_recovery_admission_interlock import RecoveryAdmissionState
from brian2026.phase89_recovery_startup_gate import RecoveryStartupGateReceipt
from brian2026.phase93_recovery_worker_entrypoint import (
    EXIT_BUDGET_EXHAUSTED,
    EXIT_INPUT_ERROR,
    EXIT_MANUAL_REVIEW,
    EXIT_READY,
    EXIT_RECOVERY_BLOCKED,
    EXIT_WORKER_ERROR,
    RecoveryWorkerEntrypointError,
    main,
    parse_worker_input,
)


def _payload():
    return {
        "markets": {
            "BTCUSDT": {
                "reference_price": 100.0,
                "tick_size": 0.1,
                "snapshots": [
                    {
                        "timestamp": 10.0,
                        "bids": [[99.9, 2.0], [99.8, 3.0]],
                        "asks": [[100.1, 2.5], [100.2, 4.0]],
                    }
                ],
            }
        },
        "risk_limits_by_asset": {
            "BTCUSDT": {
                "min_notional": 5.0,
                "max_notional": 5000.0,
                "max_notional_per_order": 1000.0,
            }
        },
        "marks": {"BTCUSDT": 100.0},
    }


def _receipt(
    *,
    status="READY_FOR_NORMAL_WORK",
    ready=True,
    blocked=False,
    outcomes=(),
    processed=0,
    max_items=4,
):
    admission = (
        RecoveryAdmissionState(
            runtime_id="runtime-93",
            status="RECOVERY_BARRIER",
            blocked=True,
            original_cycle_id="o" * 64,
            cancel_risk_receipt_id="r" * 64,
            reason="REDUCING_NEW_RISK",
        )
        if blocked
        else RecoveryAdmissionState(
            runtime_id="runtime-93",
            status="OPEN",
            blocked=False,
        )
    )
    steps = tuple(
        type(
            "Step",
            (),
            {
                "outcome": outcome,
            },
        )()
        for outcome in outcomes
    )
    return RecoveryStartupGateReceipt(
        runtime_id="runtime-93",
        steps=steps,
        admission=admission,
        status=status,
        ready_for_normal_work=ready,
        processed_items=processed,
        max_items=max_items,
    )


def test_parser_builds_exact_market_risk_and_mark_contracts() -> None:
    parsed = parse_worker_input(_payload())
    market = parsed.markets["BTCUSDT"]
    assert market.reference_price == 100.0
    assert market.tick_size == 0.1
    assert market.snapshots[0].bids[0].price == 99.9
    assert market.snapshots[0].asks[0].quantity == 2.5

    limits = parsed.risk_limits_by_asset["BTCUSDT"]
    assert limits.min_notional == 5.0
    assert limits.max_notional == 5000.0
    assert limits.max_notional_per_order == 1000.0
    assert parsed.marks == {"BTCUSDT": 100.0}


def test_parser_rejects_unknown_fields_and_crossed_books() -> None:
    bad = _payload()
    bad["unexpected"] = True
    with pytest.raises(RecoveryWorkerEntrypointError, match="unknown"):
        parse_worker_input(bad)

    bad = _payload()
    bad["markets"]["BTCUSDT"]["snapshots"][0]["bids"][0][0] = 100.2
    with pytest.raises(ValueError, match="crossed"):
        parse_worker_input(bad)


@pytest.mark.parametrize(
    ("mark", "match"),
    [
        (0, "positive"),
        (-1, "positive"),
        ("nan", "finite"),
    ],
)
def test_parser_rejects_invalid_marks(mark, match) -> None:
    bad = _payload()
    bad["marks"]["BTCUSDT"] = mark
    with pytest.raises((RecoveryWorkerEntrypointError, ValueError), match=match):
        parse_worker_input(bad)


def test_ready_worker_emits_single_json_line_and_exit_zero() -> None:
    seen = {}

    def runner(**kwargs):
        seen.update(kwargs)
        return _receipt()

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "--max-items",
            "4",
            "--claim-seconds",
            "30",
            "--intent-ttl-seconds",
            "60",
            "--worker-token",
            "worker-a",
            "--source-ref",
            "phase93:test",
        ],
        env={"BRIAN_RUNTIME_ID": "runtime-93"},
        stdin=io.StringIO(json.dumps(_payload())),
        stdout=stdout,
        stderr=stderr,
        worker_runner=runner,
        clock=lambda: 123.5,
    )
    assert code == EXIT_READY
    assert stderr.getvalue() == ""
    rows = stdout.getvalue().splitlines()
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert payload["status"] == "READY_FOR_NORMAL_WORK"
    assert payload["ready_for_normal_work"] is True
    assert payload["live_execution"] is False
    assert seen["recovery_worker_token"] == "worker-a"
    assert seen["max_items"] == 4
    assert seen["recovery_claim_seconds"] == 30
    assert seen["recovery_ttl_seconds"] == 60
    assert seen["observed_at"] == 123.5
    assert seen["source_ref"] == "phase93:test"
    assert seen["env"] == {"BRIAN_RUNTIME_ID": "runtime-93"}
    assert "BTCUSDT" in seen["recovery_markets"]


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
                max_items=1,
            ),
            EXIT_BUDGET_EXHAUSTED,
        ),
    ],
)
def test_machine_exit_codes_distinguish_block_manual_and_budget(receipt, expected) -> None:
    stdout = io.StringIO()
    code = main(
        [],
        env={"BRIAN_RUNTIME_ID": "runtime-93"},
        stdin=io.StringIO("{}"),
        stdout=stdout,
        stderr=io.StringIO(),
        worker_runner=lambda **kwargs: receipt,
        clock=lambda: 1.0,
    )
    assert code == expected
    assert json.loads(stdout.getvalue())["ready_for_normal_work"] is False


def test_empty_tty_input_is_safe_empty_payload_for_idle_probe() -> None:
    class Tty(io.StringIO):
        def isatty(self):
            return True

    seen = {}

    def runner(**kwargs):
        seen.update(kwargs)
        return _receipt()

    code = main(
        [],
        env={"BRIAN_RUNTIME_ID": "runtime-93"},
        stdin=Tty("ignored"),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        worker_runner=runner,
        clock=lambda: 1.0,
    )
    assert code == EXIT_READY
    assert seen["recovery_markets"] == {}
    assert seen["recovery_risk_limits_by_asset"] == {}
    assert seen["marks"] == {}


def test_invalid_json_and_unsafe_bounds_are_input_errors_without_runner_call() -> None:
    calls = 0

    def runner(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("runner must not execute")

    stderr = io.StringIO()
    code = main(
        [],
        env={"BRIAN_RUNTIME_ID": "runtime-93"},
        stdin=io.StringIO("{bad"),
        stdout=io.StringIO(),
        stderr=stderr,
        worker_runner=runner,
    )
    assert code == EXIT_INPUT_ERROR
    assert calls == 0
    assert json.loads(stderr.getvalue())["status"] == "INPUT_ERROR"

    code = main(
        ["--max-items", "33"],
        env={"BRIAN_RUNTIME_ID": "runtime-93"},
        stdin=io.StringIO("{}"),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        worker_runner=runner,
    )
    assert code == EXIT_INPUT_ERROR
    assert calls == 0


def test_runtime_value_error_is_worker_error_not_misclassified_as_input() -> None:
    stderr = io.StringIO()
    code = main(
        [],
        env={"BRIAN_RUNTIME_ID": "runtime-93"},
        stdin=io.StringIO("{}"),
        stdout=io.StringIO(),
        stderr=stderr,
        worker_runner=lambda **kwargs: (_ for _ in ()).throw(
            ValueError("durable runtime conflict")
        ),
    )
    assert code == EXIT_WORKER_ERROR
    row = json.loads(stderr.getvalue())
    assert row["status"] == "WORKER_ERROR"
    assert "ValueError" in row["error"]


def test_worker_error_redacts_modern_legacy_and_hosted_secret_values() -> None:
    modern = "sb_secret_superprivate"
    legacy = "eyJlegacy-superprivate"
    hosted = "sb_secret_hostedprivate"
    stderr = io.StringIO()

    def runner(**kwargs):
        raise RuntimeError(
            f"failure {modern} / {legacy} / {hosted}"
        )

    code = main(
        [],
        env={
            "BRIAN_RUNTIME_ID": "runtime-93",
            "SUPABASE_SECRET_KEY": modern,
            "SUPABASE_SERVICE_ROLE_KEY": legacy,
            "SUPABASE_SECRET_KEYS": json.dumps({"default": hosted}),
        },
        stdin=io.StringIO("{}"),
        stdout=io.StringIO(),
        stderr=stderr,
        worker_runner=runner,
    )
    assert code == EXIT_WORKER_ERROR
    text = stderr.getvalue()
    assert modern not in text
    assert legacy not in text
    assert hosted not in text
    assert "<redacted>" in text


def test_generated_worker_token_is_process_unique_shape_when_not_configured() -> None:
    seen = {}

    def runner(**kwargs):
        seen.update(kwargs)
        return _receipt()

    code = main(
        [],
        env={"BRIAN_RUNTIME_ID": "runtime-93"},
        stdin=io.StringIO("{}"),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        worker_runner=runner,
        clock=lambda: 1.0,
    )
    assert code == EXIT_READY
    assert seen["recovery_worker_token"].startswith("phase93-")


def test_input_file_is_read_instead_of_stdin(tmp_path) -> None:
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    seen = {}

    def runner(**kwargs):
        seen.update(kwargs)
        return _receipt()

    code = main(
        ["--input", str(path)],
        env={"BRIAN_RUNTIME_ID": "runtime-93"},
        stdin=io.StringIO("{definitely-bad"),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        worker_runner=runner,
        clock=lambda: 1.0,
    )
    assert code == EXIT_READY
    assert "BTCUSDT" in seen["recovery_markets"]
