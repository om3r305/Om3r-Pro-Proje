from __future__ import annotations

import pytest

from brian2026.phase87_recovery_restart_resume import (
    RecoveryRestartWorkError,
    RecoveryRestartWorkItem,
    RecoveryRestartWorkStore,
)


def _active(**overrides):
    row = {
        "has_work": True,
        "status": "WORK",
        "runtime_id": "runtime-87",
        "original_cycle_id": "o" * 64,
        "dispatch_id": "d" * 64,
        "cancel_risk_version": 7,
        "cancel_risk_receipt_id": "r" * 64,
        "cancel_reason": "REDUCING_NEW_RISK",
        "requested_at": "2026-09-23T12:00:00Z",
        "runtime_version": 14,
        "runtime_checkpoint_id": "k" * 64,
        "runtime_head_state_id": "h" * 64,
        "directive_exists": True,
        "recovery_status": "READY_REDUCE_ONLY",
        "directive_runtime_version": 14,
        "directive_state_id": "h" * 64,
        "directive_prepared_at": "2026-09-23T12:00:01Z",
        "claim_status": None,
        "claim_worker_token": None,
        "claim_fencing_token": None,
        "claim_until": None,
        "recovery_cycle_id": None,
        "progress_runtime_version": None,
        "progress_head_state_id": None,
        "progress_checkpoint_id": None,
        "started": False,
        "started_at": None,
        "recovery_journal_stage": None,
        "phase84_terminal_event": False,
        "work_state": "NEEDS_CLAIM",
    }
    row.update(overrides)
    return row


class FakeRpc:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, name, params):
        self.calls.append((name, dict(params)))
        return dict(self.payload)


def test_idle_reader_returns_none_action() -> None:
    rpc = FakeRpc({
        "has_work": False,
        "status": "IDLE",
        "runtime_id": "runtime-87",
    })
    item = RecoveryRestartWorkStore(rpc).read_next(runtime_id="runtime-87")
    assert item.has_work is False
    assert item.work_state == "IDLE"
    assert item.action == "NONE"
    assert rpc.calls == [
        (
            "brian_read_next_shadow_recovery_work",
            {"p_runtime_id": "runtime-87"},
        )
    ]


def test_needs_directive_is_the_only_active_state_without_directive() -> None:
    row = _active(
        directive_exists=False,
        recovery_status=None,
        directive_runtime_version=None,
        directive_state_id=None,
        directive_prepared_at=None,
        work_state="NEEDS_DIRECTIVE",
    )
    item = RecoveryRestartWorkStore(FakeRpc(row)).read_next(
        runtime_id="runtime-87"
    )
    assert item.action == "PREPARE_DIRECTIVE"

    row["work_state"] = "NEEDS_CLAIM"
    with pytest.raises(ValueError, match="requires durable directive"):
        RecoveryRestartWorkStore(FakeRpc(row)).read_next(
            runtime_id="runtime-87"
        )


def test_claim_expired_routes_to_takeover() -> None:
    row = _active(
        claim_status="CLAIMED",
        claim_worker_token="dead-worker",
        claim_fencing_token=3,
        claim_until="2026-09-23T12:01:00Z",
        work_state="CLAIM_EXPIRED",
    )
    item = RecoveryRestartWorkStore(FakeRpc(row)).read_next(
        runtime_id="runtime-87"
    )
    assert item.action == "TAKE_OVER_CLAIM"
    assert item.claim_fencing_token == 3


def test_started_without_recovery_cycle_routes_to_execution() -> None:
    row = _active(
        claim_status="CLAIMED",
        claim_worker_token="worker-a",
        claim_fencing_token=4,
        claim_until="2026-09-23T12:10:00Z",
        started=True,
        started_at="2026-09-23T12:02:00Z",
        work_state="STARTED_NEEDS_EXECUTION",
    )
    item = RecoveryRestartWorkStore(FakeRpc(row)).read_next(
        runtime_id="runtime-87"
    )
    assert item.action == "EXECUTE_RECOVERY"


def test_recovery_progress_requires_durable_progress_anchors() -> None:
    row = _active(
        claim_status="CLAIMED",
        claim_worker_token="worker-a",
        claim_fencing_token=5,
        claim_until="2026-09-23T12:10:00Z",
        started=True,
        started_at="2026-09-23T12:02:00Z",
        recovery_cycle_id="c" * 64,
        progress_runtime_version=15,
        progress_head_state_id="h" * 64,
        progress_checkpoint_id="p" * 64,
        recovery_journal_stage="PAPER_APPLIED",
        work_state="RECOVERY_PROGRESS",
    )
    item = RecoveryRestartWorkStore(FakeRpc(row)).read_next(
        runtime_id="runtime-87"
    )
    assert item.action == "RESUME_RECOVERY"

    row["progress_checkpoint_id"] = None
    with pytest.raises(ValueError, match="progress checkpoint"):
        RecoveryRestartWorkStore(FakeRpc(row)).read_next(
            runtime_id="runtime-87"
        )


def test_needs_audit_requires_exact_phase84_terminal_evidence() -> None:
    row = _active(
        claim_status="CLAIMED",
        claim_worker_token="worker-a",
        claim_fencing_token=6,
        claim_until="2026-09-23T12:10:00Z",
        started=True,
        started_at="2026-09-23T12:02:00Z",
        recovery_cycle_id="c" * 64,
        progress_runtime_version=16,
        progress_head_state_id="f" * 64,
        progress_checkpoint_id="p" * 64,
        recovery_journal_stage="COMMITTED",
        phase84_terminal_event=True,
        work_state="NEEDS_AUDIT",
    )
    item = RecoveryRestartWorkStore(FakeRpc(row)).read_next(
        runtime_id="runtime-87"
    )
    assert item.action == "AUDIT"

    row["phase84_terminal_event"] = False
    with pytest.raises(ValueError, match="terminal Phase84"):
        RecoveryRestartWorkStore(FakeRpc(row)).read_next(
            runtime_id="runtime-87"
        )


def test_manual_review_and_completed_without_certificate_are_fail_closed_routes() -> None:
    manual = RecoveryRestartWorkStore(
        FakeRpc(_active(
            recovery_status="MANUAL_REVIEW",
            work_state="MANUAL_REVIEW",
        ))
    ).read_next(runtime_id="runtime-87")
    assert manual.action == "MANUAL_REVIEW"

    broken = RecoveryRestartWorkStore(
        FakeRpc(_active(
            claim_status="COMPLETED",
            claim_worker_token="worker-a",
            claim_fencing_token=7,
            work_state="COMPLETED_WITHOUT_CERTIFICATE",
        ))
    ).read_next(runtime_id="runtime-87")
    assert broken.action == "FAIL_CLOSED"


def test_reader_rejects_identity_and_boolean_drift() -> None:
    row = _active()
    row["runtime_id"] = "other-runtime"
    with pytest.raises(RecoveryRestartWorkError, match="runtime_id"):
        RecoveryRestartWorkStore(FakeRpc(row)).read_next(
            runtime_id="runtime-87"
        )

    row = _active()
    row["directive_exists"] = 1
    with pytest.raises(RecoveryRestartWorkError, match="directive_exists"):
        RecoveryRestartWorkStore(FakeRpc(row)).read_next(
            runtime_id="runtime-87"
        )


def test_active_work_rejects_malformed_hashes_and_unknown_state() -> None:
    with pytest.raises(RecoveryRestartWorkError, match="dispatch_id"):
        RecoveryRestartWorkStore(
            FakeRpc(_active(dispatch_id="bad"))
        ).read_next(runtime_id="runtime-87")

    row = _active(work_state="MAGIC")
    with pytest.raises(ValueError, match="unsupported"):
        RecoveryRestartWorkStore(FakeRpc(row)).read_next(
            runtime_id="runtime-87"
        )


def test_dataclass_rejects_live_boundary() -> None:
    with pytest.raises(ValueError, match="shadow-only"):
        RecoveryRestartWorkItem(
            runtime_id="runtime-87",
            has_work=False,
            status="IDLE",
            work_state="IDLE",
            shadow_only=False,
        )
