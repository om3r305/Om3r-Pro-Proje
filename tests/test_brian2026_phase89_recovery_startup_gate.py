from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase86_recovery_admission_interlock import RecoveryAdmissionState
from brian2026.phase87_recovery_restart_resume import RecoveryRestartWorkItem
from brian2026.phase88_recovery_restart_orchestrator import RecoveryRestartStep
from brian2026.phase89_recovery_startup_gate import RecoveryStartupGate


RUNTIME_ID = "runtime-89"
CHECKPOINT = "k" * 64


def _idle_work():
    return RecoveryRestartWorkItem(
        runtime_id=RUNTIME_ID,
        has_work=False,
        status="IDLE",
        work_state="IDLE",
    )


def _active_work():
    return RecoveryRestartWorkItem(
        runtime_id=RUNTIME_ID,
        has_work=True,
        status="WORK",
        work_state="NEEDS_CLAIM",
        original_cycle_id="o" * 64,
        dispatch_id="d" * 64,
        cancel_risk_version=7,
        cancel_risk_receipt_id="r" * 64,
        cancel_reason="REDUCING_NEW_RISK",
        runtime_version=10,
        runtime_checkpoint_id=CHECKPOINT,
        runtime_head_state_id="h" * 64,
        directive_exists=True,
        recovery_status="READY_REDUCE_ONLY",
        directive_runtime_version=10,
        directive_state_id="h" * 64,
    )


def _step(outcome):
    return RecoveryRestartStep(
        work=_idle_work() if outcome == "IDLE" else _active_work(),
        directive=None,
        claim=None,
        start=None,
        execution=None,
        audit=None,
        outcome=outcome,
        persisted_version=10,
        checkpoint_id=CHECKPOINT,
    )


def _open():
    return RecoveryAdmissionState(
        runtime_id=RUNTIME_ID,
        status="OPEN",
        blocked=False,
    )


def _blocked():
    return RecoveryAdmissionState(
        runtime_id=RUNTIME_ID,
        status="RECOVERY_BARRIER",
        blocked=True,
        original_cycle_id="o" * 64,
        cancel_risk_receipt_id="r" * 64,
        reason="REDUCING_NEW_RISK",
    )


class _Recovery:
    def __init__(self, outcomes):
        self.steps = [_step(value) for value in outcomes]
        self.calls = 0
        self.runtime_supervisor = SimpleNamespace(runtime_id=RUNTIME_ID)

    def resume_next(self, **kwargs):
        assert kwargs["recovery_worker_token"] == "recovery-a"
        self.calls += 1
        if not self.steps:
            raise AssertionError("startup gate over-drained recovery work")
        return self.steps.pop(0)


class _Admission:
    def __init__(self, state):
        self.state = state
        self.calls = 0

    def read(self, *, runtime_id):
        assert runtime_id == RUNTIME_ID
        self.calls += 1
        return self.state


def _run(outcomes, admission, *, max_items=4):
    recovery = _Recovery(outcomes)
    admissions = _Admission(admission)
    receipt = RecoveryStartupGate(
        recovery=recovery,
        admission=admissions,
    ).run(
        max_items=max_items,
        recovery_worker_token="recovery-a",
        recovery_claim_seconds=30,
        recovery_markets={},
        recovery_risk_limits_by_asset={},
        recovery_ttl_seconds=60,
        marks={},
        observed_at=1.0,
        source_ref="phase89",
    )
    return receipt, recovery, admissions


def test_idle_plus_open_admission_releases_normal_work() -> None:
    receipt, recovery, admission = _run(["IDLE"], _open())
    assert receipt.ready_for_normal_work is True
    assert receipt.status == "READY_FOR_NORMAL_WORK"
    assert receipt.processed_items == 0
    assert recovery.calls == 1
    assert admission.calls == 1


def test_multiple_completed_items_are_bounded_then_idle_releases_gate() -> None:
    receipt, recovery, _ = _run(
        ["RECOVERY_COMPLETED", "NO_RECOVERY_REQUIRED", "IDLE"],
        _open(),
        max_items=4,
    )
    assert receipt.ready_for_normal_work is True
    assert receipt.status == "READY_FOR_NORMAL_WORK"
    assert receipt.processed_items == 2
    assert recovery.calls == 3


def test_nonterminal_wait_stops_immediately_and_never_spins() -> None:
    receipt, recovery, _ = _run(
        ["WAIT_RISK_RELEASE", "RECOVERY_COMPLETED"],
        _blocked(),
    )
    assert receipt.ready_for_normal_work is False
    assert receipt.status == "RECOVERY_BLOCKED"
    assert receipt.processed_items == 1
    assert recovery.calls == 1


def test_manual_review_stops_immediately() -> None:
    receipt, recovery, _ = _run(
        ["MANUAL_REVIEW_REQUIRED", "RECOVERY_COMPLETED"],
        _blocked(),
    )
    assert receipt.ready_for_normal_work is False
    assert receipt.status == "RECOVERY_BLOCKED"
    assert recovery.calls == 1


def test_new_barrier_between_phase87_idle_and_final_admission_stays_closed() -> None:
    receipt, _, _ = _run(["IDLE"], _blocked())
    assert receipt.ready_for_normal_work is False
    assert receipt.status == "RECOVERY_BARRIER_APPEARED"


def test_budget_exhaustion_never_releases_blocked_backlog() -> None:
    receipt, recovery, _ = _run(
        ["RECOVERY_COMPLETED", "RECOVERY_COMPLETED"],
        _blocked(),
        max_items=2,
    )
    assert receipt.ready_for_normal_work is False
    assert receipt.status == "RECOVERY_BUDGET_EXHAUSTED"
    assert receipt.processed_items == 2
    assert recovery.calls == 2


def test_open_admission_does_not_override_nonterminal_worker_outcome() -> None:
    receipt, _, _ = _run(["WAIT_ACTIVE_RECOVERY_OWNER"], _open())
    assert receipt.ready_for_normal_work is False
    assert receipt.status == "RECOVERY_OUTCOME_NOT_TERMINAL"


def test_invalid_budget_is_rejected_before_worker_call() -> None:
    recovery = _Recovery(["IDLE"])
    gate = RecoveryStartupGate(
        recovery=recovery,
        admission=_Admission(_open()),
    )
    with pytest.raises(ValueError, match="max_items"):
        gate.run(
            max_items=0,
            recovery_worker_token="recovery-a",
            recovery_claim_seconds=30,
            recovery_markets={},
            recovery_risk_limits_by_asset={},
            recovery_ttl_seconds=60,
            marks={},
            observed_at=1.0,
            source_ref="phase89",
        )
    assert recovery.calls == 0
