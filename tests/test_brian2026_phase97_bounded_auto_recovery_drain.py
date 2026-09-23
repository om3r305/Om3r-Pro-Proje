from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase71_persisted_runtime_supervisor import PersistedRuntimeStaleError
from brian2026.phase86_recovery_admission_interlock import RecoveryAdmissionState
from brian2026.phase89_recovery_startup_gate import RecoveryStartupGateReceipt
from brian2026.phase95_auto_binance_recovery_worker import AutoRecoveryStartupReceipt
from brian2026.phase97_bounded_auto_recovery_drain import (
    BoundedAutoRecoveryDrainError,
    run_bounded_auto_binance_recovery,
)


RUNTIME = "runtime-97"
CYCLE = "c" * 64
RISK = "r" * 64


def _admission(*, blocked: bool, runtime_id: str = RUNTIME) -> RecoveryAdmissionState:
    if blocked:
        return RecoveryAdmissionState(
            runtime_id=runtime_id,
            status="RECOVERY_BARRIER",
            blocked=True,
            original_cycle_id=CYCLE,
            cancel_risk_receipt_id=RISK,
            reason="AFTER_START_CANCEL",
        )
    return RecoveryAdmissionState(
        runtime_id=runtime_id,
        status="OPEN",
        blocked=False,
    )


def _attempt(
    *,
    status: str,
    ready: bool,
    blocked: bool,
    processed: int,
    outcome: str,
    runtime_id: str = RUNTIME,
) -> AutoRecoveryStartupReceipt:
    gate = RecoveryStartupGateReceipt(
        runtime_id=runtime_id,
        steps=(SimpleNamespace(outcome=outcome),),
        admission=_admission(blocked=blocked, runtime_id=runtime_id),
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
        directive_status=None,
        evidence_assets=(),
        evidence_observed_at=(),
        decision_at=100.0,
    )


class _AdmissionReader:
    def __init__(self, states):
        self.states = list(states)
        self.calls = 0

    def read(self, *, runtime_id: str):
        assert runtime_id == RUNTIME
        self.calls += 1
        if not self.states:
            raise AssertionError("unexpected Phase86 admission read")
        return self.states.pop(0)


class _Session:
    def __init__(self, admission_states=()):
        self.runtime_id = RUNTIME
        self.closed = False
        self.runtime_supervisor = SimpleNamespace(valid=True)
        self.stack = SimpleNamespace(
            admission=_AdmissionReader(admission_states),
        )


def _run(session, runner, *, max_items=4):
    marker_factory = lambda: object()
    marker_clock = lambda: 100.0
    result = run_bounded_auto_binance_recovery(
        session,
        max_items=max_items,
        recovery_worker_token="worker-97",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        source_ref="phase97:test",
        provider_factory=marker_factory,
        clock=marker_clock,
        item_runner=runner,
    )
    return result, marker_factory, marker_clock


def test_idle_ready_rechecks_phase86_before_releasing_normal_work() -> None:
    session = _Session([_admission(blocked=False)])
    seen = []

    def runner(*args, **kwargs):
        seen.append((args, kwargs))
        return _attempt(
            status="READY_FOR_NORMAL_WORK",
            ready=True,
            blocked=False,
            processed=0,
            outcome="IDLE",
        )

    result, provider_factory, clock = _run(session, runner)

    assert result.ready_for_normal_work is True
    assert result.status == "READY_FOR_NORMAL_WORK"
    assert result.processed_items == 0
    assert len(result.attempts) == 1
    assert result.final_admission.blocked is False
    assert session.stack.admission.calls == 1
    assert seen[0][0] == (session,)
    assert seen[0][1]["provider_factory"] is provider_factory
    assert seen[0][1]["clock"] is clock
    assert seen[0][1]["recovery_worker_token"] == "worker-97"


def test_multiple_recovery_items_use_separate_phase95_attempts_same_session() -> None:
    session = _Session([_admission(blocked=False)])
    receipts = iter([
        _attempt(
            status="RECOVERY_BUDGET_EXHAUSTED",
            ready=False,
            blocked=True,
            processed=1,
            outcome="RECOVERY_COMPLETED",
        ),
        _attempt(
            status="READY_FOR_NORMAL_WORK",
            ready=True,
            blocked=False,
            processed=1,
            outcome="RECOVERY_COMPLETED",
        ),
    ])
    seen_sessions = []

    def runner(session_arg, **kwargs):
        seen_sessions.append(session_arg)
        return next(receipts)

    result, _, _ = _run(session, runner)

    assert result.ready_for_normal_work is True
    assert result.processed_items == 2
    assert len(result.attempts) == 2
    assert seen_sessions == [session, session]
    assert session.stack.admission.calls == 1


def test_ready_to_blocked_race_is_drained_with_fresh_second_attempt() -> None:
    session = _Session([
        _admission(blocked=True),
        _admission(blocked=False),
    ])
    receipts = iter([
        _attempt(
            status="READY_FOR_NORMAL_WORK",
            ready=True,
            blocked=False,
            processed=0,
            outcome="IDLE",
        ),
        _attempt(
            status="READY_FOR_NORMAL_WORK",
            ready=True,
            blocked=False,
            processed=1,
            outcome="RECOVERY_COMPLETED",
        ),
    ])

    result, _, _ = _run(session, lambda *args, **kwargs: next(receipts))

    assert result.ready_for_normal_work is True
    assert result.status == "READY_FOR_NORMAL_WORK"
    assert len(result.attempts) == 2
    assert result.processed_items == 1
    assert session.stack.admission.calls == 2


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        ("RECOVERY_BLOCKED", "WAIT_RISK_RELEASE"),
        ("RECOVERY_BLOCKED", "MANUAL_REVIEW_REQUIRED"),
        ("RECOVERY_OUTCOME_NOT_TERMINAL", "RECONCILIATION_BLOCKED"),
    ],
)
def test_nonterminal_recovery_stops_without_spinning(status: str, outcome: str) -> None:
    session = _Session()
    calls = 0

    def runner(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _attempt(
            status=status,
            ready=False,
            blocked=True,
            processed=1,
            outcome=outcome,
        )

    result, _, _ = _run(session, runner)

    assert result.ready_for_normal_work is False
    assert result.status == status
    assert len(result.attempts) == 1
    assert calls == 1
    assert session.stack.admission.calls == 0


def test_retryable_items_stop_at_drain_budget() -> None:
    session = _Session()
    calls = 0

    def runner(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _attempt(
            status="RECOVERY_BUDGET_EXHAUSTED",
            ready=False,
            blocked=True,
            processed=1,
            outcome="RECOVERY_COMPLETED",
        )

    result, _, _ = _run(session, runner, max_items=2)

    assert result.ready_for_normal_work is False
    assert result.status == "RECOVERY_DRAIN_BUDGET_EXHAUSTED"
    assert result.processed_items == 2
    assert len(result.attempts) == 2
    assert calls == 2
    assert result.final_admission.blocked is True


def test_barrier_after_ready_consumes_remaining_bounded_attempt_budget() -> None:
    session = _Session([_admission(blocked=True)])
    calls = 0

    def runner(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _attempt(
            status="READY_FOR_NORMAL_WORK",
            ready=True,
            blocked=False,
            processed=0,
            outcome="IDLE",
        )

    result, _, _ = _run(session, runner, max_items=1)

    assert result.ready_for_normal_work is False
    assert result.status == "RECOVERY_DRAIN_BUDGET_EXHAUSTED"
    assert result.final_admission.blocked is True
    assert calls == 1


def test_cross_runtime_phase95_receipt_fails_closed() -> None:
    session = _Session()
    other = "runtime-other"

    with pytest.raises(
        BoundedAutoRecoveryDrainError,
        match="runtime differs",
    ):
        _run(
            session,
            lambda *args, **kwargs: _attempt(
                status="RECOVERY_BLOCKED",
                ready=False,
                blocked=True,
                processed=1,
                outcome="WAIT_RISK_RELEASE",
                runtime_id=other,
            ),
        )


def test_stale_or_closed_session_fails_before_item_runner() -> None:
    session = _Session()
    session.runtime_supervisor.valid = False
    with pytest.raises(PersistedRuntimeStaleError, match="became stale"):
        _run(session, lambda *args, **kwargs: pytest.fail("must not run"))

    closed = _Session()
    closed.closed = True
    with pytest.raises(BoundedAutoRecoveryDrainError, match="closed"):
        _run(closed, lambda *args, **kwargs: pytest.fail("must not run"))


@pytest.mark.parametrize("max_items", [0, -1, 33])
def test_invalid_drain_budget_is_rejected(max_items: int) -> None:
    session = _Session()
    with pytest.raises(ValueError, match="max_items"):
        _run(
            session,
            lambda *args, **kwargs: pytest.fail("must not run"),
            max_items=max_items,
        )


def test_missing_phase86_reader_fails_closed_on_ready_handoff() -> None:
    session = _Session()
    session.stack = SimpleNamespace(admission=None)

    with pytest.raises(
        BoundedAutoRecoveryDrainError,
        match="Phase86 admission reader",
    ):
        _run(
            session,
            lambda *args, **kwargs: _attempt(
                status="READY_FOR_NORMAL_WORK",
                ready=True,
                blocked=False,
                processed=0,
                outcome="IDLE",
            ),
        )
