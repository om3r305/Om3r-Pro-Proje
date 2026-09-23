from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase71_persisted_runtime_supervisor import PersistedRuntimeStaleError
from brian2026.phase86_recovery_admission_interlock import RecoveryAdmissionState
from brian2026.phase89_recovery_startup_gate import RecoveryStartupGateReceipt
from brian2026.phase95_auto_binance_recovery_worker import AutoRecoveryStartupReceipt
from brian2026.phase97_bounded_auto_recovery_drain import (
    BoundedAutoRecoveryDrainReceipt,
)
from brian2026.phase100_recovery_first_shadow_worker import (
    RecoveryFirstShadowWorkerError,
    RecoveryFirstShadowWorkerSession,
)


RUNTIME = "runtime-100"
CYCLE = "c" * 64
RISK = "r" * 64


def _admission(*, blocked: bool):
    if blocked:
        return RecoveryAdmissionState(
            runtime_id=RUNTIME,
            status="RECOVERY_BARRIER",
            blocked=True,
            original_cycle_id=CYCLE,
            cancel_risk_receipt_id=RISK,
            reason="AFTER_START_CANCEL",
        )
    return RecoveryAdmissionState(
        runtime_id=RUNTIME,
        status="OPEN",
        blocked=False,
    )


def _recovery(
    *,
    ready: bool,
    status: str,
    blocked: bool,
) -> BoundedAutoRecoveryDrainReceipt:
    admission = _admission(blocked=blocked)
    gate = RecoveryStartupGateReceipt(
        runtime_id=RUNTIME,
        steps=(SimpleNamespace(outcome="IDLE"),),
        admission=admission,
        status=status,
        ready_for_normal_work=ready,
        processed_items=0,
        max_items=1,
    )
    attempt = AutoRecoveryStartupReceipt(
        gate=gate,
        original_cycle_id=None,
        cancel_risk_receipt_id=None,
        preflight_work_state="IDLE",
        directive_status=None,
        evidence_assets=(),
        evidence_observed_at=(),
        decision_at=100.0,
    )
    return BoundedAutoRecoveryDrainReceipt(
        runtime_id=RUNTIME,
        attempts=(attempt,),
        final_admission=admission,
        status=status,
        ready_for_normal_work=ready,
        processed_items=0,
        max_items=1,
    )


class _Session:
    def __init__(self):
        self.runtime_id = RUNTIME
        self.runtime_supervisor = SimpleNamespace(valid=True)
        self.closed = False
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        self.closed = True
        return True


class _Handoff:
    def __init__(self):
        self.calls = []

    def process_governed_cycle(self, governed, **kwargs):
        self.calls.append((governed, kwargs))
        return SimpleNamespace(
            shadow_only=True,
            live_execution=False,
            status="NORMAL_SHADOW_PROCESSED",
        )


def test_blocked_recovery_does_not_construct_handoff_or_release_normal_work() -> None:
    session = _Session()
    calls = {"handoff": 0}

    def recovery_runner(session_arg, **kwargs):
        assert session_arg is session
        return _recovery(
            ready=False,
            status="RECOVERY_BLOCKED",
            blocked=True,
        )

    def handoff_factory(**kwargs):
        calls["handoff"] += 1
        raise AssertionError("handoff must not be built")

    worker = RecoveryFirstShadowWorkerSession(
        session=session,
        recovery_runner=recovery_runner,
        handoff_factory=handoff_factory,
    )
    startup = worker.run_recovery_gate(
        max_items=4,
        recovery_worker_token="recovery-100",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        source_ref="phase100:test",
        clock=lambda: 100.0,
    )

    assert startup.ready_for_normal_shadow is False
    assert startup.status == "RECOVERY_BLOCKED"
    assert worker.ready_for_normal_shadow is False
    assert calls["handoff"] == 0

    with pytest.raises(
        RecoveryFirstShadowWorkerError,
        match="not released",
    ):
        worker.process_governed_cycle(
            SimpleNamespace(),
            worker_token="normal-100",
            claim_seconds=30,
            marks={},
            observed_at=101.0,
            source_ref="phase100:normal",
        )


def test_blocked_startup_can_retry_then_release_same_session() -> None:
    session = _Session()
    receipts = iter([
        _recovery(
            ready=False,
            status="RECOVERY_BLOCKED",
            blocked=True,
        ),
        _recovery(
            ready=True,
            status="READY_FOR_NORMAL_WORK",
            blocked=False,
        ),
    ])
    handoff = _Handoff()
    seen_sessions = []

    def recovery_runner(session_arg, **kwargs):
        seen_sessions.append(session_arg)
        return next(receipts)

    def handoff_factory(**kwargs):
        assert kwargs["session"] is session
        assert kwargs["recovery"].ready_for_normal_work is True
        return handoff

    worker = RecoveryFirstShadowWorkerSession(
        session=session,
        recovery_runner=recovery_runner,
        handoff_factory=handoff_factory,
    )

    first = worker.run_recovery_gate(
        max_items=1,
        recovery_worker_token="recovery-100",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        source_ref="phase100:first",
    )
    second = worker.run_recovery_gate(
        max_items=4,
        recovery_worker_token="recovery-100",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        source_ref="phase100:second",
    )

    assert first.ready_for_normal_shadow is False
    assert second.ready_for_normal_shadow is True
    assert second.status == "READY_FOR_NORMAL_SHADOW"
    assert worker.ready_for_normal_shadow is True
    assert seen_sessions == [session, session]

    governed = SimpleNamespace(shadow_only=True, live_execution=False)
    result = worker.process_governed_cycle(
        governed,
        worker_token="normal-100",
        claim_seconds=45,
        marks={"BTCUSDT": 100.0},
        observed_at=101.0,
        source_ref="phase100:normal",
    )
    assert result.status == "NORMAL_SHADOW_PROCESSED"
    assert len(handoff.calls) == 1
    seen_governed, kwargs = handoff.calls[0]
    assert seen_governed is governed
    assert kwargs == {
        "worker_token": "normal-100",
        "claim_seconds": 45,
        "marks": {"BTCUSDT": 100.0},
        "observed_at": 101.0,
        "source_ref": "phase100:normal",
    }


def test_ready_recovery_cannot_be_reexecuted_after_handoff_release() -> None:
    session = _Session()
    calls = 0

    def recovery_runner(session_arg, **kwargs):
        nonlocal calls
        calls += 1
        return _recovery(
            ready=True,
            status="READY_FOR_NORMAL_WORK",
            blocked=False,
        )

    worker = RecoveryFirstShadowWorkerSession(
        session=session,
        recovery_runner=recovery_runner,
        handoff_factory=lambda **kwargs: _Handoff(),
    )
    startup = worker.run_recovery_gate(
        max_items=4,
        recovery_worker_token="recovery-100",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        source_ref="phase100:test",
    )
    assert startup.ready_for_normal_shadow is True

    with pytest.raises(
        RecoveryFirstShadowWorkerError,
        match="already released",
    ):
        worker.run_recovery_gate(
            max_items=4,
            recovery_worker_token="recovery-100",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            source_ref="phase100:test",
        )
    assert calls == 1


def test_cross_runtime_recovery_fails_closed() -> None:
    session = _Session()
    other = "runtime-other"
    open_admission = RecoveryAdmissionState(
        runtime_id=other,
        status="OPEN",
        blocked=False,
    )
    gate = RecoveryStartupGateReceipt(
        runtime_id=other,
        steps=(SimpleNamespace(outcome="IDLE"),),
        admission=open_admission,
        status="READY_FOR_NORMAL_WORK",
        ready_for_normal_work=True,
        processed_items=0,
        max_items=1,
    )
    attempt = AutoRecoveryStartupReceipt(
        gate=gate,
        original_cycle_id=None,
        cancel_risk_receipt_id=None,
        preflight_work_state="IDLE",
        directive_status=None,
        evidence_assets=(),
        evidence_observed_at=(),
        decision_at=100.0,
    )
    recovery = BoundedAutoRecoveryDrainReceipt(
        runtime_id=other,
        attempts=(attempt,),
        final_admission=open_admission,
        status="READY_FOR_NORMAL_WORK",
        ready_for_normal_work=True,
        processed_items=0,
        max_items=1,
    )

    worker = RecoveryFirstShadowWorkerSession(
        session=session,
        recovery_runner=lambda *args, **kwargs: recovery,
        handoff_factory=lambda **kwargs: _Handoff(),
    )

    with pytest.raises(
        RecoveryFirstShadowWorkerError,
        match="recovery runtime differs",
    ):
        worker.run_recovery_gate(
            max_items=1,
            recovery_worker_token="recovery-100",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            source_ref="phase100:test",
        )


def test_stale_or_closed_session_fails_closed() -> None:
    stale = _Session()
    stale.runtime_supervisor.valid = False
    with pytest.raises(PersistedRuntimeStaleError, match="valid Phase71"):
        RecoveryFirstShadowWorkerSession(session=stale)

    closed = _Session()
    closed.closed = True
    with pytest.raises(RecoveryFirstShadowWorkerError, match="closed"):
        RecoveryFirstShadowWorkerSession(session=closed)


def test_runtime_becoming_stale_before_recovery_is_rejected() -> None:
    session = _Session()
    worker = RecoveryFirstShadowWorkerSession(
        session=session,
        recovery_runner=lambda *args, **kwargs: pytest.fail("must not run"),
    )
    session.runtime_supervisor.valid = False

    with pytest.raises(PersistedRuntimeStaleError, match="became stale"):
        worker.run_recovery_gate(
            max_items=1,
            recovery_worker_token="recovery-100",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            source_ref="phase100:test",
        )


def test_owned_from_env_session_is_closed_on_context_exit() -> None:
    session = _Session()
    seen = {}

    def session_factory(**kwargs):
        seen.update(kwargs)
        return session

    with RecoveryFirstShadowWorkerSession.from_env(
        env={"BRIAN_RUNTIME_ID": RUNTIME},
        initial_runtime="initial",
        foreign_cycle_aborter="aborter",
        client="client",
        session_factory=session_factory,
        recovery_runner=lambda *args, **kwargs: _recovery(
            ready=False,
            status="RECOVERY_BLOCKED",
            blocked=True,
        ),
    ) as worker:
        assert worker.closed is False
        assert session.close_calls == 0

    assert session.close_calls == 1
    assert session.closed is True
    assert worker.closed is True
    assert seen == {
        "env": {"BRIAN_RUNTIME_ID": RUNTIME},
        "initial_runtime": "initial",
        "foreign_cycle_aborter": "aborter",
        "client": "client",
    }


def test_external_session_is_not_closed_by_wrapper() -> None:
    session = _Session()
    worker = RecoveryFirstShadowWorkerSession(session=session)

    assert worker.close() is True
    assert worker.close() is True
    assert worker.closed is True
    assert session.close_calls == 0
    assert session.closed is False


def test_from_env_constructor_failure_closes_opened_session() -> None:
    session = _Session()

    with pytest.raises(TypeError, match="recovery_runner"):
        RecoveryFirstShadowWorkerSession.from_env(
            env={"BRIAN_RUNTIME_ID": RUNTIME},
            session_factory=lambda **kwargs: session,
            recovery_runner=None,
        )

    assert session.close_calls == 1
    assert session.closed is True


def test_handoff_factory_failure_does_not_publish_ready_startup() -> None:
    session = _Session()
    worker = RecoveryFirstShadowWorkerSession(
        session=session,
        recovery_runner=lambda *args, **kwargs: _recovery(
            ready=True,
            status="READY_FOR_NORMAL_WORK",
            blocked=False,
        ),
        handoff_factory=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("handoff assembly failed")
        ),
    )

    with pytest.raises(RuntimeError, match="assembly failed"):
        worker.run_recovery_gate(
            max_items=1,
            recovery_worker_token="recovery-100",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            source_ref="phase100:test",
        )

    assert worker.startup is None
    assert worker.ready_for_normal_shadow is False


def test_process_rejected_after_worker_close_even_if_previously_ready() -> None:
    session = _Session()
    worker = RecoveryFirstShadowWorkerSession(
        session=session,
        recovery_runner=lambda *args, **kwargs: _recovery(
            ready=True,
            status="READY_FOR_NORMAL_WORK",
            blocked=False,
        ),
        handoff_factory=lambda **kwargs: _Handoff(),
    )
    worker.run_recovery_gate(
        max_items=1,
        recovery_worker_token="recovery-100",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        source_ref="phase100:test",
    )
    worker.close()

    with pytest.raises(RecoveryFirstShadowWorkerError, match="closed"):
        worker.process_governed_cycle(
            SimpleNamespace(shadow_only=True, live_execution=False),
            worker_token="normal-100",
            claim_seconds=30,
            marks={},
            observed_at=101.0,
            source_ref="phase100:normal",
        )
