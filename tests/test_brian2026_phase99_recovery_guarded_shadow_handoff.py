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
from brian2026.phase99_recovery_guarded_shadow_handoff import (
    RecoveryGuardedShadowBlockedError,
    RecoveryGuardedShadowHandoff,
    RecoveryGuardedShadowHandoffError,
    build_normal_shadow_execution_stack,
)


RUNTIME = "runtime-99"
CYCLE = "c" * 64
RISK = "r" * 64


def _admission(*, blocked: bool, runtime_id: str = RUNTIME):
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


def _recovery(
    *,
    ready: bool = True,
    status: str = "READY_FOR_NORMAL_WORK",
    blocked: bool = False,
    runtime_id: str = RUNTIME,
):
    admission = _admission(blocked=blocked, runtime_id=runtime_id)
    gate = RecoveryStartupGateReceipt(
        runtime_id=runtime_id,
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
        runtime_id=runtime_id,
        attempts=(attempt,),
        final_admission=admission,
        status=status,
        ready_for_normal_work=ready,
        processed_items=0,
        max_items=1,
    )


class _AdmissionReader:
    def __init__(self, states):
        self.states = list(states)
        self.calls = 0

    def read(self, *, runtime_id: str):
        assert runtime_id == RUNTIME
        self.calls += 1
        if not self.states:
            raise AssertionError("unexpected admission read")
        return self.states.pop(0)


class _Execution:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result = result or SimpleNamespace(
            shadow_only=True,
            live_execution=False,
            outcome="COMMITTED",
        )
        self.error = error

    def process_governed_cycle(self, governed, **kwargs):
        self.calls.append((governed, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def _session(states=()):
    supervisor = SimpleNamespace(valid=True)
    admission = _AdmissionReader(states)
    return SimpleNamespace(
        closed=False,
        runtime_id=RUNTIME,
        runtime_supervisor=supervisor,
        rpc=lambda name, params: {},
        stack=SimpleNamespace(admission=admission),
    )


def _normal(session, execution=None):
    return SimpleNamespace(
        runtime_supervisor=session.runtime_supervisor,
        execution=execution or _Execution(),
    )


def _governed(*, shadow_only=True, live_execution=False):
    return SimpleNamespace(
        shadow_only=shadow_only,
        live_execution=live_execution,
    )


def test_build_normal_stack_reuses_one_runtime_and_rpc_authority() -> None:
    rpc = lambda name, params: {}
    supervisor = SimpleNamespace(valid=True)

    stack = build_normal_shadow_execution_stack(
        rpc=rpc,
        runtime_supervisor=supervisor,
    )

    assert stack.runtime_supervisor is supervisor
    assert stack.governed.runtime_supervisor is supervisor
    assert stack.governed.risk_store is stack.risk_store
    assert stack.governed.atomic_store is stack.atomic_writeahead
    assert stack.dispatched.governed_supervisor is stack.governed
    assert stack.dispatched.outbox is stack.outbox
    assert stack.claimed.dispatched_supervisor is stack.dispatched
    assert stack.claimed.claims is stack.claims
    assert stack.execution.claimed_supervisor is stack.claimed
    assert stack.execution.starts is stack.starts
    assert stack.execution.checkpoints is stack.checkpoints
    assert stack.execution.kill_switch is stack.kill_switch

    for store in (
        stack.risk_store,
        stack.atomic_writeahead,
        stack.outbox,
        stack.claims,
        stack.kill_switch,
        stack.starts,
        stack.checkpoints,
    ):
        assert store._rpc is rpc


def test_ready_recovery_handoff_rechecks_phase86_then_processes_same_runtime() -> None:
    session = _session([_admission(blocked=False)])
    execution = _Execution()
    handoff = RecoveryGuardedShadowHandoff(
        session=session,
        recovery=_recovery(),
        normal=_normal(session, execution),
    )
    governed = _governed()

    receipt = handoff.process_governed_cycle(
        governed,
        worker_token="worker-99",
        claim_seconds=45,
        marks={"BTCUSDT": 100.0},
        observed_at=101.0,
        source_ref="phase99:test",
    )

    assert receipt.runtime_id == RUNTIME
    assert receipt.status == "NORMAL_SHADOW_PROCESSED"
    assert receipt.admission.blocked is False
    assert receipt.execution.outcome == "COMMITTED"
    assert receipt.shadow_only is True
    assert receipt.live_execution is False
    assert session.stack.admission.calls == 1
    assert len(execution.calls) == 1
    seen_governed, kwargs = execution.calls[0]
    assert seen_governed is governed
    assert kwargs == {
        "worker_token": "worker-99",
        "claim_seconds": 45,
        "marks": {"BTCUSDT": 100.0},
        "observed_at": 101.0,
        "source_ref": "phase99:test",
    }


def test_phase86_barrier_blocks_before_normal_execution_side_effect() -> None:
    session = _session([_admission(blocked=True)])
    execution = _Execution()
    handoff = RecoveryGuardedShadowHandoff(
        session=session,
        recovery=_recovery(),
        normal=_normal(session, execution),
    )

    with pytest.raises(
        RecoveryGuardedShadowBlockedError,
        match="Phase86 recovery barrier",
    ):
        handoff.process_governed_cycle(
            _governed(),
            worker_token="worker-99",
            claim_seconds=30,
            marks={},
            observed_at=101.0,
            source_ref="phase99:test",
        )

    assert execution.calls == []
    assert session.stack.admission.calls == 1


def test_phase97_not_ready_cannot_construct_handoff() -> None:
    session = _session()
    recovery = _recovery(
        ready=False,
        status="RECOVERY_BLOCKED",
        blocked=True,
    )

    with pytest.raises(
        RecoveryGuardedShadowBlockedError,
        match="did not authorize",
    ):
        RecoveryGuardedShadowHandoff(
            session=session,
            recovery=recovery,
            normal=_normal(session),
        )


def test_cross_runtime_or_cross_authority_handoff_is_rejected() -> None:
    session = _session()

    with pytest.raises(
        RecoveryGuardedShadowHandoffError,
        match="recovery runtime differs",
    ):
        RecoveryGuardedShadowHandoff(
            session=session,
            recovery=_recovery(runtime_id="other-runtime"),
            normal=_normal(session),
        )

    with pytest.raises(
        RecoveryGuardedShadowHandoffError,
        match="different runtime authority",
    ):
        RecoveryGuardedShadowHandoff(
            session=session,
            recovery=_recovery(),
            normal=SimpleNamespace(
                runtime_supervisor=SimpleNamespace(valid=True),
                execution=_Execution(),
            ),
        )


def test_closed_or_stale_session_fails_closed() -> None:
    closed = _session()
    closed.closed = True
    with pytest.raises(RecoveryGuardedShadowHandoffError, match="closed"):
        RecoveryGuardedShadowHandoff(
            session=closed,
            recovery=_recovery(),
            normal=_normal(closed),
        )

    stale = _session()
    stale.runtime_supervisor.valid = False
    with pytest.raises(PersistedRuntimeStaleError, match="valid Phase71"):
        RecoveryGuardedShadowHandoff(
            session=stale,
            recovery=_recovery(),
            normal=_normal(stale),
        )


def test_session_must_expose_phase86_reader() -> None:
    session = _session()
    session.stack = SimpleNamespace(admission=None)

    with pytest.raises(
        RecoveryGuardedShadowHandoffError,
        match="Phase86 admission reader",
    ):
        RecoveryGuardedShadowHandoff(
            session=session,
            recovery=_recovery(),
            normal=_normal(session),
        )


@pytest.mark.parametrize(
    ("governed", "match"),
    [
        (_governed(shadow_only=False), "not shadow-only"),
        (_governed(live_execution=True), "live-execution"),
    ],
)
def test_live_or_nonshadow_governed_cycle_is_rejected(governed, match) -> None:
    session = _session([_admission(blocked=False)])
    execution = _Execution()
    handoff = RecoveryGuardedShadowHandoff(
        session=session,
        recovery=_recovery(),
        normal=_normal(session, execution),
    )

    with pytest.raises(RecoveryGuardedShadowHandoffError, match=match):
        handoff.process_governed_cycle(
            governed,
            worker_token="worker-99",
            claim_seconds=30,
            marks={},
            observed_at=101.0,
            source_ref="phase99:test",
        )
    assert execution.calls == []
    assert session.stack.admission.calls == 0


@pytest.mark.parametrize(
    ("worker_token", "claim_seconds", "observed_at", "source_ref", "match"),
    [
        ("", 30, 101.0, "phase99:test", "worker_token"),
        ("worker", 9, 101.0, "phase99:test", "claim_seconds"),
        ("worker", 301, 101.0, "phase99:test", "claim_seconds"),
        ("worker", 30, float("nan"), "phase99:test", "observed_at"),
        ("worker", 30, 101.0, "", "source_ref"),
    ],
)
def test_invalid_normal_worker_inputs_fail_before_admission_or_execution(
    worker_token,
    claim_seconds,
    observed_at,
    source_ref,
    match,
) -> None:
    session = _session([_admission(blocked=False)])
    execution = _Execution()
    handoff = RecoveryGuardedShadowHandoff(
        session=session,
        recovery=_recovery(),
        normal=_normal(session, execution),
    )

    with pytest.raises((ValueError, RecoveryGuardedShadowHandoffError), match=match):
        handoff.process_governed_cycle(
            _governed(),
            worker_token=worker_token,
            claim_seconds=claim_seconds,
            marks={},
            observed_at=observed_at,
            source_ref=source_ref,
        )

    assert session.stack.admission.calls == 0
    assert execution.calls == []


def test_execution_failure_propagates_without_false_success_receipt() -> None:
    session = _session([_admission(blocked=False)])
    execution = _Execution(error=RuntimeError("phase86 transactional barrier race"))
    handoff = RecoveryGuardedShadowHandoff(
        session=session,
        recovery=_recovery(),
        normal=_normal(session, execution),
    )

    with pytest.raises(RuntimeError, match="barrier race"):
        handoff.process_governed_cycle(
            _governed(),
            worker_token="worker-99",
            claim_seconds=30,
            marks={},
            observed_at=101.0,
            source_ref="phase99:test",
        )

    assert session.stack.admission.calls == 1
    assert len(execution.calls) == 1


def test_from_recovery_assembles_normal_stack_from_same_session_authority() -> None:
    session = _session()

    handoff = RecoveryGuardedShadowHandoff.from_recovery(
        session=session,
        recovery=_recovery(),
    )

    assert handoff.runtime_id == RUNTIME
    assert handoff.normal.runtime_supervisor is session.runtime_supervisor
    assert handoff.admission is session.stack.admission
