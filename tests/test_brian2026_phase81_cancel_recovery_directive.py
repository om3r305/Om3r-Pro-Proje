from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase70_durable_runtime_store import RuntimeLease
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from brian2026.phase81_cancel_recovery_directive import (
    CancelRecoveryDirectiveError,
    CancelRecoveryDirectiveStore,
    CancelRecoveryLeg,
    PersistedRecoveryObligationSupervisor,
)


def _lease() -> RuntimeLease:
    return RuntimeLease(
        runtime_id="runtime-81",
        owner_token="owner-a",
        fencing_token=31,
        version=12,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )


def _leg():
    return {
        "asset_id": "BTCUSDT",
        "before_weight": 0.10,
        "current_weight": 0.25,
        "target_weight": 0.10,
        "reduce_weight": 0.15,
        "current_direction": 1,
        "order_direction": -1,
        "reduce_only": True,
    }


def _prepared_row(*, recovery_status="READY_REDUCE_ONLY", state="REDUCING"):
    legs = [] if recovery_status == "NO_RECOVERY_REQUIRED" else [_leg()]
    unsafe = (
        [{"asset_id": "ETHUSDT", "reason": "ROLLBACK_NOT_REDUCE_ONLY"}]
        if recovery_status == "MANUAL_REVIEW"
        else []
    )
    return {
        "prepared": True,
        "duplicate": False,
        "status": "PREPARED",
        "runtime_id": "runtime-81",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "runtime_version": 12,
        "fencing_token": 31,
        "cancel_risk_version": 8,
        "cancel_risk_receipt_id": "r" * 64,
        "cancel_reason": "REDUCING_NEW_RISK",
        "pre_state_id": "a" * 64,
        "current_state_id": "b" * 64,
        "current_risk_version": 9,
        "current_risk_receipt_id": "s" * 64,
        "current_risk_state": state,
        "recovery_status": recovery_status,
        "recovery_legs": legs,
        "unsafe_assets": unsafe,
    }


class FakeRpc:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, name, params):
        self.calls.append((name, dict(params)))
        value = self.response
        return value(dict(params)) if callable(value) else dict(value)


def test_recovery_leg_can_only_move_toward_authoritative_precycle_exposure() -> None:
    leg = CancelRecoveryLeg(**_leg())
    assert leg.reduce_only is True
    assert leg.target_weight == 0.10

    bad = dict(_leg())
    bad["target_weight"] = 0.30
    bad["reduce_weight"] = 0.05
    with pytest.raises(ValueError, match="pre-cycle"):
        CancelRecoveryLeg(**bad)


def test_store_prepares_exact_reduce_only_recovery_contract() -> None:
    rpc = FakeRpc(_prepared_row())
    receipt = CancelRecoveryDirectiveStore(rpc).prepare(
        _lease(),
        cycle_id="c" * 64,
        expected_runtime_version=12,
    )
    assert receipt.prepared is True
    assert receipt.recovery_status == "READY_REDUCE_ONLY"
    assert len(receipt.recovery_legs) == 1
    assert receipt.recovery_legs[0].reduce_weight == pytest.approx(0.15)

    assert rpc.calls[0][0] == "brian_prepare_shadow_cancel_recovery"
    assert rpc.calls[0][1] == {
        "p_runtime_id": "runtime-81",
        "p_owner_token": "owner-a",
        "p_fencing_token": 31,
        "p_cycle_id": "c" * 64,
        "p_expected_runtime_version": 12,
    }


def test_halted_recovery_is_wait_only_not_executable_reduce_only() -> None:
    row = _prepared_row(
        recovery_status="WAIT_RISK_RELEASE",
        state="HALTED",
    )
    receipt = CancelRecoveryDirectiveStore(FakeRpc(row)).prepare(
        _lease(),
        cycle_id="c" * 64,
        expected_runtime_version=12,
    )
    assert receipt.recovery_status == "WAIT_RISK_RELEASE"
    assert receipt.current_risk_state == "HALTED"
    assert receipt.recovery_legs


def test_store_rejects_ready_recovery_while_halted() -> None:
    with pytest.raises(ValueError, match="READY_REDUCE_ONLY"):
        CancelRecoveryDirectiveStore(
            FakeRpc(_prepared_row(state="HALTED"))
        ).prepare(
            _lease(),
            cycle_id="c" * 64,
            expected_runtime_version=12,
        )


def test_manual_review_requires_unsafe_evidence() -> None:
    row = _prepared_row(recovery_status="MANUAL_REVIEW")
    receipt = CancelRecoveryDirectiveStore(FakeRpc(row)).prepare(
        _lease(),
        cycle_id="c" * 64,
        expected_runtime_version=12,
    )
    assert receipt.recovery_status == "MANUAL_REVIEW"
    assert receipt.unsafe_assets[0]["asset_id"] == "ETHUSDT"


def test_no_cancel_request_is_a_nonprepared_normal_result() -> None:
    row = {
        "prepared": False,
        "duplicate": False,
        "status": "NO_CANCEL_REQUEST",
        "runtime_id": "runtime-81",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "runtime_version": 12,
        "fencing_token": 31,
    }
    receipt = CancelRecoveryDirectiveStore(FakeRpc(row)).prepare(
        _lease(),
        cycle_id="c" * 64,
        expected_runtime_version=12,
    )
    assert receipt.prepared is False
    assert receipt.status == "NO_CANCEL_REQUEST"


def test_prepared_response_cannot_drift_from_authoritative_runtime_version() -> None:
    row = _prepared_row()
    row["runtime_version"] = 13
    with pytest.raises(
        CancelRecoveryDirectiveError,
        match="runtime version",
    ):
        CancelRecoveryDirectiveStore(FakeRpc(row)).prepare(
            _lease(),
            cycle_id="c" * 64,
            expected_runtime_version=12,
        )


class _Checkpoint:
    checkpoint_id = "z" * 64


class _Runtime:
    def checkpoint(self):
        return _Checkpoint()


class _RuntimeSupervisor:
    def __init__(self):
        self.lease = _lease()
        self.persisted_version = 12
        self._valid = True
        self.runtime = _Runtime()

    @property
    def valid(self):
        return self._valid


class _ExecutionSupervisor:
    def __init__(self, step):
        self.step = step
        runtime = _RuntimeSupervisor()
        self.claimed_supervisor = SimpleNamespace(
            dispatched_supervisor=SimpleNamespace(
                governed_supervisor=SimpleNamespace(
                    runtime_supervisor=runtime
                )
            )
        )

    def process_governed_cycle(self, *args, **kwargs):
        del args, kwargs
        return self.step


class _RecoveryStore:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = 0

    def prepare(self, lease, *, cycle_id, expected_runtime_version):
        assert lease.runtime_id == "runtime-81"
        assert cycle_id == "c" * 64
        assert expected_runtime_version == 12
        self.calls += 1
        return self.receipt


def _execution_step(*, committed=True, started=True, outcome="COMMITTED"):
    return SimpleNamespace(
        start=SimpleNamespace(started=started) if started else None,
        checkpoint_commit=(
            SimpleNamespace(committed=committed)
            if committed
            else None
        ),
        claim=SimpleNamespace(cycle_id="c" * 64),
        outcome=outcome,
        persisted_version=12,
        checkpoint_id="z" * 64,
    )


def _receipt_from_row(row):
    return CancelRecoveryDirectiveStore(FakeRpc(row)).prepare(
        _lease(),
        cycle_id="c" * 64,
        expected_runtime_version=12,
    )


def test_wrapper_marks_ready_recovery_without_rewriting_committed_history() -> None:
    execution = _ExecutionSupervisor(_execution_step())
    recovery = _RecoveryStore(_receipt_from_row(_prepared_row()))
    wrapper = PersistedRecoveryObligationSupervisor(
        execution_supervisor=execution,
        recovery_store=recovery,
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase81-ready",
    )
    assert step.outcome == "COMMITTED_RECOVERY_READY"
    assert recovery.calls == 1
    assert execution.claimed_supervisor.dispatched_supervisor.governed_supervisor.runtime_supervisor.valid is True


def test_wrapper_reports_wait_when_original_cycle_not_yet_committed() -> None:
    row = {
        "prepared": False,
        "duplicate": False,
        "status": "WAIT_ORIGINAL_COMMIT",
        "runtime_id": "runtime-81",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "runtime_version": 12,
        "fencing_token": 31,
        "cancel_risk_version": 8,
        "cancel_risk_receipt_id": "r" * 64,
        "cancel_reason": "HALTED",
        "journal_stage": "PAPER_APPLIED",
    }
    execution = _ExecutionSupervisor(
        _execution_step(outcome="RECONCILIATION_BLOCKED_CANCEL_REQUESTED")
    )
    recovery = _RecoveryStore(_receipt_from_row(row))
    wrapper = PersistedRecoveryObligationSupervisor(
        execution_supervisor=execution,
        recovery_store=recovery,
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase81-wait",
    )
    assert step.outcome.endswith("RECOVERY_WAIT_ORIGINAL_COMMIT")


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("LEASE_LOST", PersistedRuntimeLeaseError),
        ("RUNTIME_VERSION_CONFLICT", PersistedRuntimeStaleError),
        ("HEAD_MOVED", PersistedRuntimeStaleError),
        ("RISK_STATE_UNAVAILABLE", PersistedRuntimeStaleError),
        ("EVIDENCE_INVALID", PersistedRuntimeStaleError),
    ],
)
def test_wrapper_fails_closed_on_recovery_authority_errors(status, error) -> None:
    row = {
        "prepared": False,
        "duplicate": False,
        "status": status,
        "runtime_id": "runtime-81",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "runtime_version": 12,
        "fencing_token": 31,
    }
    execution = _ExecutionSupervisor(_execution_step())
    wrapper = PersistedRecoveryObligationSupervisor(
        execution_supervisor=execution,
        recovery_store=_RecoveryStore(_receipt_from_row(row)),
    )
    with pytest.raises(error):
        wrapper.process_governed_cycle(
            object(),
            worker_token="worker-a",
            claim_seconds=30,
            marks={},
            observed_at=1.0,
            source_ref="phase81-fail",
        )
    runtime = (
        execution.claimed_supervisor
        .dispatched_supervisor
        .governed_supervisor
        .runtime_supervisor
    )
    assert runtime.valid is False


def test_wrapper_does_not_probe_recovery_before_authoritative_checkpoint() -> None:
    execution = _ExecutionSupervisor(
        _execution_step(committed=False, started=True, outcome="CLAIM_LOST")
    )
    recovery = _RecoveryStore(_receipt_from_row(_prepared_row()))
    wrapper = PersistedRecoveryObligationSupervisor(
        execution_supervisor=execution,
        recovery_store=recovery,
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase81-no-probe",
    )
    assert step.recovery is None
    assert step.outcome == "CLAIM_LOST"
    assert recovery.calls == 0


class _SequentialRecoveryStore:
    def __init__(self, receipts):
        self.receipts = list(receipts)
        self.calls = []

    def prepare(self, lease, *, cycle_id, expected_runtime_version):
        assert lease.runtime_id == "runtime-81"
        assert cycle_id == "c" * 64
        self.calls.append(expected_runtime_version)
        if not self.receipts:
            raise AssertionError("unexpected recovery prepare retry")
        return self.receipts.pop(0)


class _Phase75AbortHarness:
    def __init__(self, runtime):
        self.runtime_supervisor = runtime
        self.aborts = []

    def abort_authorized_cycle(self, *, cycle_id, reason):
        self.aborts.append((cycle_id, reason))
        self.runtime_supervisor.persisted_version += 1
        return SimpleNamespace(committed=True)


class _ForeignExecutionSupervisor:
    def __init__(self, step):
        self.step = step
        runtime = _RuntimeSupervisor()
        phase75 = _Phase75AbortHarness(runtime)
        self.phase75 = phase75
        self.claimed_supervisor = SimpleNamespace(
            dispatched_supervisor=SimpleNamespace(
                governed_supervisor=phase75
            )
        )

    def process_governed_cycle(self, *args, **kwargs):
        del args, kwargs
        return self.step


def test_foreign_prepaper_cycle_is_aborted_before_phase81_recovery_is_frozen() -> None:
    foreign_row = {
        "prepared": False,
        "duplicate": False,
        "status": "FOREIGN_CYCLE_ACTIVE",
        "runtime_id": "runtime-81",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "runtime_version": 12,
        "fencing_token": 31,
        "cancel_risk_version": 8,
        "cancel_risk_receipt_id": "r" * 64,
        "cancel_reason": "HALTED",
        "foreign_cycle_id": "f" * 64,
        "foreign_cycle_stage": "CYCLE_CREATED",
    }
    prepared_row = _prepared_row()
    prepared_row["runtime_version"] = 13

    execution = _ForeignExecutionSupervisor(_execution_step())
    recovery = _SequentialRecoveryStore([
        _receipt_from_row(foreign_row),
        CancelRecoveryDirectiveStore(FakeRpc(prepared_row)).prepare(
            RuntimeLease(
                runtime_id="runtime-81",
                owner_token="owner-a",
                fencing_token=31,
                version=13,
                status="ACQUIRED",
                acquired=True,
                lease_until=None,
            ),
            cycle_id="c" * 64,
            expected_runtime_version=13,
        ),
    ])
    wrapper = PersistedRecoveryObligationSupervisor(
        execution_supervisor=execution,
        recovery_store=recovery,
    )

    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase86-foreign-quarantine",
    )

    assert execution.phase75.aborts == [
        ("f" * 64, "phase86:recovery_admission_interlock")
    ]
    assert recovery.calls == [12, 13]
    assert step.recovery is not None and step.recovery.prepared is True
    assert step.outcome.endswith("RECOVERY_READY_FOREIGN_CYCLE_ABORTED")


def test_side_effected_foreign_cycle_is_never_auto_aborted() -> None:
    foreign_row = {
        "prepared": False,
        "duplicate": False,
        "status": "FOREIGN_CYCLE_ACTIVE",
        "runtime_id": "runtime-81",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "runtime_version": 12,
        "fencing_token": 31,
        "cancel_risk_version": 8,
        "cancel_risk_receipt_id": "r" * 64,
        "cancel_reason": "HALTED",
        "foreign_cycle_id": "f" * 64,
        "foreign_cycle_stage": "PAPER_APPLIED",
    }
    execution = _ForeignExecutionSupervisor(_execution_step())
    recovery = _SequentialRecoveryStore([_receipt_from_row(foreign_row)])
    wrapper = PersistedRecoveryObligationSupervisor(
        execution_supervisor=execution,
        recovery_store=recovery,
    )

    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase86-foreign-wait",
    )

    assert execution.phase75.aborts == []
    assert recovery.calls == [12]
    assert step.recovery is not None
    assert step.recovery.status == "FOREIGN_CYCLE_ACTIVE"
    assert step.outcome.endswith("RECOVERY_WAIT_FOREIGN_CYCLE")
