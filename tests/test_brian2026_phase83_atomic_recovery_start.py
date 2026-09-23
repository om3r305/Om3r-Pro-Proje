from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase70_durable_runtime_store import RuntimeLease
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from brian2026.phase81_cancel_recovery_directive import CancelRecoveryLeg
from brian2026.phase82_recovery_claim_fencing import (
    RecoveryClaimError,
    RecoveryClaimReceipt,
)
from brian2026.phase83_atomic_recovery_start import (
    AtomicRecoveryStartError,
    AtomicRecoveryStartReceipt,
    AtomicRecoveryStartStore,
    PersistedAtomicRecoveryStartSupervisor,
)


def _lease() -> RuntimeLease:
    return RuntimeLease(
        runtime_id="runtime-83",
        owner_token="owner-a",
        fencing_token=51,
        version=18,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )


def _leg() -> CancelRecoveryLeg:
    return CancelRecoveryLeg(
        asset_id="BTCUSDT",
        before_weight=0.10,
        current_weight=0.25,
        target_weight=0.10,
        reduce_weight=0.15,
        current_direction=1,
        order_direction=-1,
        reduce_only=True,
    )


def _claim() -> RecoveryClaimReceipt:
    return RecoveryClaimReceipt(
        runtime_id="runtime-83",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        cancel_risk_receipt_id="r" * 64,
        runtime_version=18,
        head_state_id="h" * 64,
        fencing_token=51,
        claim_fencing_token=4,
        status="CLAIMED",
        claimed=True,
        terminal=False,
        worker_token="recovery-a",
        claim_until="2026-09-23T19:00:00Z",
        risk_version=12,
        risk_receipt_id="s" * 64,
        risk_state="REDUCING",
        recovery_status="READY_REDUCE_ONLY",
        recovery_legs=(_leg(),),
    )


def _row(*, status="STARTED", started=True, duplicate=False, resume=False, state="REDUCING"):
    return {
        "started": started,
        "duplicate": duplicate,
        "resume_only": resume,
        "status": status,
        "runtime_id": "runtime-83",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "cancel_risk_receipt_id": "r" * 64,
        "runtime_version": 18,
        "head_state_id": "h" * 64,
        "fencing_token": 51,
        "recovery_claim_fencing_token": 4,
        "risk_version": 13 if state else None,
        "risk_receipt_id": ("t" * 64) if state else None,
        "risk_state": state,
        "recovery_legs": [_leg().to_dict()] if started else [],
    }


class FakeRpc:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, name, params):
        self.calls.append((name, dict(params)))
        value = self.response
        return value(dict(params)) if callable(value) else dict(value)


def test_store_marks_exact_claim_and_recovery_legs_started() -> None:
    rpc = FakeRpc(_row())
    receipt = AtomicRecoveryStartStore(rpc).mark_started(
        _lease(),
        _claim(),
        worker_token="recovery-a",
    )
    assert receipt.started is True
    assert receipt.status == "STARTED"
    assert receipt.recovery_legs == (_leg(),)
    assert rpc.calls[0][0] == "brian_mark_shadow_recovery_started"
    assert rpc.calls[0][1]["p_recovery_claim_fencing_token"] == 4


def test_store_accepts_exact_lost_response_duplicate() -> None:
    receipt = AtomicRecoveryStartStore(
        FakeRpc(_row(
            status="STARTED_ALREADY",
            duplicate=True,
        ))
    ).mark_started(
        _lease(),
        _claim(),
        worker_token="recovery-a",
    )
    assert receipt.started is True
    assert receipt.duplicate is True
    assert receipt.resume_only is False


def test_store_accepts_takeover_resume_boundary() -> None:
    receipt = AtomicRecoveryStartStore(
        FakeRpc(_row(
            status="STARTED_RESUME",
            duplicate=True,
            resume=True,
        ))
    ).mark_started(
        _lease(),
        _claim(),
        worker_token="recovery-a",
    )
    assert receipt.started is True
    assert receipt.resume_only is True


def test_store_rejects_started_leg_drift() -> None:
    row = _row()
    row["recovery_legs"][0]["target_weight"] = 0.05
    row["recovery_legs"][0]["reduce_weight"] = 0.20
    with pytest.raises(AtomicRecoveryStartError, match="legs differ"):
        AtomicRecoveryStartStore(FakeRpc(row)).mark_started(
            _lease(),
            _claim(),
            worker_token="recovery-a",
        )


def test_halted_risk_cannot_cross_recovery_started() -> None:
    row = _row(
        status="WAIT_RISK_RELEASE",
        started=False,
        state="HALTED",
    )
    row["recovery_legs"] = []
    receipt = AtomicRecoveryStartStore(FakeRpc(row)).mark_started(
        _lease(),
        _claim(),
        worker_token="recovery-a",
    )
    assert receipt.started is False
    assert receipt.status == "WAIT_RISK_RELEASE"
    assert receipt.risk_state == "HALTED"


def test_store_rejects_claim_worker_mismatch_before_rpc() -> None:
    rpc = FakeRpc(_row())
    with pytest.raises(AtomicRecoveryStartError, match="worker"):
        AtomicRecoveryStartStore(rpc).mark_started(
            _lease(),
            _claim(),
            worker_token="recovery-b",
        )
    assert rpc.calls == []


class _Checkpoint:
    checkpoint_id = "z" * 64


class _Runtime:
    def checkpoint(self):
        return _Checkpoint()


class _RuntimeSupervisor:
    def __init__(self):
        self.lease = _lease()
        self.persisted_version = 18
        self._valid = True
        self.runtime = _Runtime()

    @property
    def valid(self):
        return self._valid


class _ClaimSupervisor:
    def __init__(self, claim):
        self.claim = claim
        runtime = _RuntimeSupervisor()
        self.recovery_supervisor = SimpleNamespace(
            execution_supervisor=SimpleNamespace(
                claimed_supervisor=SimpleNamespace(
                    dispatched_supervisor=SimpleNamespace(
                        governed_supervisor=SimpleNamespace(
                            runtime_supervisor=runtime
                        )
                    )
                )
            )
        )

    def process_governed_cycle(self, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            claim=self.claim,
            outcome="COMMITTED_RECOVERY_READY_RECOVERY_CLAIMED",
            persisted_version=18,
            checkpoint_id="z" * 64,
        )


class _Starts:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = 0

    def mark_started(self, lease, claim, *, worker_token):
        assert lease.runtime_id == "runtime-83"
        assert claim.cycle_id == "c" * 64
        assert worker_token == "recovery-a"
        self.calls += 1
        return self.receipt


def _start_receipt(status="STARTED", *, started=True, duplicate=False, resume=False):
    return AtomicRecoveryStartReceipt(
        runtime_id="runtime-83",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        cancel_risk_receipt_id="r" * 64,
        runtime_version=18,
        head_state_id="h" * 64,
        fencing_token=51,
        recovery_claim_fencing_token=4,
        status=status,
        started=started,
        duplicate=duplicate,
        resume_only=resume,
        risk_version=13 if status not in {"LEASE_LOST","HEAD_MOVED","CLAIM_LOST","DIRECTIVE_MISSING","RISK_STATE_UNAVAILABLE","EVIDENCE_INVALID"} else None,
        risk_receipt_id=("t" * 64) if status not in {"LEASE_LOST","HEAD_MOVED","CLAIM_LOST","DIRECTIVE_MISSING","RISK_STATE_UNAVAILABLE","EVIDENCE_INVALID"} else None,
        risk_state=("HALTED" if status == "WAIT_RISK_RELEASE" else "REDUCING") if status not in {"LEASE_LOST","HEAD_MOVED","CLAIM_LOST","DIRECTIVE_MISSING","RISK_STATE_UNAVAILABLE","EVIDENCE_INVALID"} else None,
        recovery_legs=(_leg(),) if started else (),
    )


def test_wrapper_crosses_started_only_after_phase82_claim() -> None:
    claims = _ClaimSupervisor(_claim())
    starts = _Starts(_start_receipt())
    wrapper = PersistedAtomicRecoveryStartSupervisor(
        claim_supervisor=claims,
        starts=starts,
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        recovery_worker_token="recovery-a",
        recovery_claim_seconds=45,
        marks={},
        observed_at=1.0,
        source_ref="phase83",
    )
    assert step.start is not None and step.start.started is True
    assert step.outcome.endswith("RECOVERY_STARTED")
    assert starts.calls == 1


def test_wrapper_does_not_start_when_phase82_has_no_claim() -> None:
    claims = _ClaimSupervisor(None)
    starts = _Starts(_start_receipt())
    wrapper = PersistedAtomicRecoveryStartSupervisor(
        claim_supervisor=claims,
        starts=starts,
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        recovery_worker_token="recovery-a",
        recovery_claim_seconds=45,
        marks={},
        observed_at=1.0,
        source_ref="phase83-none",
    )
    assert step.start is None
    assert starts.calls == 0


def test_wrapper_keeps_halted_start_wait_nonterminal() -> None:
    claims = _ClaimSupervisor(_claim())
    start = AtomicRecoveryStartReceipt(
        runtime_id="runtime-83",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        cancel_risk_receipt_id="r" * 64,
        runtime_version=18,
        head_state_id="h" * 64,
        fencing_token=51,
        recovery_claim_fencing_token=4,
        status="WAIT_RISK_RELEASE",
        started=False,
        duplicate=False,
        resume_only=False,
        risk_version=13,
        risk_receipt_id="t" * 64,
        risk_state="HALTED",
        recovery_legs=(),
    )
    wrapper = PersistedAtomicRecoveryStartSupervisor(
        claim_supervisor=claims,
        starts=_Starts(start),
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        recovery_worker_token="recovery-a",
        recovery_claim_seconds=45,
        marks={},
        observed_at=1.0,
        source_ref="phase83-wait",
    )
    assert step.outcome.endswith("START_WAIT_RISK_RELEASE")


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("LEASE_LOST", PersistedRuntimeLeaseError),
        ("HEAD_MOVED", PersistedRuntimeStaleError),
        ("DIRECTIVE_MISSING", PersistedRuntimeStaleError),
        ("RISK_STATE_UNAVAILABLE", PersistedRuntimeStaleError),
        ("EVIDENCE_INVALID", PersistedRuntimeStaleError),
    ],
)
def test_wrapper_fails_closed_on_recovery_start_authority_errors(status, error) -> None:
    claims = _ClaimSupervisor(_claim())
    start = _start_receipt(status, started=False)
    wrapper = PersistedAtomicRecoveryStartSupervisor(
        claim_supervisor=claims,
        starts=_Starts(start),
    )
    with pytest.raises(error):
        wrapper.process_governed_cycle(
            object(),
            worker_token="worker-a",
            claim_seconds=30,
            recovery_worker_token="recovery-a",
            recovery_claim_seconds=45,
            marks={},
            observed_at=1.0,
            source_ref="phase83-fail",
        )
    runtime = (
        claims.recovery_supervisor.execution_supervisor.claimed_supervisor
        .dispatched_supervisor.governed_supervisor.runtime_supervisor
    )
    assert runtime.valid is False


def test_claim_loss_at_start_requires_reclaim_without_forging_runtime_stale() -> None:
    claims = _ClaimSupervisor(_claim())
    start = _start_receipt("CLAIM_LOST", started=False)
    wrapper = PersistedAtomicRecoveryStartSupervisor(
        claim_supervisor=claims,
        starts=_Starts(start),
    )
    with pytest.raises(RecoveryClaimError, match="fresh recovery claim"):
        wrapper.process_governed_cycle(
            object(),
            worker_token="worker-a",
            claim_seconds=30,
            recovery_worker_token="recovery-a",
            recovery_claim_seconds=45,
            marks={},
            observed_at=1.0,
            source_ref="phase83-claim-lost",
        )
    runtime = (
        claims.recovery_supervisor.execution_supervisor.claimed_supervisor
        .dispatched_supervisor.governed_supervisor.runtime_supervisor
    )
    assert runtime.valid is True
