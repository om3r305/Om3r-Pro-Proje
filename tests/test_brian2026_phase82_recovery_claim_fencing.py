from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase70_durable_runtime_store import RuntimeLease
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from brian2026.phase81_cancel_recovery_directive import (
    CancelRecoveryDirectiveReceipt,
    CancelRecoveryLeg,
)
from brian2026.phase82_recovery_claim_fencing import (
    PersistedRecoveryClaimSupervisor,
    RecoveryClaimError,
    RecoveryClaimReceipt,
    RecoveryClaimStore,
)


def _lease() -> RuntimeLease:
    return RuntimeLease(
        runtime_id="runtime-82",
        owner_token="owner-a",
        fencing_token=41,
        version=15,
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


def _claim_row(*, status="CLAIMED", claimed=True, terminal=False, state="REDUCING"):
    return {
        "claimed": claimed,
        "terminal": terminal,
        "status": status,
        "runtime_id": "runtime-82",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "cancel_risk_receipt_id": "r" * 64,
        "runtime_version": 15,
        "head_state_id": "h" * 64,
        "fencing_token": 41,
        "claim_fencing_token": 3 if status not in {"WAIT_RISK_RELEASE", "HEAD_MOVED"} else None,
        "claim_until": "2026-09-23T18:00:00Z" if claimed else None,
        "worker_token": "recovery-a" if claimed else None,
        "risk_version": 11,
        "risk_receipt_id": "s" * 64,
        "risk_state": state,
        "recovery_status": "READY_REDUCE_ONLY",
        "recovery_legs": [_leg().to_dict()] if claimed else [],
    }


class FakeRpc:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.calls = []

    def __call__(self, name, params):
        self.calls.append((name, dict(params)))
        value = self.responses[name]
        return value(dict(params)) if callable(value) else dict(value)


def test_claim_store_submits_exact_runtime_and_worker_fences() -> None:
    rpc = FakeRpc({
        "brian_claim_shadow_cancel_recovery": _claim_row(),
    })
    receipt = RecoveryClaimStore(rpc).claim(
        _lease(),
        cycle_id="c" * 64,
        worker_token="recovery-a",
        claim_seconds=30,
    )
    assert receipt.claimed is True
    assert receipt.status == "CLAIMED"
    assert receipt.claim_fencing_token == 3
    assert receipt.recovery_legs == (_leg(),)

    assert rpc.calls[0][1] == {
        "p_runtime_id": "runtime-82",
        "p_owner_token": "owner-a",
        "p_fencing_token": 41,
        "p_cycle_id": "c" * 64,
        "p_worker_token": "recovery-a",
        "p_claim_seconds": 30,
    }


def test_halted_recovery_remains_unclaimed_wait_state() -> None:
    row = _claim_row(
        status="WAIT_RISK_RELEASE",
        claimed=False,
        terminal=False,
        state="HALTED",
    )
    row["dispatch_id"] = "d" * 64
    row["cancel_risk_receipt_id"] = "r" * 64
    receipt = RecoveryClaimStore(
        FakeRpc({"brian_claim_shadow_cancel_recovery": row})
    ).claim(
        _lease(),
        cycle_id="c" * 64,
        worker_token="recovery-a",
        claim_seconds=30,
    )
    assert receipt.claimed is False
    assert receipt.status == "WAIT_RISK_RELEASE"
    assert receipt.risk_state == "HALTED"


def test_claimed_recovery_cannot_be_returned_under_halted_risk() -> None:
    row = _claim_row(state="HALTED")
    with pytest.raises(ValueError, match="HALTED"):
        RecoveryClaimStore(
            FakeRpc({"brian_claim_shadow_cancel_recovery": row})
        ).claim(
            _lease(),
            cycle_id="c" * 64,
            worker_token="recovery-a",
            claim_seconds=30,
        )


def test_claim_store_rejects_worker_anchor_drift() -> None:
    row = _claim_row()
    row["worker_token"] = "other-worker"
    with pytest.raises(RecoveryClaimError, match="worker"):
        RecoveryClaimStore(
            FakeRpc({"brian_claim_shadow_cancel_recovery": row})
        ).claim(
            _lease(),
            cycle_id="c" * 64,
            worker_token="recovery-a",
            claim_seconds=30,
        )


def test_renewal_uses_owned_claim_fence_and_accepts_current_nonhalted_risk() -> None:
    claim = RecoveryClaimReceipt(
        runtime_id="runtime-82",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        cancel_risk_receipt_id="r" * 64,
        runtime_version=15,
        head_state_id="h" * 64,
        fencing_token=41,
        claim_fencing_token=3,
        status="CLAIMED",
        claimed=True,
        terminal=False,
        worker_token="recovery-a",
        claim_until="2026-09-23T18:00:00Z",
        risk_version=11,
        risk_receipt_id="s" * 64,
        risk_state="REDUCING",
        recovery_status="READY_REDUCE_ONLY",
        recovery_legs=(_leg(),),
    )
    rpc = FakeRpc({
        "brian_renew_shadow_cancel_recovery_claim": {
            "renewed": True,
            "status": "RENEWED",
            "runtime_id": "runtime-82",
            "dispatch_id": "d" * 64,
            "cycle_id": "c" * 64,
            "runtime_version": 15,
            "head_state_id": "h" * 64,
            "fencing_token": 41,
            "claim_fencing_token": 3,
            "claim_until": "2026-09-23T18:01:00Z",
            "risk_version": 12,
            "risk_receipt_id": "t" * 64,
            "risk_state": "ACTIVE",
        }
    })
    renewed = RecoveryClaimStore(rpc).renew(
        _lease(),
        claim,
        worker_token="recovery-a",
        claim_seconds=60,
    )
    assert renewed.renewed is True
    assert renewed.status == "RENEWED"
    assert renewed.risk_version == 12
    assert rpc.calls[0][1]["p_claim_fencing_token"] == 3


def test_renewal_is_not_extended_when_risk_becomes_halted() -> None:
    claim = RecoveryClaimReceipt(
        runtime_id="runtime-82",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        cancel_risk_receipt_id="r" * 64,
        runtime_version=15,
        head_state_id="h" * 64,
        fencing_token=41,
        claim_fencing_token=3,
        status="CLAIMED",
        claimed=True,
        terminal=False,
        worker_token="recovery-a",
        risk_version=11,
        risk_receipt_id="s" * 64,
        risk_state="REDUCING",
        recovery_status="READY_REDUCE_ONLY",
        recovery_legs=(_leg(),),
    )
    rpc = FakeRpc({
        "brian_renew_shadow_cancel_recovery_claim": {
            "renewed": False,
            "status": "RENEWAL_BLOCKED_RISK",
            "runtime_id": "runtime-82",
            "dispatch_id": "d" * 64,
            "cycle_id": "c" * 64,
            "runtime_version": 15,
            "head_state_id": "h" * 64,
            "fencing_token": 41,
            "claim_fencing_token": 3,
            "risk_version": 12,
            "risk_receipt_id": "t" * 64,
            "risk_state": "HALTED",
        }
    })
    renewed = RecoveryClaimStore(rpc).renew(
        _lease(),
        claim,
        worker_token="recovery-a",
        claim_seconds=60,
    )
    assert renewed.renewed is False
    assert renewed.status == "RENEWAL_BLOCKED_RISK"
    assert renewed.risk_state == "HALTED"


def _directive(status="READY_REDUCE_ONLY"):
    return CancelRecoveryDirectiveReceipt(
        runtime_id="runtime-82",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=15,
        fencing_token=41,
        status="PREPARED",
        prepared=True,
        duplicate=False,
        cancel_risk_version=8,
        cancel_risk_receipt_id="r" * 64,
        cancel_reason="REDUCING_NEW_RISK",
        pre_state_id="a" * 64,
        current_state_id="h" * 64,
        current_risk_version=11,
        current_risk_receipt_id="s" * 64,
        current_risk_state="REDUCING" if status != "WAIT_RISK_RELEASE" else "HALTED",
        recovery_status=status,
        recovery_legs=(_leg(),),
        unsafe_assets=(),
    )


class _Checkpoint:
    checkpoint_id = "z" * 64


class _Runtime:
    def checkpoint(self):
        return _Checkpoint()


class _RuntimeSupervisor:
    def __init__(self):
        self.lease = _lease()
        self.persisted_version = 15
        self._valid = True
        self.runtime = _Runtime()

    @property
    def valid(self):
        return self._valid


class _RecoverySupervisor:
    def __init__(self, recovery):
        runtime = _RuntimeSupervisor()
        self.execution_supervisor = SimpleNamespace(
            claimed_supervisor=SimpleNamespace(
                dispatched_supervisor=SimpleNamespace(
                    governed_supervisor=SimpleNamespace(
                        runtime_supervisor=runtime
                    )
                )
            )
        )
        self.recovery = recovery

    def process_governed_cycle(self, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            recovery=self.recovery,
            outcome="COMMITTED_RECOVERY_READY",
            persisted_version=15,
            checkpoint_id="z" * 64,
        )


class _ClaimStore:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = 0

    def claim(self, lease, *, cycle_id, worker_token, claim_seconds):
        assert lease.runtime_id == "runtime-82"
        assert cycle_id == "c" * 64
        assert worker_token == "recovery-a"
        assert claim_seconds == 45
        self.calls += 1
        return self.receipt


def _claim_receipt(status="CLAIMED", *, claimed=True):
    return RecoveryClaimReceipt(
        runtime_id="runtime-82",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        cancel_risk_receipt_id="r" * 64,
        runtime_version=15,
        head_state_id="h" * 64,
        fencing_token=41,
        claim_fencing_token=3 if claimed else None,
        status=status,
        claimed=claimed,
        terminal=False,
        worker_token="recovery-a" if claimed else None,
        risk_version=12,
        risk_receipt_id="t" * 64,
        risk_state="REDUCING" if claimed else "HALTED",
        recovery_status="READY_REDUCE_ONLY",
        recovery_legs=(_leg(),) if claimed else (),
    )


def test_wrapper_claims_ready_recovery_after_phase81_obligation() -> None:
    recovery_supervisor = _RecoverySupervisor(_directive())
    claims = _ClaimStore(_claim_receipt())
    wrapper = PersistedRecoveryClaimSupervisor(
        recovery_supervisor=recovery_supervisor,
        claims=claims,
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        recovery_worker_token="recovery-a",
        recovery_claim_seconds=45,
        marks={},
        observed_at=1.0,
        source_ref="phase82",
    )
    assert step.claim is not None and step.claim.claimed is True
    assert step.outcome.endswith("RECOVERY_CLAIMED")
    assert claims.calls == 1


def test_wrapper_keeps_halted_recovery_waiting_without_invalidating_runtime() -> None:
    recovery_supervisor = _RecoverySupervisor(_directive("WAIT_RISK_RELEASE"))
    claims = _ClaimStore(
        RecoveryClaimReceipt(
            runtime_id="runtime-82",
            cycle_id="c" * 64,
            dispatch_id="d" * 64,
            cancel_risk_receipt_id="r" * 64,
            runtime_version=15,
            head_state_id="h" * 64,
            fencing_token=41,
            claim_fencing_token=None,
            status="WAIT_RISK_RELEASE",
            claimed=False,
            terminal=False,
            risk_version=12,
            risk_receipt_id="t" * 64,
            risk_state="HALTED",
        )
    )
    wrapper = PersistedRecoveryClaimSupervisor(
        recovery_supervisor=recovery_supervisor,
        claims=claims,
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        recovery_worker_token="recovery-a",
        recovery_claim_seconds=45,
        marks={},
        observed_at=1.0,
        source_ref="phase82-wait",
    )
    assert step.outcome.endswith("CLAIM_WAIT_RISK_RELEASE")
    runtime = (
        recovery_supervisor.execution_supervisor
        .claimed_supervisor.dispatched_supervisor
        .governed_supervisor.runtime_supervisor
    )
    assert runtime.valid is True


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
def test_wrapper_fails_closed_on_recovery_claim_authority_errors(status, error) -> None:
    recovery_supervisor = _RecoverySupervisor(_directive())
    claim = RecoveryClaimReceipt(
        runtime_id="runtime-82",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        cancel_risk_receipt_id="r" * 64,
        runtime_version=15,
        head_state_id="h" * 64,
        fencing_token=41,
        claim_fencing_token=None,
        status=status,
        claimed=False,
        terminal=False,
    )
    wrapper = PersistedRecoveryClaimSupervisor(
        recovery_supervisor=recovery_supervisor,
        claims=_ClaimStore(claim),
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
            source_ref="phase82-fail",
        )
    runtime = (
        recovery_supervisor.execution_supervisor
        .claimed_supervisor.dispatched_supervisor
        .governed_supervisor.runtime_supervisor
    )
    assert runtime.valid is False


def test_manual_review_obligation_is_never_auto_claimed() -> None:
    unsafe = ({"asset_id": "BTCUSDT", "reason": "ROLLBACK_NOT_REDUCE_ONLY"},)
    recovery = CancelRecoveryDirectiveReceipt(
        runtime_id="runtime-82",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=15,
        fencing_token=41,
        status="PREPARED",
        prepared=True,
        duplicate=False,
        cancel_risk_version=8,
        cancel_risk_receipt_id="r" * 64,
        cancel_reason="HALTED",
        pre_state_id="a" * 64,
        current_state_id="h" * 64,
        current_risk_version=11,
        current_risk_receipt_id="s" * 64,
        current_risk_state="ACTIVE",
        recovery_status="MANUAL_REVIEW",
        recovery_legs=(_leg(),),
        unsafe_assets=unsafe,
    )
    recovery_supervisor = _RecoverySupervisor(recovery)
    claims = _ClaimStore(_claim_receipt())
    wrapper = PersistedRecoveryClaimSupervisor(
        recovery_supervisor=recovery_supervisor,
        claims=claims,
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        recovery_worker_token="recovery-a",
        recovery_claim_seconds=45,
        marks={},
        observed_at=1.0,
        source_ref="phase82-manual",
    )
    assert step.claim is None
    assert claims.calls == 0
