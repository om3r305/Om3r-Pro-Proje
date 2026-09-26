from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase70_durable_runtime_store import RuntimeLease
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from brian2026.phase84_recovery_execution_checkpoint import (
    RecoveryCheckpointReceipt,
)
from brian2026.phase85_recovery_completion_audit import (
    PersistedRecoveryCompletionAuditSupervisor,
    RecoveryCompletionAuditError,
    RecoveryCompletionAuditReceipt,
    RecoveryCompletionAuditStore,
)


def _lease() -> RuntimeLease:
    return RuntimeLease(
        runtime_id="runtime-85",
        owner_token="owner-a",
        fencing_token=61,
        version=21,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )


def _audit_row(*, status="CERTIFIED", certified=True, duplicate=False):
    return {
        "certified": certified,
        "duplicate": duplicate,
        "status": status,
        "runtime_id": "runtime-85",
        "dispatch_id": "d" * 64,
        "original_cycle_id": "o" * 64,
        "recovery_cycle_id": "c" * 64,
        "cancel_risk_receipt_id": "r" * 64 if certified else None,
        "runtime_version": 21,
        "completion_checkpoint_id": "k" * 64,
        "start_head_state_id": "h" * 64 if certified else None,
        "final_head_state_id": "z" * 64 if certified else None,
        "paper_checkpoint_id": "p" * 64 if certified else None,
        "recovery_claim_fencing_token": 4 if certified else None,
        "recovery_fill_count": 1 if certified else None,
        "leg_audits": (
            [{
                "asset_id": "BTCUSDT",
                "pre_recovery_quantity": 2.5,
                "recovery_delta_quantity": -1.5,
                "final_quantity": 1.0,
                "direction_preserved": True,
                "quantity_exposure_reduced": True,
            }]
            if certified else []
        ),
        "failures": [],
        "fencing_token": 61,
    }


class FakeRpc:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, name, params):
        self.calls.append((name, dict(params)))
        value = self.response
        return value(dict(params)) if callable(value) else dict(value)


def test_store_certifies_exact_completion_lineage() -> None:
    rpc = FakeRpc(_audit_row())
    receipt = RecoveryCompletionAuditStore(rpc).certify(
        _lease(),
        original_cycle_id="o" * 64,
        recovery_cycle_id="c" * 64,
        expected_checkpoint_id="k" * 64,
    )
    assert receipt.certified is True
    assert receipt.status == "CERTIFIED"
    assert receipt.recovery_fill_count == 1
    assert receipt.leg_audits[0]["quantity_exposure_reduced"] is True
    assert rpc.calls[0][1] == {
        "p_runtime_id": "runtime-85",
        "p_owner_token": "owner-a",
        "p_fencing_token": 61,
        "p_original_cycle_id": "o" * 64,
        "p_recovery_cycle_id": "c" * 64,
        "p_expected_checkpoint_id": "k" * 64,
    }


def test_store_accepts_exact_duplicate_certificate() -> None:
    receipt = RecoveryCompletionAuditStore(
        FakeRpc(_audit_row(status="DUPLICATE", certified=True, duplicate=True))
    ).certify(
        _lease(),
        original_cycle_id="o" * 64,
        recovery_cycle_id="c" * 64,
        expected_checkpoint_id="k" * 64,
    )
    assert receipt.certified is True
    assert receipt.duplicate is True
    assert receipt.status == "DUPLICATE"


def test_audit_failed_requires_explicit_failure_evidence() -> None:
    row = _audit_row(status="AUDIT_FAILED", certified=False)
    row["failures"] = [{
        "asset_id": "BTCUSDT",
        "reason": "FINAL_PAPER_DIRECTION_FLIPPED",
    }]
    row["leg_audits"] = [{
        "asset_id": "BTCUSDT",
        "pre_recovery_quantity": 1.0,
        "final_quantity": -0.2,
    }]
    receipt = RecoveryCompletionAuditStore(FakeRpc(row)).certify(
        _lease(),
        original_cycle_id="o" * 64,
        recovery_cycle_id="c" * 64,
        expected_checkpoint_id="k" * 64,
    )
    assert receipt.certified is False
    assert receipt.failures[0]["reason"] == "FINAL_PAPER_DIRECTION_FLIPPED"


def test_store_rejects_completion_checkpoint_drift() -> None:
    row = _audit_row()
    row["completion_checkpoint_id"] = "x" * 64
    with pytest.raises(RecoveryCompletionAuditError, match="checkpoint drift"):
        RecoveryCompletionAuditStore(FakeRpc(row)).certify(
            _lease(),
            original_cycle_id="o" * 64,
            recovery_cycle_id="c" * 64,
            expected_checkpoint_id="k" * 64,
        )


class _Checkpoint:
    checkpoint_id = "k" * 64


class _Runtime:
    def checkpoint(self):
        return _Checkpoint()


class _RuntimeSupervisor:
    def __init__(self):
        self.lease = _lease()
        self.persisted_version = 21
        self._valid = True
        self.runtime = _Runtime()

    @property
    def valid(self):
        return self._valid


class _RecoverySupervisor:
    def __init__(self, progress):
        self.progress = progress
        self.runtime = _RuntimeSupervisor()

    def _runtime_supervisor(self):
        return self.runtime

    def process_governed_cycle(self, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            progress=self.progress,
            recovery_cycle_id=(
                None if self.progress is None else self.progress.recovery_cycle_id
            ),
            outcome="RECOVERY_COMMITTED_PENDING_AUDIT",
            persisted_version=21,
            checkpoint_id="k" * 64,
        )


class _Audits:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = 0

    def certify(self, lease, *, original_cycle_id, recovery_cycle_id, expected_checkpoint_id):
        assert lease.runtime_id == "runtime-85"
        assert original_cycle_id == "o" * 64
        assert recovery_cycle_id == "c" * 64
        assert expected_checkpoint_id == "k" * 64
        self.calls += 1
        return self.receipt


def _progress(*, terminal=True):
    return RecoveryCheckpointReceipt(
        runtime_id="runtime-85",
        original_cycle_id="o" * 64,
        recovery_cycle_id="c" * 64,
        dispatch_id="d" * 64,
        checkpoint_id="k" * 64,
        journal_stage="COMMITTED" if terminal else "PAPER_APPLIED",
        version=21,
        current_version=21,
        fencing_token=61,
        recovery_claim_fencing_token=4,
        status="RECOVERY_COMMITTED_PENDING_AUDIT" if terminal else "COMMITTED",
        committed=True,
        duplicate=False,
        terminal=terminal,
        head_state_id="z" * 64 if terminal else "h" * 64,
    )


def _receipt_from_row(row):
    return RecoveryCompletionAuditStore(FakeRpc(row)).certify(
        _lease(),
        original_cycle_id="o" * 64,
        recovery_cycle_id="c" * 64,
        expected_checkpoint_id="k" * 64,
    )


def test_wrapper_certifies_terminal_phase84_progress() -> None:
    recovery = _RecoverySupervisor(_progress())
    audits = _Audits(_receipt_from_row(_audit_row()))
    wrapper = PersistedRecoveryCompletionAuditSupervisor(
        recovery_supervisor=recovery,
        audits=audits,
    )
    step = wrapper.process_governed_cycle(object())
    assert step.audit is not None and step.audit.certified is True
    assert step.outcome.endswith("RECOVERY_COMPLETED")
    assert audits.calls == 1
    assert recovery.runtime.valid is True


def test_wrapper_does_not_audit_nonterminal_recovery_progress() -> None:
    recovery = _RecoverySupervisor(_progress(terminal=False))
    audits = _Audits(_receipt_from_row(_audit_row()))
    wrapper = PersistedRecoveryCompletionAuditSupervisor(
        recovery_supervisor=recovery,
        audits=audits,
    )
    step = wrapper.process_governed_cycle(object())
    assert step.audit is None
    assert audits.calls == 0


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("LEASE_LOST", PersistedRuntimeLeaseError),
        ("HEAD_MOVED", PersistedRuntimeStaleError),
        ("CLAIM_STATE_INVALID", PersistedRuntimeStaleError),
        ("EVIDENCE_INVALID", PersistedRuntimeStaleError),
    ],
)
def test_wrapper_fails_closed_on_completion_authority_errors(status, error) -> None:
    row = _audit_row(status=status, certified=False)
    row["completion_checkpoint_id"] = "k" * 64
    recovery = _RecoverySupervisor(_progress())
    wrapper = PersistedRecoveryCompletionAuditSupervisor(
        recovery_supervisor=recovery,
        audits=_Audits(_receipt_from_row(row)),
    )
    with pytest.raises(error):
        wrapper.process_governed_cycle(object())
    assert recovery.runtime.valid is False


def test_wrapper_invalidates_automation_on_actual_recovery_audit_failure() -> None:
    row = _audit_row(status="AUDIT_FAILED", certified=False)
    row["failures"] = [{
        "asset_id": "BTCUSDT",
        "reason": "RECOVERY_OVERREDUCED_AND_FLIPPED_QUANTITY",
    }]
    row["leg_audits"] = [{"asset_id": "BTCUSDT"}]
    recovery = _RecoverySupervisor(_progress())
    wrapper = PersistedRecoveryCompletionAuditSupervisor(
        recovery_supervisor=recovery,
        audits=_Audits(_receipt_from_row(row)),
    )
    with pytest.raises(RecoveryCompletionAuditError, match="OVERREDUCED"):
        wrapper.process_governed_cycle(object())
    assert recovery.runtime.valid is False
