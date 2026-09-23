from __future__ import annotations

import pytest

from brian2026.phase86_recovery_admission_interlock import (
    RecoveryAdmissionInterlockError,
    RecoveryAdmissionInterlockStore,
    RecoveryAdmissionState,
)


class FakeRpc:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, name, params):
        self.calls.append((name, dict(params)))
        value = self.response
        return value(dict(params)) if callable(value) else dict(value)


def test_open_admission_contract() -> None:
    rpc = FakeRpc({
        "blocked": False,
        "status": "OPEN",
        "runtime_id": "runtime-86",
    })
    state = RecoveryAdmissionInterlockStore(rpc).read(runtime_id="runtime-86")
    assert state.blocked is False
    assert state.status == "OPEN"
    assert state.original_cycle_id is None
    assert rpc.calls == [(
        "brian_read_shadow_recovery_admission",
        {"p_runtime_id": "runtime-86"},
    )]


def test_blocked_admission_requires_exact_cancel_lineage() -> None:
    rpc = FakeRpc({
        "blocked": True,
        "status": "RECOVERY_BARRIER",
        "runtime_id": "runtime-86",
        "original_cycle_id": "c" * 64,
        "cancel_risk_receipt_id": "r" * 64,
        "reason": "HALTED",
        "requested_at": "2026-09-23T15:30:00Z",
    })
    state = RecoveryAdmissionInterlockStore(rpc).read(runtime_id="runtime-86")
    assert state.blocked is True
    assert state.original_cycle_id == "c" * 64
    assert state.cancel_risk_receipt_id == "r" * 64
    assert state.reason == "HALTED"


def test_blocked_admission_cannot_omit_content_hash_anchors() -> None:
    with pytest.raises(ValueError, match="original cycle"):
        RecoveryAdmissionState(
            runtime_id="runtime-86",
            status="RECOVERY_BARRIER",
            blocked=True,
            original_cycle_id=None,
            cancel_risk_receipt_id="r" * 64,
            reason="HALTED",
        )


def test_reader_rejects_runtime_identity_drift() -> None:
    rpc = FakeRpc({
        "blocked": False,
        "status": "OPEN",
        "runtime_id": "other-runtime",
    })
    with pytest.raises(RecoveryAdmissionInterlockError, match="runtime_id"):
        RecoveryAdmissionInterlockStore(rpc).read(runtime_id="runtime-86")


def test_open_state_cannot_smuggle_blocked_status() -> None:
    with pytest.raises(ValueError, match="OPEN"):
        RecoveryAdmissionState(
            runtime_id="runtime-86",
            status="RECOVERY_BARRIER",
            blocked=False,
        )
