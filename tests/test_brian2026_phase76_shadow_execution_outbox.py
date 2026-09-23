from __future__ import annotations

from dataclasses import dataclass

import pytest

from brian2026.phase70_durable_runtime_store import RuntimeLease
from brian2026.phase75_atomic_governed_writeahead import AtomicGovernedWriteAheadReceipt
from brian2026.phase76_shadow_execution_outbox import (
    PersistedDispatchedRuntimeSupervisor,
    ShadowExecutionOutboxError,
    ShadowExecutionOutboxStore,
    dispatch_id_for_authorization,
)
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)


def _lease() -> RuntimeLease:
    return RuntimeLease(
        runtime_id="runtime-76",
        owner_token="owner-a",
        fencing_token=9,
        version=2,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )


def _authorization() -> AtomicGovernedWriteAheadReceipt:
    return AtomicGovernedWriteAheadReceipt(
        runtime_id="runtime-76",
        cycle_id="c" * 64,
        checkpoint_id="k" * 64,
        runtime_version_before=1,
        runtime_version_after=2,
        current_runtime_version=2,
        risk_version=4,
        risk_ledger_hash="l" * 64,
        risk_receipt_id="r" * 64,
        governed_result_id="g" * 64,
        policy_fingerprint="p" * 64,
        fencing_token=9,
        status="AUTHORIZED_AND_PERSISTED",
        authorized=True,
        duplicate=False,
    )


class FakeRpc:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, name, params):
        self.calls.append((name, dict(params)))
        if callable(self.response):
            return self.response(name, dict(params))
        return dict(self.response)


def _submitted_row(auth):
    dispatch_id = dispatch_id_for_authorization(auth)
    return {
        "submitted": True,
        "duplicate": False,
        "status": "SUBMITTED",
        "runtime_id": auth.runtime_id,
        "cycle_id": auth.cycle_id,
        "dispatch_id": dispatch_id,
        "governed_result_id": auth.governed_result_id,
        "policy_fingerprint": auth.policy_fingerprint,
        "authorization_checkpoint_id": auth.checkpoint_id,
        "authorization_runtime_version": auth.runtime_version_after,
        "current_runtime_version": auth.runtime_version_after,
        "risk_version": auth.risk_version,
        "risk_ledger_hash": auth.risk_ledger_hash,
        "risk_receipt_id": auth.risk_receipt_id,
        "fencing_token": auth.fencing_token,
    }


def test_dispatch_id_is_deterministic_from_authorization_evidence() -> None:
    auth = _authorization()
    first = dispatch_id_for_authorization(auth)
    second = dispatch_id_for_authorization(auth)
    assert first == second
    assert len(first) == 64


def test_submit_uses_only_runtime_fence_cycle_and_deterministic_dispatch_id() -> None:
    auth = _authorization()
    rpc = FakeRpc(_submitted_row(auth))
    receipt = ShadowExecutionOutboxStore(rpc).submit(_lease(), auth)

    assert receipt.submitted is True
    assert receipt.status == "SUBMITTED"
    assert receipt.dispatch_id == dispatch_id_for_authorization(auth)
    params = rpc.calls[0][1]
    assert params == {
        "p_runtime_id": "runtime-76",
        "p_owner_token": "owner-a",
        "p_fencing_token": 9,
        "p_cycle_id": auth.cycle_id,
        "p_dispatch_id": receipt.dispatch_id,
    }


def test_submitted_response_must_echo_immutable_authorization_anchors() -> None:
    auth = _authorization()
    row = _submitted_row(auth)
    row["risk_receipt_id"] = "0" * 64
    with pytest.raises(ShadowExecutionOutboxError, match="risk_receipt_id"):
        ShadowExecutionOutboxStore(FakeRpc(row)).submit(_lease(), auth)


@pytest.mark.parametrize(
    "status",
    [
        "AUTHORIZATION_MISSING",
        "LEASE_LOST",
        "RUNTIME_VERSION_CONFLICT",
        "RISK_VERSION_CONFLICT",
        "RECOVERY_BARRIER",
    ],
)
def test_fail_closed_dispatch_statuses_remain_non_submitted(status: str) -> None:
    auth = _authorization()
    row = {
        "submitted": False,
        "duplicate": False,
        "status": status,
        "runtime_id": auth.runtime_id,
        "cycle_id": auth.cycle_id,
        "dispatch_id": dispatch_id_for_authorization(auth),
        "runtime_version": 2,
        "risk_version": 5 if status == "RISK_VERSION_CONFLICT" else 4,
        "fencing_token": 9,
    }
    receipt = ShadowExecutionOutboxStore(FakeRpc(row)).submit(_lease(), auth)
    assert receipt.submitted is False
    assert receipt.status == status


def test_duplicate_dispatch_is_idempotent_but_historical_is_distinguishable() -> None:
    auth = _authorization()
    current = _submitted_row(auth)
    current["status"] = "DUPLICATE_CURRENT"
    current["duplicate"] = True
    receipt = ShadowExecutionOutboxStore(FakeRpc(current)).submit(_lease(), auth)
    assert receipt.submitted is True
    assert receipt.duplicate is True
    assert receipt.status == "DUPLICATE_CURRENT"

    historical = dict(current)
    historical["status"] = "DUPLICATE_HISTORICAL"
    historical["current_runtime_version"] = 3
    receipt = ShadowExecutionOutboxStore(FakeRpc(historical)).submit(_lease(), auth)
    assert receipt.current_runtime_version == 3
    assert receipt.status == "DUPLICATE_HISTORICAL"


@dataclass
class _Counter:
    state_version: int = 0


class _Journal:
    def __init__(self):
        self.stage = "CYCLE_CREATED"

    def latest_stage(self, cycle_id):
        del cycle_id
        return self.stage


class _Checkpoint:
    checkpoint_id = "z" * 64


class _Runtime:
    def __init__(self):
        self.venue = _Counter()
        self.projector = _Counter()
        self.journal = _Journal()

    def checkpoint(self):
        return _Checkpoint()


class _RuntimeSupervisor:
    def __init__(self):
        self.lease = _lease()
        self.persisted_version = 2
        self._valid = True
        self.runtime = _Runtime()

    @property
    def valid(self):
        return self._valid


class _Advanced:
    def __init__(self, status):
        self.durable_receipt = type("Receipt", (), {"status": status})()


class _GovernedSupervisor:
    def __init__(self, auth):
        self.runtime_supervisor = _RuntimeSupervisor()
        self.auth = auth
        self.aborted = False

    def authorize_write_ahead(self, governed):
        del governed
        assert self.runtime_supervisor.runtime.venue.state_version == 0
        return self.auth

    def abort_authorized_cycle(self, *, cycle_id, reason):
        assert cycle_id == self.auth.cycle_id
        assert reason
        self.aborted = True
        self.runtime_supervisor.runtime.journal.stage = "ABORTED"
        self.runtime_supervisor.persisted_version = 3

    def advance_authorized(self, *, marks, observed_at, source_ref):
        del marks, observed_at, source_ref
        self.runtime_supervisor.runtime.venue.state_version += 1
        self.runtime_supervisor.runtime.projector.state_version += 1
        self.runtime_supervisor.persisted_version = 3
        return _Advanced("COMMITTED")


class _Outbox:
    def __init__(self, receipt, governed_supervisor):
        self.receipt = receipt
        self.governed_supervisor = governed_supervisor

    def submit(self, lease, authorization):
        assert lease.fencing_token == authorization.fencing_token
        # Durable dispatch must be attempted before paper/local side effects.
        assert self.governed_supervisor.runtime_supervisor.runtime.venue.state_version == 0
        assert self.governed_supervisor.runtime_supervisor.runtime.projector.state_version == 0
        return self.receipt


def _dispatch_receipt(*, status="SUBMITTED", submitted=True, current_runtime_version=2):
    auth = _authorization()
    return type(
        "DispatchReceipt",
        (),
        {
            "submitted": submitted,
            "status": status,
            "current_runtime_version": current_runtime_version,
            "fencing_token": auth.fencing_token,
        },
    )()


def test_supervisor_executes_only_after_durable_dispatch_submission() -> None:
    auth = _authorization()
    governed = _GovernedSupervisor(auth)
    outbox = _Outbox(_dispatch_receipt(), governed)
    wrapper = PersistedDispatchedRuntimeSupervisor(
        governed_supervisor=governed,
        outbox=outbox,
    )

    step = wrapper.process_governed_cycle(
        object(),
        marks={},
        observed_at=1.0,
        source_ref="phase76-success",
    )
    assert step.outcome == "COMMITTED"
    assert governed.runtime_supervisor.runtime.venue.state_version == 1
    assert governed.runtime_supervisor.runtime.projector.state_version == 1


def test_risk_change_before_dispatch_aborts_without_paper_side_effect() -> None:
    auth = _authorization()
    governed = _GovernedSupervisor(auth)
    outbox = _Outbox(
        _dispatch_receipt(status="RISK_VERSION_CONFLICT", submitted=False),
        governed,
    )
    wrapper = PersistedDispatchedRuntimeSupervisor(
        governed_supervisor=governed,
        outbox=outbox,
    )

    step = wrapper.process_governed_cycle(
        object(),
        marks={},
        observed_at=1.0,
        source_ref="phase76-risk-stale",
    )
    assert step.outcome == "ABORTED_RISK_STALE"
    assert governed.aborted is True
    assert governed.runtime_supervisor.runtime.journal.stage == "ABORTED"
    assert governed.runtime_supervisor.runtime.venue.state_version == 0
    assert governed.runtime_supervisor.runtime.projector.state_version == 0
    assert governed.runtime_supervisor.valid is True



def test_recovery_barrier_before_dispatch_aborts_without_paper_side_effect() -> None:
    auth = _authorization()
    governed = _GovernedSupervisor(auth)
    outbox = _Outbox(
        _dispatch_receipt(status="RECOVERY_BARRIER", submitted=False),
        governed,
    )
    wrapper = PersistedDispatchedRuntimeSupervisor(
        governed_supervisor=governed,
        outbox=outbox,
    )

    step = wrapper.process_governed_cycle(
        object(),
        marks={},
        observed_at=1.0,
        source_ref="phase86-recovery-barrier",
    )
    assert step.outcome == "ABORTED_RECOVERY_BARRIER"
    assert governed.aborted is True
    assert governed.runtime_supervisor.runtime.journal.stage == "ABORTED"
    assert governed.runtime_supervisor.runtime.venue.state_version == 0
    assert governed.runtime_supervisor.runtime.projector.state_version == 0
    assert governed.runtime_supervisor.valid is True

def test_runtime_conflict_before_dispatch_invalidates_supervisor() -> None:
    auth = _authorization()
    governed = _GovernedSupervisor(auth)
    outbox = _Outbox(
        _dispatch_receipt(status="RUNTIME_VERSION_CONFLICT", submitted=False),
        governed,
    )
    wrapper = PersistedDispatchedRuntimeSupervisor(
        governed_supervisor=governed,
        outbox=outbox,
    )

    with pytest.raises(PersistedRuntimeStaleError, match="RUNTIME_VERSION_CONFLICT"):
        wrapper.process_governed_cycle(
            object(),
            marks={},
            observed_at=1.0,
            source_ref="phase76-runtime-stale",
        )
    assert governed.runtime_supervisor.valid is False
    assert governed.runtime_supervisor.runtime.venue.state_version == 0


def test_lease_loss_before_dispatch_invalidates_supervisor() -> None:
    auth = _authorization()
    governed = _GovernedSupervisor(auth)
    outbox = _Outbox(
        _dispatch_receipt(status="LEASE_LOST", submitted=False),
        governed,
    )
    wrapper = PersistedDispatchedRuntimeSupervisor(
        governed_supervisor=governed,
        outbox=outbox,
    )
    with pytest.raises(PersistedRuntimeLeaseError, match="lease was lost"):
        wrapper.process_governed_cycle(
            object(),
            marks={},
            observed_at=1.0,
            source_ref="phase76-lease-lost",
        )
    assert governed.runtime_supervisor.valid is False


def test_historical_dispatch_retry_requires_runtime_reload() -> None:
    auth = _authorization()
    governed = _GovernedSupervisor(auth)
    outbox = _Outbox(
        _dispatch_receipt(
            status="DUPLICATE_HISTORICAL",
            submitted=True,
            current_runtime_version=3,
        ),
        governed,
    )
    wrapper = PersistedDispatchedRuntimeSupervisor(
        governed_supervisor=governed,
        outbox=outbox,
    )
    with pytest.raises(PersistedRuntimeStaleError, match="historical"):
        wrapper.process_governed_cycle(
            object(),
            marks={},
            observed_at=1.0,
            source_ref="phase76-historical",
        )
    assert governed.runtime_supervisor.valid is False
