from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase60_shadow_state_ledger import ShadowAccountState, ShadowStateLedger
from brian2026.phase61_stateful_paper_venue import PaperVenue, PaperVenueConfig
from brian2026.phase64_local_execution_projector import LocalExecutionProjector
from brian2026.phase66_runtime_coordinator import ShadowPaperRuntimeCoordinator
from brian2026.phase67_durable_runtime_orchestrator import DurableShadowPaperRuntime
from brian2026.phase70_durable_runtime_store import RuntimeLease
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)
from brian2026.phase77_execution_claim_lifecycle import (
    ClaimCompletionReceipt,
    ExecutionClaimReceipt,
)
from brian2026.phase78_execution_kill_switch import ExecutionKillSwitchDecision
from brian2026.phase79_atomic_execution_start import AtomicExecutionStartReceipt
from brian2026.phase80_claim_fenced_checkpoint import (
    ClaimFencedCheckpointError,
    ClaimFencedCheckpointReceipt,
    ClaimFencedCheckpointStore,
    PersistedClaimFencedRuntimeSupervisor,
)


TS = 1_760_000_000.0


def _lease() -> RuntimeLease:
    return RuntimeLease(
        runtime_id="runtime-80",
        owner_token="owner-a",
        fencing_token=21,
        version=9,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )


def _claim() -> ExecutionClaimReceipt:
    return ExecutionClaimReceipt(
        runtime_id="runtime-80",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=9,
        fencing_token=21,
        claim_fencing_token=7,
        status="CLAIMED",
        claimed=True,
        cancelled=False,
        terminal=False,
        risk_version=15,
        risk_receipt_id="r" * 64,
        journal_stage="CYCLE_CREATED",
        resume_only=False,
        cancel_reason=None,
        claim_until="2026-09-23T14:00:00Z",
        completion_checkpoint_id=None,
    )


def _checkpoint():
    genesis = ShadowAccountState(
        account_id="paper-acct",
        observed_at=TS,
        equity_usd=1000.0,
        available_cash_usd=1000.0,
        position_weights=(),
        covered_assets=("BTCUSDT",),
        source_kind="GENESIS",
        source_ref="phase80-genesis",
    )
    runtime = DurableShadowPaperRuntime(
        ShadowPaperRuntimeCoordinator(
            ShadowStateLedger(genesis),
            PaperVenue(
                PaperVenueConfig(
                    account_id="paper-acct",
                    starting_cash_usd=1000.0,
                    fee_bps=0.0,
                )
            ),
            LocalExecutionProjector("paper-acct"),
        )
    )
    return runtime.checkpoint()


class FakeRpc:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, name, params):
        self.calls.append((name, dict(params)))
        value = self.response
        return value(dict(params)) if callable(value) else dict(value)


def _commit_row(checkpoint, *, status="COMMITTED", committed=True, duplicate=False):
    return {
        "committed": committed,
        "duplicate": duplicate,
        "status": status,
        "runtime_id": "runtime-80",
        "dispatch_id": "d" * 64,
        "cycle_id": "c" * 64,
        "checkpoint_id": checkpoint.checkpoint_id,
        "journal_stage": "COMMITTED",
        "version": 10 if committed else 9,
        "current_version": 10 if committed else 9,
        "fencing_token": 21,
        "claim_fencing_token": 7,
    }


def test_store_commits_exact_claim_fence_and_checkpoint_identity() -> None:
    checkpoint = _checkpoint()
    rpc = FakeRpc(_commit_row(checkpoint))
    receipt = ClaimFencedCheckpointStore(rpc).commit(
        _lease(),
        _claim(),
        worker_token="worker-a",
        expected_version=9,
        checkpoint=checkpoint,
    )

    assert receipt.committed is True
    assert receipt.status == "COMMITTED"
    assert receipt.version == 10
    assert rpc.calls[0][0] == "brian_commit_claimed_shadow_runtime_checkpoint"
    params = rpc.calls[0][1]
    assert params["p_worker_token"] == "worker-a"
    assert params["p_claim_fencing_token"] == 7
    assert params["p_expected_version"] == 9
    assert params["p_checkpoint"] == checkpoint.to_dict()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dispatch_id", "0" * 64, "dispatch_id"),
        ("checkpoint_id", "0" * 64, "checkpoint_id"),
        ("fencing_token", 99, "runtime fence"),
        ("claim_fencing_token", 99, "claim fence"),
    ],
)
def test_store_rejects_database_commit_anchor_drift(field, value, message) -> None:
    checkpoint = _checkpoint()
    row = _commit_row(checkpoint)
    row[field] = value
    with pytest.raises(ClaimFencedCheckpointError, match=message):
        ClaimFencedCheckpointStore(FakeRpc(row)).commit(
            _lease(),
            _claim(),
            worker_token="worker-a",
            expected_version=9,
            checkpoint=checkpoint,
        )


def test_store_preserves_claim_lost_as_non_committing_receipt() -> None:
    checkpoint = _checkpoint()
    row = _commit_row(
        checkpoint,
        status="CLAIM_LOST",
        committed=False,
        duplicate=False,
    )
    receipt = ClaimFencedCheckpointStore(FakeRpc(row)).commit(
        _lease(),
        _claim(),
        worker_token="worker-a",
        expected_version=9,
        checkpoint=checkpoint,
    )
    assert receipt.committed is False
    assert receipt.status == "CLAIM_LOST"
    assert receipt.version == 9


class _Journal:
    def __init__(self):
        self.stage = "CYCLE_CREATED"

    def latest_stage(self, cycle_id):
        del cycle_id
        return self.stage


class _Checkpoint:
    def __init__(self, checkpoint_id="z" * 64):
        self.checkpoint_id = checkpoint_id


class _Runtime:
    def __init__(self):
        self.journal = _Journal()

    def checkpoint(self):
        return _Checkpoint()


class _RuntimeSupervisor:
    def __init__(self, *, outcome="COMMITTED"):
        self.lease = _lease()
        self.persisted_version = 9
        self._valid = True
        self.runtime = _Runtime()
        self.outcome = outcome
        self.advance_calls = 0
        self.external_commits = []

    @property
    def valid(self):
        return self._valid

    def advance_pending_in_memory(self, *, marks, observed_at, source_ref):
        del marks, observed_at, source_ref
        self.advance_calls += 1
        self.runtime.journal.stage = (
            "COMMITTED" if self.outcome == "COMMITTED" else "PAPER_APPLIED"
        )
        return SimpleNamespace(status=self.outcome)

    def accept_external_checkpoint_commit(self, *, checkpoint_id, version):
        assert checkpoint_id == self.runtime.checkpoint().checkpoint_id
        assert version > self.persisted_version
        self.external_commits.append((checkpoint_id, version))
        self.persisted_version = version
        self.lease = RuntimeLease(
            runtime_id=self.lease.runtime_id,
            owner_token=self.lease.owner_token,
            fencing_token=self.lease.fencing_token,
            version=version,
            status=self.lease.status,
            acquired=True,
            lease_until=self.lease.lease_until,
        )


class _Phase75:
    def __init__(self, runtime_supervisor):
        self.runtime_supervisor = runtime_supervisor
        self.aborted = False

    def abort_authorized_cycle(self, *, cycle_id, reason):
        assert cycle_id == "c" * 64
        assert reason.startswith("phase80:")
        self.aborted = True
        self.runtime_supervisor.runtime.journal.stage = "ABORTED"
        self.runtime_supervisor.persisted_version += 1


class _Phase76:
    def __init__(self, runtime_supervisor):
        self.governed_supervisor = _Phase75(runtime_supervisor)


class _Claims:
    def __init__(self):
        self.complete_calls = 0

    def complete(self, lease, claim, *, worker_token):
        del lease, claim, worker_token
        self.complete_calls += 1
        return ClaimCompletionReceipt(
            runtime_id="runtime-80",
            cycle_id="c" * 64,
            dispatch_id="d" * 64,
            claim_fencing_token=7,
            status="COMPLETED",
            completed=True,
            duplicate=False,
            completion_checkpoint_id="z" * 64,
        )


class _Phase77:
    def __init__(self, runtime_supervisor, *, terminal=None):
        self.dispatched_supervisor = _Phase76(runtime_supervisor)
        self.claims = _Claims()
        self.terminal = terminal

    def authorize_submit_and_claim(self, governed, *, worker_token, claim_seconds):
        del governed, worker_token, claim_seconds
        return object(), object(), _claim(), self.terminal


class _Starts:
    def __init__(self, receipt):
        self.receipt = receipt

    def mark_started(self, lease, claim, *, worker_token):
        del lease, claim, worker_token
        return self.receipt


class _CheckpointStore:
    def __init__(self, runtime_supervisor, *, status="COMMITTED"):
        self.runtime_supervisor = runtime_supervisor
        self.status = status
        self.calls = 0
        self.saw_advanced_state = False

    def commit(
        self,
        lease,
        claim,
        *,
        worker_token,
        expected_version,
        checkpoint,
    ):
        del worker_token
        self.calls += 1
        self.saw_advanced_state = self.runtime_supervisor.advance_calls == 1
        committed = self.status in {"COMMITTED", "DUPLICATE_CURRENT"}
        return ClaimFencedCheckpointReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=claim.cycle_id,
            dispatch_id=claim.dispatch_id,
            checkpoint_id=checkpoint.checkpoint_id,
            journal_stage=self.runtime_supervisor.runtime.journal.stage,
            version=expected_version + 1 if committed else expected_version,
            current_version=expected_version + 1 if committed else expected_version,
            fencing_token=lease.fencing_token,
            claim_fencing_token=claim.claim_fencing_token,
            status=self.status,
            committed=committed,
            duplicate=self.status == "DUPLICATE_CURRENT",
        )


class _Kill:
    def __init__(self, decision):
        self.decision = decision

    def check(self, lease, claim, *, worker_token):
        del lease, claim, worker_token
        return self.decision


def _start_receipt(*, status="STARTED", started=True, terminal=False, reason=None):
    return AtomicExecutionStartReceipt(
        runtime_id="runtime-80",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=9,
        fencing_token=21,
        claim_fencing_token=7,
        status=status,
        started=started,
        duplicate=False,
        terminal=terminal,
        cancel_requested=status == "CANCELLED_BEFORE_EXECUTION",
        risk_version=16 if status not in {"LEASE_LOST", "CLAIM_LOST", "RISK_STATE_UNAVAILABLE"} else None,
        risk_receipt_id=("s" * 64) if status not in {"LEASE_LOST", "CLAIM_LOST", "RISK_STATE_UNAVAILABLE"} else None,
        risk_state=("HALTED" if status == "CANCELLED_BEFORE_EXECUTION" else "ACTIVE") if status not in {"LEASE_LOST", "CLAIM_LOST", "RISK_STATE_UNAVAILABLE"} else None,
        journal_stage="CYCLE_CREATED",
        phase78_status=(
            "CANCELLED_BEFORE_EXECUTION"
            if status == "CANCELLED_BEFORE_EXECUTION"
            else "PROCEED"
        ),
        reason=reason,
        resume_only=False,
    )


def _decision(
    status="COMPLETED",
    *,
    proceed=False,
    cancel=False,
    terminal=True,
    stage="COMMITTED",
    reason=None,
):
    return ExecutionKillSwitchDecision(
        runtime_id="runtime-80",
        cycle_id="c" * 64,
        dispatch_id="d" * 64,
        runtime_version=10,
        status=status,
        proceed=proceed,
        cancel_requested=cancel,
        terminal=terminal,
        risk_version=17,
        risk_receipt_id="t" * 64,
        journal_stage=stage,
        reason=reason,
        phase="AFTER_START" if cancel else None,
        completion_checkpoint_id=None,
    )


def test_supervisor_stages_in_memory_then_claim_fenced_commits_then_completes() -> None:
    runtime = _RuntimeSupervisor(outcome="COMMITTED")
    phase77 = _Phase77(runtime)
    checkpoint_store = _CheckpointStore(runtime)
    wrapper = PersistedClaimFencedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=_Starts(_start_receipt()),
        checkpoints=checkpoint_store,
        kill_switch=_Kill(_decision()),
    )

    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase80-success",
    )
    assert runtime.advance_calls == 1
    assert checkpoint_store.saw_advanced_state is True
    assert runtime.external_commits == [("z" * 64, 10)]
    assert runtime.persisted_version == 10
    assert phase77.claims.complete_calls == 1
    assert step.outcome == "COMMITTED"


def test_claim_lost_at_commit_invalidates_advanced_local_runtime() -> None:
    runtime = _RuntimeSupervisor(outcome="MARKS_REQUIRED")
    phase77 = _Phase77(runtime)
    wrapper = PersistedClaimFencedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=_Starts(_start_receipt()),
        checkpoints=_CheckpointStore(runtime, status="CLAIM_LOST"),
        kill_switch=_Kill(_decision()),
    )

    with pytest.raises(PersistedRuntimeStaleError, match="CLAIM_LOST"):
        wrapper.process_governed_cycle(
            object(),
            worker_token="worker-a",
            claim_seconds=30,
            marks={},
            observed_at=1.0,
            source_ref="phase80-claim-lost",
        )
    assert runtime.advance_calls == 1
    assert runtime.external_commits == []
    assert runtime.valid is False
    assert phase77.claims.complete_calls == 0


def test_runtime_lease_loss_at_commit_invalidates_local_runtime() -> None:
    runtime = _RuntimeSupervisor(outcome="MARKS_REQUIRED")
    phase77 = _Phase77(runtime)
    wrapper = PersistedClaimFencedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=_Starts(_start_receipt()),
        checkpoints=_CheckpointStore(runtime, status="LEASE_LOST"),
        kill_switch=_Kill(_decision()),
    )
    with pytest.raises(PersistedRuntimeLeaseError, match="lease lost"):
        wrapper.process_governed_cycle(
            object(),
            worker_token="worker-a",
            claim_seconds=30,
            marks={},
            observed_at=1.0,
            source_ref="phase80-lease-lost",
        )
    assert runtime.valid is False


def test_pre_start_cancel_never_advances_or_uses_claim_fenced_commit() -> None:
    runtime = _RuntimeSupervisor()
    phase77 = _Phase77(runtime)
    checkpoint_store = _CheckpointStore(runtime)
    wrapper = PersistedClaimFencedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=_Starts(
            _start_receipt(
                status="CANCELLED_BEFORE_EXECUTION",
                started=False,
                terminal=True,
                reason="HALTED",
            )
        ),
        checkpoints=checkpoint_store,
        kill_switch=_Kill(_decision()),
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase80-cancel",
    )
    assert step.outcome == "CANCELLED_BEFORE_EXECUTION"
    assert runtime.advance_calls == 0
    assert checkpoint_store.calls == 0
    assert phase77.dispatched_supervisor.governed_supervisor.aborted is True


def test_post_commit_cancel_request_keeps_committed_history_and_completes_claim() -> None:
    runtime = _RuntimeSupervisor(outcome="COMMITTED")
    phase77 = _Phase77(runtime)
    wrapper = PersistedClaimFencedRuntimeSupervisor(
        claimed_supervisor=phase77,
        starts=_Starts(_start_receipt()),
        checkpoints=_CheckpointStore(runtime),
        kill_switch=_Kill(
            _decision(
                "CANCEL_REQUESTED",
                proceed=True,
                cancel=True,
                terminal=False,
                stage="COMMITTED",
                reason="HALTED",
            )
        ),
    )
    step = wrapper.process_governed_cycle(
        object(),
        worker_token="worker-a",
        claim_seconds=30,
        marks={},
        observed_at=1.0,
        source_ref="phase80-post-commit-kill",
    )
    assert step.outcome == "COMMITTED_CANCEL_REQUESTED"
    assert runtime.external_commits == [("z" * 64, 10)]
    assert phase77.claims.complete_calls == 1
