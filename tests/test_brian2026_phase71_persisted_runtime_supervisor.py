from __future__ import annotations

from dataclasses import replace

import pytest

from brian2026.phase46_execution_simulator import SimulatedExecutionReceipt
from brian2026.phase56_pretrade_risk_engine import PreTradeRiskReceipt
from brian2026.phase57_shadow_execution_cycle import (
    ShadowExecutionCycle,
    ShadowExecutionCycleItem,
)
from brian2026.phase60_shadow_state_ledger import ShadowAccountState, ShadowStateLedger
from brian2026.phase61_stateful_paper_venue import PaperVenue, PaperVenueConfig
from brian2026.phase64_local_execution_projector import LocalExecutionProjector
from brian2026.phase66_runtime_coordinator import ShadowPaperRuntimeCoordinator
from brian2026.phase67_durable_runtime_orchestrator import DurableShadowPaperRuntime
from brian2026.phase70_durable_runtime_store import (
    RuntimeCommitReceipt,
    RuntimeLease,
    StoredRuntimeCheckpoint,
)
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedDurableRuntimeSupervisor,
    PersistedRuntimeError,
    PersistedRuntimeLeaseError,
    PersistedRuntimeStaleError,
)


TS = 1_760_000_000.0


def _runtime() -> DurableShadowPaperRuntime:
    genesis = ShadowAccountState(
        account_id="paper-acct",
        observed_at=TS,
        equity_usd=1000.0,
        available_cash_usd=1000.0,
        position_weights=(),
        covered_assets=("BTCUSDT",),
        source_kind="GENESIS",
        source_ref="phase71-genesis",
    )
    coordinator = ShadowPaperRuntimeCoordinator(
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
    return DurableShadowPaperRuntime(coordinator)


def _cycle(cycle_id: str = "cycle-71", *, price: float = 100.0) -> ShadowExecutionCycle:
    risk = PreTradeRiskReceipt(
        action="ALLOW",
        trading_state="ACTIVE",
        asset_id="BTCUSDT",
        requested_notional_usd=price,
        reduce_only=False,
        reasons=(),
        projected_position_weight=0.1,
        checks=(("fixture", True),),
    )
    execution = SimulatedExecutionReceipt(
        status="FILLED",
        side="BUY",
        order_type="MARKET",
        submit_timestamp=TS,
        venue_timestamp=TS + 0.1,
        snapshot_timestamp=TS + 0.1,
        requested_base=1.0,
        filled_base=1.0,
        fill_fraction=1.0,
        average_fill_price=price,
        best_reference_price=price,
        adverse_slippage_bps=0.0,
        levels_consumed=1,
        slipped_one_tick=False,
        reason="phase71 fixture",
    )
    item = ShadowExecutionCycleItem(
        instruction_kind="OPEN",
        asset_id="BTCUSDT",
        risk_receipt=risk,
        execution_receipt=execution,
        pending_reversal=None,
        new_risk_cash_reserved_usd=price,
        status="fixture",
    )
    return ShadowExecutionCycle(
        source_plan_id=f"plan-{cycle_id}",
        items=(item,),
        initial_available_cash_usd=1000.0,
        reserved_new_risk_cash_usd=price,
        remaining_unreserved_cash_usd=1000.0 - price,
        denied_assets=(),
        pending_reversal_assets=(),
        cycle_id=cycle_id,
    )


class MemoryStore:
    def __init__(self):
        self.version = 0
        self.checkpoint = None
        self.owner = None
        self.fence = 0
        self.commit_history = []
        self.fail_next_status = None
        self.duplicate_next = False
        self.drift_version_on_renew = False

    def acquire(self, *, runtime_id, owner_token, lease_seconds):
        del lease_seconds
        if self.owner is not None and self.owner != owner_token:
            return RuntimeLease(
                runtime_id=runtime_id,
                owner_token=owner_token,
                fencing_token=self.fence,
                version=self.version,
                status="BLOCKED_ACTIVE",
                acquired=False,
                lease_until=None,
            )
        if self.owner is None:
            self.fence += 1
            self.owner = owner_token
            status = "ACQUIRED" if self.fence == 1 else "EXPIRED_RECOVERY"
        else:
            status = "ALREADY_OWNED"
        return RuntimeLease(
            runtime_id=runtime_id,
            owner_token=owner_token,
            fencing_token=self.fence,
            version=self.version,
            status=status,
            acquired=True,
            lease_until=None,
        )

    def load(self, *, runtime_id):
        if self.checkpoint is None:
            return None
        return StoredRuntimeCheckpoint(
            runtime_id=runtime_id,
            version=self.version,
            checkpoint=self.checkpoint,
            journal_hash=str(self.checkpoint.journal_manifest["journal_hash"]),
            head_state_id=str(
                self.checkpoint.runtime_checkpoint.shadow_ledger_manifest["head_state_id"]
            ),
            pending_cycle_id=self.checkpoint.runtime_checkpoint.pending_cycle_id,
            fencing_token=self.fence,
            lease_until=None,
        )

    def commit(self, lease, *, expected_version, checkpoint):
        if lease.owner_token != self.owner or lease.fencing_token != self.fence:
            return RuntimeCommitReceipt(
                runtime_id=lease.runtime_id,
                checkpoint_id=checkpoint.checkpoint_id,
                fencing_token=self.fence,
                version=self.version,
                current_version=self.version,
                status="LEASE_LOST",
                committed=False,
                duplicate=False,
            )

        if self.fail_next_status is not None:
            status = self.fail_next_status
            self.fail_next_status = None
            return RuntimeCommitReceipt(
                runtime_id=lease.runtime_id,
                checkpoint_id=checkpoint.checkpoint_id,
                fencing_token=self.fence,
                version=self.version,
                current_version=self.version,
                status=status,
                committed=False,
                duplicate=False,
            )

        if self.checkpoint is not None and checkpoint.checkpoint_id == self.checkpoint.checkpoint_id:
            return RuntimeCommitReceipt(
                runtime_id=lease.runtime_id,
                checkpoint_id=checkpoint.checkpoint_id,
                fencing_token=self.fence,
                version=self.version,
                current_version=self.version,
                status="DUPLICATE_CURRENT",
                committed=True,
                duplicate=True,
            )

        if expected_version != self.version:
            return RuntimeCommitReceipt(
                runtime_id=lease.runtime_id,
                checkpoint_id=checkpoint.checkpoint_id,
                fencing_token=self.fence,
                version=self.version,
                current_version=self.version,
                status="CAS_CONFLICT",
                committed=False,
                duplicate=False,
            )

        self.version += 1
        self.checkpoint = checkpoint
        self.commit_history.append((self.version, checkpoint))

        if self.duplicate_next:
            self.duplicate_next = False
            return RuntimeCommitReceipt(
                runtime_id=lease.runtime_id,
                checkpoint_id=checkpoint.checkpoint_id,
                fencing_token=self.fence,
                version=self.version,
                current_version=self.version,
                status="DUPLICATE_CURRENT",
                committed=True,
                duplicate=True,
            )

        return RuntimeCommitReceipt(
            runtime_id=lease.runtime_id,
            checkpoint_id=checkpoint.checkpoint_id,
            fencing_token=self.fence,
            version=self.version,
            current_version=self.version,
            status="COMMITTED",
            committed=True,
            duplicate=False,
        )

    def renew(self, lease, *, lease_seconds):
        del lease_seconds
        if lease.owner_token != self.owner or lease.fencing_token != self.fence:
            return replace(lease, acquired=False, status="RENEWAL_LOST", version=self.version)
        version = self.version + 1 if self.drift_version_on_renew else self.version
        return replace(lease, acquired=True, status="RENEWED", version=version)

    def release(self, lease):
        if lease.owner_token != self.owner or lease.fencing_token != self.fence:
            return False
        self.owner = None
        return True


def test_fresh_acquire_persists_bootstrap_checkpoint_before_work() -> None:
    store = MemoryStore()
    supervisor = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-a",
        lease_seconds=30,
        initial_runtime=_runtime(),
    )

    assert supervisor.persisted_version == 1
    assert supervisor.lease.version == 1
    assert len(store.commit_history) == 1
    assert store.checkpoint is not None
    assert store.checkpoint.journal_manifest["entries"] == []
    assert supervisor.valid is True


def test_cycle_is_persisted_write_ahead_before_paper_side_effects() -> None:
    store = MemoryStore()
    supervisor = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-a",
        lease_seconds=30,
        initial_runtime=_runtime(),
    )
    cycle = _cycle("write-ahead")

    step = supervisor.persist_write_ahead_cycle(cycle)

    assert step.persisted_version == 2
    assert supervisor.runtime.venue.state_version == 0
    assert supervisor.runtime.projector.projection_version == 0
    assert store.version == 2
    persisted = store.checkpoint
    assert persisted is not None
    assert persisted.journal_manifest["cycles"][cycle.cycle_id]["cycle_id"] == cycle.cycle_id
    assert persisted.journal_manifest["entries"][0]["stage"] == "CYCLE_CREATED"


def test_full_process_cycle_uses_two_distinct_durable_commits() -> None:
    store = MemoryStore()
    supervisor = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-a",
        lease_seconds=30,
        initial_runtime=_runtime(),
    )

    write_ahead, advanced = supervisor.process_cycle(
        _cycle("complete"),
        marks={"BTCUSDT": 110.0},
        observed_at=TS + 10,
        source_ref="phase71-complete",
    )

    assert write_ahead.persisted_version == 2
    assert advanced.persisted_version == 3
    assert advanced.durable_receipt is not None
    assert advanced.durable_receipt.status == "COMMITTED"
    assert supervisor.runtime.ledger.pending_cycle_id is None
    assert supervisor.runtime.venue.state_version == 1
    assert supervisor.runtime.projector.projection_version == 1
    assert len(store.commit_history) == 3
    assert store.checkpoint.journal_manifest["entries"][-1]["stage"] == "COMMITTED"


def test_restart_after_write_ahead_replays_same_cycle_without_duplicate_fill() -> None:
    store = MemoryStore()
    first = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-a",
        lease_seconds=30,
        initial_runtime=_runtime(),
    )
    cycle = _cycle("restart")
    first.persist_write_ahead_cycle(cycle)
    assert first.runtime.venue.state_version == 0
    assert store.version == 2
    assert first.release() is True

    second = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-b",
        lease_seconds=30,
    )
    assert second.persisted_version == 2
    assert second.runtime.venue.state_version == 0
    assert second.runtime.journal.latest_stage(cycle.cycle_id) == "CYCLE_CREATED"

    result = second.advance_pending(
        marks={"BTCUSDT": 110.0},
        observed_at=TS + 20,
        source_ref="phase71-restart",
    )

    assert result.durable_receipt is not None
    assert result.durable_receipt.status == "COMMITTED"
    assert second.runtime.venue.state_version == 1
    assert second.runtime.projector.projection_version == 1
    assert second.runtime.venue.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert second.persisted_version == 3


def test_cas_conflict_after_local_advance_invalidates_supervisor_and_db_stays_write_ahead() -> None:
    store = MemoryStore()
    supervisor = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-a",
        lease_seconds=30,
        initial_runtime=_runtime(),
    )
    cycle = _cycle("cas-conflict")
    supervisor.persist_write_ahead_cycle(cycle)
    persisted_before = store.checkpoint
    assert persisted_before is not None
    assert store.version == 2

    store.fail_next_status = "CAS_CONFLICT"
    with pytest.raises(PersistedRuntimeStaleError, match="CAS_CONFLICT"):
        supervisor.advance_pending(
            marks={"BTCUSDT": 105.0},
            observed_at=TS + 10,
            source_ref="phase71-cas-conflict",
        )

    assert supervisor.valid is False
    assert store.version == 2
    assert store.checkpoint == persisted_before
    assert store.checkpoint.journal_manifest["entries"][-1]["stage"] == "CYCLE_CREATED"
    with pytest.raises(PersistedRuntimeStaleError, match="reload"):
        supervisor.advance_pending(
            marks={"BTCUSDT": 105.0},
            observed_at=TS + 11,
            source_ref="must-not-continue",
        )


def test_after_cas_conflict_restart_recovers_from_authoritative_write_ahead_checkpoint() -> None:
    store = MemoryStore()
    first = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-a",
        lease_seconds=30,
        initial_runtime=_runtime(),
    )
    cycle = _cycle("cas-recover")
    first.persist_write_ahead_cycle(cycle)
    store.fail_next_status = "CAS_CONFLICT"
    with pytest.raises(PersistedRuntimeStaleError):
        first.advance_pending(
            marks={"BTCUSDT": 105.0},
            observed_at=TS + 10,
            source_ref="phase71-failed-local",
        )
    assert first.release() is True

    recovered = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-b",
        lease_seconds=30,
    )
    assert recovered.runtime.venue.state_version == 0
    assert recovered.runtime.journal.latest_stage(cycle.cycle_id) == "CYCLE_CREATED"

    final = recovered.advance_pending(
        marks={"BTCUSDT": 105.0},
        observed_at=TS + 20,
        source_ref="phase71-recovered",
    )
    assert final.durable_receipt.status == "COMMITTED"
    assert recovered.runtime.venue.position("BTCUSDT").quantity == pytest.approx(1.0)
    assert recovered.runtime.venue.state_version == 1


def test_lost_lease_during_commit_invalidates_supervisor() -> None:
    store = MemoryStore()
    supervisor = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-a",
        lease_seconds=30,
        initial_runtime=_runtime(),
    )
    supervisor.persist_write_ahead_cycle(_cycle("lease-lost"))
    store.fail_next_status = "LEASE_LOST"

    with pytest.raises(PersistedRuntimeLeaseError, match="ownership was lost"):
        supervisor.advance_pending(
            marks={"BTCUSDT": 100.0},
            observed_at=TS + 10,
            source_ref="phase71-lease-lost",
        )
    assert supervisor.valid is False


def test_duplicate_current_is_accepted_as_network_retry_equivalent() -> None:
    store = MemoryStore()
    supervisor = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-a",
        lease_seconds=30,
        initial_runtime=_runtime(),
    )
    store.duplicate_next = True

    step = supervisor.persist_write_ahead_cycle(_cycle("duplicate-current"))

    assert step.commit_status == "DUPLICATE_CURRENT"
    assert step.persisted_version == 2
    assert supervisor.persisted_version == 2
    assert supervisor.lease.version == 2
    assert supervisor.valid is True


def test_renew_detects_external_version_drift() -> None:
    store = MemoryStore()
    supervisor = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-a",
        lease_seconds=30,
        initial_runtime=_runtime(),
    )
    store.drift_version_on_renew = True
    with pytest.raises(PersistedRuntimeStaleError, match="advanced outside"):
        supervisor.renew(lease_seconds=30)
    assert supervisor.valid is False


def test_existing_durable_head_wins_over_caller_initial_runtime() -> None:
    store = MemoryStore()
    first = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-a",
        lease_seconds=30,
        initial_runtime=_runtime(),
    )
    cycle = _cycle("existing-head")
    first.persist_write_ahead_cycle(cycle)
    first.release()

    caller_memory = _runtime()
    second = PersistedDurableRuntimeSupervisor.acquire(
        store=store,
        runtime_id="runtime-71",
        owner_token="owner-b",
        lease_seconds=30,
        initial_runtime=caller_memory,
    )

    assert second.persisted_version == 2
    assert second.runtime.journal.latest_stage(cycle.cycle_id) == "CYCLE_CREATED"
    assert caller_memory.journal.latest_stage(cycle.cycle_id) is None



def test_startup_failure_releases_acquired_lease() -> None:
    store = MemoryStore()

    with pytest.raises(PersistedRuntimeError, match="initial_runtime"):
        PersistedDurableRuntimeSupervisor.acquire(
            store=store,
            runtime_id="runtime-71",
            owner_token="owner-a",
            lease_seconds=30,
        )

    assert store.owner is None


def test_load_failure_releases_acquired_lease() -> None:
    class BrokenLoadStore(MemoryStore):
        def load(self, *, runtime_id):
            del runtime_id
            raise RuntimeError("load exploded")

    store = BrokenLoadStore()
    with pytest.raises(RuntimeError, match="load exploded"):
        PersistedDurableRuntimeSupervisor.acquire(
            store=store,
            runtime_id="runtime-71",
            owner_token="owner-a",
            lease_seconds=30,
            initial_runtime=_runtime(),
        )

    assert store.owner is None


def test_bootstrap_commit_failure_releases_acquired_lease() -> None:
    store = MemoryStore()
    store.fail_next_status = "CAS_CONFLICT"

    with pytest.raises(PersistedRuntimeStaleError, match="CAS_CONFLICT"):
        PersistedDurableRuntimeSupervisor.acquire(
            store=store,
            runtime_id="runtime-71",
            owner_token="owner-a",
            lease_seconds=30,
            initial_runtime=_runtime(),
        )

    assert store.owner is None
