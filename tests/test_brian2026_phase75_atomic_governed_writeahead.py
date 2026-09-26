from __future__ import annotations

from dataclasses import replace

import pytest

from brian2026.phase57_shadow_execution_cycle import ShadowExecutionCycle
from brian2026.phase60_shadow_state_ledger import ShadowAccountState, ShadowStateLedger
from brian2026.phase61_stateful_paper_venue import PaperVenue, PaperVenueConfig
from brian2026.phase64_local_execution_projector import LocalExecutionProjector
from brian2026.phase66_runtime_coordinator import ShadowPaperRuntimeCoordinator
from brian2026.phase67_durable_runtime_orchestrator import DurableShadowPaperRuntime
from brian2026.phase68_operational_risk_governor import (
    EquityPoint,
    OperationalRiskGovernor,
    OperationalRiskPolicy,
)
from brian2026.phase69_governed_shadow_execution import GovernedShadowExecution
from brian2026.phase70_durable_runtime_store import (
    RuntimeCommitReceipt,
    RuntimeLease,
    StoredRuntimeCheckpoint,
)
from brian2026.phase71_persisted_runtime_supervisor import (
    PersistedDurableRuntimeSupervisor,
    PersistedRuntimeStaleError,
)
from brian2026.phase72_operational_risk_ledger import OperationalRiskLedger
from brian2026.phase73_operational_risk_store import StoredOperationalRiskLedger
from brian2026.phase75_atomic_governed_writeahead import (
    AtomicGovernedWriteAheadError,
    AtomicGovernedWriteAheadReceipt,
    AtomicGovernedWriteAheadStaleError,
    AtomicGovernedWriteAheadStore,
    PersistedGovernedRuntimeSupervisor,
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
        source_ref="phase75-genesis",
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


def _risk(runtime_id: str = "runtime-75") -> StoredOperationalRiskLedger:
    policy = OperationalRiskPolicy(
        max_drawdown_fraction=0.50,
        max_daily_loss_fraction=0.50,
        stoploss_limit=10,
        max_unknown_order_outcomes=2,
    )
    governor = OperationalRiskGovernor(policy)
    receipt = governor.evaluate(
        now=TS,
        equity_points=(
            EquityPoint(TS - 60, 1000.0),
            EquityPoint(TS, 1000.0),
        ),
        closed_trades=(),
        health_events=(),
        market_data_timestamp=TS,
    )
    ledger = OperationalRiskLedger(policy)
    ledger.append(receipt)
    manifest = ledger.manifest()
    return StoredOperationalRiskLedger(
        runtime_id=runtime_id,
        version=4,
        ledger=ledger,
        ledger_hash=str(manifest["ledger_hash"]),
        policy_hash=str(manifest["policy_hash"]),
        head_entry_id=str(manifest["head_entry_id"]),
        current_state=str(manifest["current_state"]),
        halt_latched=bool(manifest["halt_latched"]),
    )


def _governed(risk: StoredOperationalRiskLedger, cycle_char: str = "c"):
    cycle = ShadowExecutionCycle(
        source_plan_id="phase75-plan",
        items=(),
        initial_available_cash_usd=1000.0,
        reserved_new_risk_cash_usd=0.0,
        remaining_unreserved_cash_usd=1000.0,
        denied_assets=(),
        pending_reversal_assets=(),
        cycle_id=cycle_char * 64,
    )
    return GovernedShadowExecution(
        operational_risk_receipt_id=risk.ledger.entries[-1].receipt.receipt_id,
        trading_state=risk.current_state,
        blocked_new_risk_assets=(),
        policy_fingerprint="p" * 64,
        cycle=cycle,
        result_id="g" * 64,
    )


class FakeRpc:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def __call__(self, name, params):
        self.calls.append((name, dict(params)))
        return self.responder(name, dict(params))


def test_atomic_store_submits_validated_writeahead_and_persisted_risk_head() -> None:
    runtime = _runtime()
    risk = _risk()
    governed = _governed(risk)
    runtime.journal_cycle(governed.cycle)
    checkpoint = runtime.checkpoint()
    lease = RuntimeLease(
        runtime_id="runtime-75",
        owner_token="owner-a",
        fencing_token=6,
        version=1,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )

    def responder(name, params):
        assert name == "brian_authorize_and_persist_governed_cycle"
        return {
            "authorized": True,
            "duplicate": False,
            "status": "AUTHORIZED_AND_PERSISTED",
            "runtime_id": "runtime-75",
            "cycle_id": governed.cycle.cycle_id,
            "runtime_version_before": 1,
            "runtime_version_after": 2,
            "current_runtime_version": 2,
            "risk_version": risk.version,
            "fencing_token": 6,
            "checkpoint_id": checkpoint.checkpoint_id,
            "risk_ledger_hash": risk.ledger_hash,
            "risk_receipt_id": governed.operational_risk_receipt_id,
            "governed_result_id": governed.result_id,
            "policy_fingerprint": governed.policy_fingerprint,
        }

    rpc = FakeRpc(responder)
    receipt = AtomicGovernedWriteAheadStore(rpc).authorize_and_persist(
        lease,
        expected_runtime_version=1,
        risk=risk,
        governed=governed,
        checkpoint=checkpoint,
    )

    assert receipt.authorized is True
    assert receipt.runtime_version_after == 2
    params = rpc.calls[0][1]
    assert params["p_checkpoint"] == checkpoint.to_dict()
    assert params["p_risk_version"] == risk.version
    assert params["p_risk_receipt_id"] == governed.operational_risk_receipt_id


def test_atomic_store_rejects_checkpoint_with_different_cycle_body_before_rpc() -> None:
    runtime = _runtime()
    risk = _risk()
    governed = _governed(risk, "c")
    runtime.journal_cycle(_governed(risk, "d").cycle)
    checkpoint = runtime.checkpoint()
    rpc = FakeRpc(lambda *_: {})

    with pytest.raises(AtomicGovernedWriteAheadError, match="does not contain governed cycle"):
        AtomicGovernedWriteAheadStore(rpc).authorize_and_persist(
            RuntimeLease(
                runtime_id="runtime-75",
                owner_token="owner-a",
                fencing_token=1,
                version=1,
                status="ACQUIRED",
                acquired=True,
                lease_until=None,
            ),
            expected_runtime_version=1,
            risk=risk,
            governed=governed,
            checkpoint=checkpoint,
        )
    assert rpc.calls == []


class MemoryRuntimeStore:
    def __init__(self, runtime: DurableShadowPaperRuntime):
        self.version = 1
        self.checkpoint = runtime.checkpoint()
        self.owner = "owner-a"
        self.fence = 1

    def load(self, *, runtime_id):
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
        return replace(lease, version=self.version, acquired=True, status="RENEWED")

    def release(self, lease):
        del lease
        return True


class StaticRiskStore:
    def __init__(self, risk):
        self.risk = risk

    def load(self, *, runtime_id):
        assert runtime_id == self.risk.runtime_id
        return self.risk


class AtomicMemoryStore:
    def __init__(self, runtime_store, supervisor, risk, *, status="AUTHORIZED_AND_PERSISTED"):
        self.runtime_store = runtime_store
        self.supervisor = supervisor
        self.risk = risk
        self.status = status
        self.called_before_paper = False

    def authorize_and_persist(
        self,
        lease,
        *,
        expected_runtime_version,
        risk,
        governed,
        checkpoint,
    ):
        assert self.supervisor.runtime.venue.state_version == 0
        assert self.supervisor.runtime.projector.projection_version == 0
        self.called_before_paper = True

        if self.status == "RISK_VERSION_CONFLICT":
            raise AtomicGovernedWriteAheadStaleError(
                "governed write-ahead rejected with RISK_VERSION_CONFLICT"
            )

        before = expected_runtime_version
        after = before + 1
        self.runtime_store.version = after
        self.runtime_store.checkpoint = checkpoint
        duplicate = self.status.startswith("DUPLICATE_")
        current = after if self.status != "DUPLICATE_HISTORICAL" else after + 1
        return AtomicGovernedWriteAheadReceipt(
            runtime_id=lease.runtime_id,
            cycle_id=governed.cycle.cycle_id,
            checkpoint_id=checkpoint.checkpoint_id,
            runtime_version_before=before,
            runtime_version_after=after,
            current_runtime_version=current,
            risk_version=risk.version,
            risk_ledger_hash=risk.ledger_hash,
            risk_receipt_id=governed.operational_risk_receipt_id,
            governed_result_id=governed.result_id,
            policy_fingerprint=governed.policy_fingerprint,
            fencing_token=lease.fencing_token,
            status=self.status,
            authorized=True,
            duplicate=duplicate,
        )


def _runtime_supervisor():
    runtime = _runtime()
    store = MemoryRuntimeStore(runtime)
    lease = RuntimeLease(
        runtime_id="runtime-75",
        owner_token="owner-a",
        fencing_token=1,
        version=1,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )
    supervisor = PersistedDurableRuntimeSupervisor(
        store=store,
        lease=lease,
        runtime=runtime,
        persisted_version=1,
    )
    return supervisor, store


def test_governed_supervisor_authorizes_and_persists_before_any_paper_side_effect() -> None:
    supervisor, runtime_store = _runtime_supervisor()
    risk = _risk()
    governed = _governed(risk)
    atomic = AtomicMemoryStore(runtime_store, supervisor, risk)
    wrapper = PersistedGovernedRuntimeSupervisor(
        runtime_supervisor=supervisor,
        risk_store=StaticRiskStore(risk),
        atomic_store=atomic,
    )

    step = wrapper.process_governed_cycle(
        governed,
        marks={},
        observed_at=TS + 1,
        source_ref="phase75-success",
    )

    assert atomic.called_before_paper is True
    assert step.authorization.status == "AUTHORIZED_AND_PERSISTED"
    assert step.advanced_status == "COMMITTED"
    assert supervisor.runtime.venue.state_version == 1
    assert supervisor.runtime.projector.projection_version == 1
    assert supervisor.runtime.ledger.pending_cycle_id is None
    assert supervisor.persisted_version == 3
    assert runtime_store.version == 3


def test_risk_conflict_invalidates_supervisor_and_prevents_paper_side_effect() -> None:
    supervisor, runtime_store = _runtime_supervisor()
    risk = _risk()
    governed = _governed(risk)
    atomic = AtomicMemoryStore(
        runtime_store,
        supervisor,
        risk,
        status="RISK_VERSION_CONFLICT",
    )
    wrapper = PersistedGovernedRuntimeSupervisor(
        runtime_supervisor=supervisor,
        risk_store=StaticRiskStore(risk),
        atomic_store=atomic,
    )

    with pytest.raises(AtomicGovernedWriteAheadStaleError, match="RISK_VERSION_CONFLICT"):
        wrapper.process_governed_cycle(
            governed,
            marks={},
            observed_at=TS + 1,
            source_ref="phase75-risk-conflict",
        )

    assert supervisor.valid is False
    assert supervisor.runtime.venue.state_version == 0
    assert supervisor.runtime.projector.projection_version == 0
    assert runtime_store.version == 1


def test_historical_duplicate_authorization_requires_reload_before_execution() -> None:
    supervisor, runtime_store = _runtime_supervisor()
    risk = _risk()
    governed = _governed(risk)
    atomic = AtomicMemoryStore(
        runtime_store,
        supervisor,
        risk,
        status="DUPLICATE_HISTORICAL",
    )
    wrapper = PersistedGovernedRuntimeSupervisor(
        runtime_supervisor=supervisor,
        risk_store=StaticRiskStore(risk),
        atomic_store=atomic,
    )

    with pytest.raises(AtomicGovernedWriteAheadStaleError, match="historical"):
        wrapper.process_governed_cycle(
            governed,
            marks={},
            observed_at=TS + 1,
            source_ref="phase75-historical",
        )
    assert supervisor.valid is False
    assert supervisor.runtime.venue.state_version == 0


def test_phase71_external_checkpoint_acceptance_rejects_wrong_checkpoint() -> None:
    supervisor, _ = _runtime_supervisor()
    with pytest.raises(PersistedRuntimeStaleError, match="does not match"):
        supervisor.accept_external_checkpoint_commit(
            checkpoint_id="0" * 64,
            version=2,
        )
    assert supervisor.valid is False
