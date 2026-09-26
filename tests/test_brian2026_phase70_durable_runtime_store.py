from __future__ import annotations

import copy

import pytest

from brian2026.phase60_shadow_state_ledger import ShadowAccountState, ShadowStateLedger
from brian2026.phase61_stateful_paper_venue import PaperVenue, PaperVenueConfig
from brian2026.phase64_local_execution_projector import LocalExecutionProjector
from brian2026.phase66_runtime_coordinator import ShadowPaperRuntimeCoordinator
from brian2026.phase67_durable_runtime_orchestrator import (
    DurableRuntimeCheckpoint,
    DurableShadowPaperRuntime,
)
from brian2026.phase70_durable_runtime_store import (
    DurableRuntimeStore,
    DurableRuntimeStoreError,
    RuntimeLease,
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
        source_ref="phase70-genesis",
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


def _checkpoint() -> DurableRuntimeCheckpoint:
    return _runtime().checkpoint()


class FakeRpc:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, name: str, params):
        self.calls.append((name, dict(params)))
        value = self.responses[name]
        if callable(value):
            return value(dict(params))
        return copy.deepcopy(value)


def _lease() -> RuntimeLease:
    return RuntimeLease(
        runtime_id="runtime-1",
        owner_token="owner-a",
        fencing_token=7,
        version=0,
        status="ACQUIRED",
        acquired=True,
        lease_until="2026-09-23T10:00:00Z",
    )


def test_phase67_checkpoint_dict_roundtrip_is_hash_verified() -> None:
    checkpoint = _checkpoint()
    restored = DurableRuntimeCheckpoint.from_dict(checkpoint.to_dict())
    assert restored == checkpoint

    tampered = checkpoint.to_dict()
    tampered["journal_manifest"]["journal_hash"] = "0" * 64
    with pytest.raises(Exception, match="content hash mismatch"):
        DurableRuntimeCheckpoint.from_dict(tampered)


def test_acquire_maps_rpc_lease_contract_without_supabase_dependency() -> None:
    rpc = FakeRpc({
        "brian_acquire_shadow_runtime_lease": {
            "acquired": True,
            "status": "ACQUIRED",
            "runtime_id": "runtime-1",
            "fencing_token": 3,
            "version": 4,
            "lease_until": "2026-09-23T10:00:00Z",
        }
    })
    store = DurableRuntimeStore(rpc)
    lease = store.acquire(
        runtime_id="runtime-1",
        owner_token="owner-a",
        lease_seconds=30,
    )
    assert lease.acquired is True
    assert lease.fencing_token == 3
    assert lease.version == 4
    assert rpc.calls[0][0] == "brian_acquire_shadow_runtime_lease"
    assert rpc.calls[0][1]["p_lease_seconds"] == 30


def test_commit_transports_canonical_checkpoint_and_parses_success() -> None:
    checkpoint = _checkpoint()
    rpc = FakeRpc({
        "brian_commit_shadow_runtime_checkpoint": {
            "committed": True,
            "duplicate": False,
            "status": "COMMITTED",
            "runtime_id": "runtime-1",
            "fencing_token": 7,
            "version": 1,
            "checkpoint_id": checkpoint.checkpoint_id,
            "journal_hash": checkpoint.journal_manifest["journal_hash"],
            "head_state_id": checkpoint.runtime_checkpoint.shadow_ledger_manifest["head_state_id"],
        }
    })
    store = DurableRuntimeStore(rpc)
    receipt = store.commit(
        _lease(),
        expected_version=0,
        checkpoint=checkpoint,
    )
    assert receipt.committed is True
    assert receipt.duplicate is False
    assert receipt.status == "COMMITTED"
    assert receipt.version == 1
    sent = rpc.calls[0][1]["p_checkpoint"]
    assert sent == checkpoint.to_dict()


def test_commit_rejects_database_receipt_for_different_checkpoint() -> None:
    checkpoint = _checkpoint()
    rpc = FakeRpc({
        "brian_commit_shadow_runtime_checkpoint": {
            "committed": True,
            "duplicate": False,
            "status": "COMMITTED",
            "runtime_id": "runtime-1",
            "fencing_token": 7,
            "version": 1,
            "checkpoint_id": "0" * 64,
        }
    })
    with pytest.raises(DurableRuntimeStoreError, match="does not match submitted"):
        DurableRuntimeStore(rpc).commit(
            _lease(),
            expected_version=0,
            checkpoint=checkpoint,
        )


@pytest.mark.parametrize("status", ["CAS_CONFLICT", "LEASE_LOST"])
def test_noncommitting_database_statuses_remain_fail_closed(status: str) -> None:
    checkpoint = _checkpoint()
    rpc = FakeRpc({
        "brian_commit_shadow_runtime_checkpoint": {
            "committed": False,
            "status": status,
            "runtime_id": "runtime-1",
            "fencing_token": 7,
            "version": 2,
            "checkpoint_id": checkpoint.checkpoint_id,
        }
    })
    receipt = DurableRuntimeStore(rpc).commit(
        _lease(),
        expected_version=0,
        checkpoint=checkpoint,
    )
    assert receipt.committed is False
    assert receipt.status == status
    assert receipt.current_version == 2


def test_load_validates_every_database_head_anchor_and_checkpoint_hash() -> None:
    checkpoint = _checkpoint()
    head_id = checkpoint.runtime_checkpoint.shadow_ledger_manifest["head_state_id"]
    journal_hash = checkpoint.journal_manifest["journal_hash"]
    rpc = FakeRpc({
        "brian_read_shadow_runtime_checkpoint": {
            "runtime_id": "runtime-1",
            "version": 1,
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_payload": checkpoint.to_dict(),
            "journal_hash": journal_hash,
            "head_state_id": head_id,
            "pending_cycle_id": None,
            "fencing_token": 7,
            "lease_until": "2026-09-23T10:00:00Z",
            "shadow_only": True,
            "live_execution": False,
        }
    })
    stored = DurableRuntimeStore(rpc).load(runtime_id="runtime-1")
    assert stored is not None
    assert stored.version == 1
    assert stored.checkpoint == checkpoint
    assert stored.journal_hash == journal_hash
    assert stored.head_state_id == head_id

    restored_runtime = DurableShadowPaperRuntime.restore(stored.checkpoint)
    assert restored_runtime.ledger.head_state.state_id == head_id
    assert restored_runtime.venue.cash_usd == pytest.approx(1000.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_id", "0" * 64, "checkpoint_id"),
        ("journal_hash", "0" * 64, "journal_hash"),
        ("head_state_id", "0" * 64, "head_state_id"),
        ("pending_cycle_id", "forged-cycle", "pending_cycle_id"),
    ],
)
def test_load_rejects_database_head_metadata_that_disagrees_with_payload(
    field: str,
    value,
    message: str,
) -> None:
    checkpoint = _checkpoint()
    row = {
        "runtime_id": "runtime-1",
        "version": 1,
        "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint_payload": checkpoint.to_dict(),
        "journal_hash": checkpoint.journal_manifest["journal_hash"],
        "head_state_id": checkpoint.runtime_checkpoint.shadow_ledger_manifest["head_state_id"],
        "pending_cycle_id": None,
        "fencing_token": 7,
        "lease_until": None,
    }
    row[field] = value
    rpc = FakeRpc({"brian_read_shadow_runtime_checkpoint": row})
    with pytest.raises(DurableRuntimeStoreError, match=message):
        DurableRuntimeStore(rpc).load(runtime_id="runtime-1")


def test_load_rejects_tampered_checkpoint_even_when_database_metadata_is_changed_with_it() -> None:
    checkpoint = _checkpoint()
    tampered = checkpoint.to_dict()
    tampered["runtime_checkpoint"]["shadow_ledger_manifest"]["head_state_id"] = "0" * 64
    rpc = FakeRpc({
        "brian_read_shadow_runtime_checkpoint": {
            "runtime_id": "runtime-1",
            "version": 1,
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_payload": tampered,
            "journal_hash": checkpoint.journal_manifest["journal_hash"],
            "head_state_id": "0" * 64,
            "pending_cycle_id": None,
            "fencing_token": 7,
            "lease_until": None,
        }
    })
    with pytest.raises(DurableRuntimeStoreError, match="Phase67 validation"):
        DurableRuntimeStore(rpc).load(runtime_id="runtime-1")


def test_version_zero_head_is_not_misrepresented_as_persisted_checkpoint() -> None:
    rpc = FakeRpc({
        "brian_read_shadow_runtime_checkpoint": {
            "runtime_id": "runtime-1",
            "version": 0,
            "checkpoint_id": None,
            "checkpoint_payload": None,
            "journal_hash": None,
            "head_state_id": None,
            "pending_cycle_id": None,
            "fencing_token": 1,
            "lease_until": None,
        }
    })
    assert DurableRuntimeStore(rpc).load(runtime_id="runtime-1") is None


def test_renew_and_release_use_exact_owner_and_fencing_token() -> None:
    rpc = FakeRpc({
        "brian_renew_shadow_runtime_lease": {
            "renewed": True,
            "status": "RENEWED",
            "runtime_id": "runtime-1",
            "fencing_token": 7,
            "version": 3,
            "lease_until": "2026-09-23T10:01:00Z",
        },
        "brian_release_shadow_runtime_lease": {
            "released": True,
            "status": "RELEASED",
            "runtime_id": "runtime-1",
            "fencing_token": 7,
            "version": 3,
        },
    })
    store = DurableRuntimeStore(rpc)
    renewed = store.renew(_lease(), lease_seconds=60)
    assert renewed.acquired is True
    assert renewed.status == "RENEWED"
    assert renewed.version == 3
    assert store.release(renewed) is True

    renew_params = rpc.calls[0][1]
    release_params = rpc.calls[1][1]
    assert renew_params["p_owner_token"] == "owner-a"
    assert renew_params["p_fencing_token"] == 7
    assert release_params["p_owner_token"] == "owner-a"
    assert release_params["p_fencing_token"] == 7
