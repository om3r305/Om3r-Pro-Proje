from __future__ import annotations

import copy

import pytest

from brian2026.phase68_operational_risk_governor import (
    EquityPoint,
    OperationalRiskPolicy,
)
from brian2026.phase70_durable_runtime_store import RuntimeLease
from brian2026.phase72_operational_risk_ledger import OperationalRiskLedger
from brian2026.phase73_operational_risk_store import (
    OperationalRiskStore,
    OperationalRiskStoreError,
)


TS = 1_760_000_000.0


def _lease() -> RuntimeLease:
    return RuntimeLease(
        runtime_id="runtime-risk",
        owner_token="owner-a",
        fencing_token=4,
        version=3,
        status="ACQUIRED",
        acquired=True,
        lease_until=None,
    )


def _ledger(*, halted: bool = False) -> OperationalRiskLedger:
    policy = OperationalRiskPolicy(
        max_drawdown_fraction=0.50,
        max_daily_loss_fraction=0.50,
        max_unknown_order_outcomes=1,
        max_market_data_age_seconds=10.0,
    )
    ledger = OperationalRiskLedger(policy)
    governor = ledger.governor()
    receipt = governor.evaluate(
        now=TS,
        equity_points=(
            EquityPoint(TS - 60, 1000.0),
            EquityPoint(TS, 1000.0),
        ),
        closed_trades=(),
        health_events=(),
        market_data_timestamp=TS - 20 if halted else TS,
    )
    ledger.append(receipt)
    return ledger


class FakeRpc:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, name: str, params):
        self.calls.append((name, dict(params)))
        result = self.responses[name]
        if callable(result):
            return result(dict(params))
        return copy.deepcopy(result)


def test_commit_sends_validated_manifest_and_parses_success() -> None:
    ledger = _ledger()
    manifest = ledger.manifest()
    rpc = FakeRpc({
        "brian_commit_operational_risk_ledger": {
            "committed": True,
            "duplicate": False,
            "status": "COMMITTED",
            "runtime_id": "runtime-risk",
            "fencing_token": 4,
            "version": 1,
            "ledger_hash": manifest["ledger_hash"],
        }
    })
    store = OperationalRiskStore(rpc)
    receipt = store.commit(_lease(), expected_version=0, ledger=ledger)

    assert receipt.committed is True
    assert receipt.duplicate is False
    assert receipt.version == 1
    assert receipt.ledger_hash == manifest["ledger_hash"]
    assert rpc.calls[0][0] == "brian_commit_operational_risk_ledger"
    assert rpc.calls[0][1]["p_manifest"] == manifest


@pytest.mark.parametrize("status", ["CAS_CONFLICT", "LEASE_LOST"])
def test_commit_fail_closed_statuses_remain_noncommitting(status: str) -> None:
    ledger = _ledger()
    manifest = ledger.manifest()
    rpc = FakeRpc({
        "brian_commit_operational_risk_ledger": {
            "committed": False,
            "duplicate": False,
            "status": status,
            "runtime_id": "runtime-risk",
            "fencing_token": 4,
            "version": 2,
            "ledger_hash": manifest["ledger_hash"],
        }
    })
    receipt = OperationalRiskStore(rpc).commit(
        _lease(),
        expected_version=0,
        ledger=ledger,
    )
    assert receipt.committed is False
    assert receipt.status == status
    assert receipt.current_version == 2


def test_duplicate_current_is_parsed_as_idempotent_success() -> None:
    ledger = _ledger()
    manifest = ledger.manifest()
    rpc = FakeRpc({
        "brian_commit_operational_risk_ledger": {
            "committed": True,
            "duplicate": True,
            "status": "DUPLICATE_CURRENT",
            "runtime_id": "runtime-risk",
            "fencing_token": 4,
            "version": 3,
            "current_version": 3,
            "ledger_hash": manifest["ledger_hash"],
        }
    })
    receipt = OperationalRiskStore(rpc).commit(
        _lease(),
        expected_version=2,
        ledger=ledger,
    )
    assert receipt.committed is True
    assert receipt.duplicate is True
    assert receipt.status == "DUPLICATE_CURRENT"


def test_commit_rejects_database_hash_for_different_ledger() -> None:
    ledger = _ledger()
    rpc = FakeRpc({
        "brian_commit_operational_risk_ledger": {
            "committed": True,
            "duplicate": False,
            "status": "COMMITTED",
            "runtime_id": "runtime-risk",
            "fencing_token": 4,
            "version": 1,
            "ledger_hash": "0" * 64,
        }
    })
    with pytest.raises(OperationalRiskStoreError, match="does not match submitted"):
        OperationalRiskStore(rpc).commit(
            _lease(),
            expected_version=0,
            ledger=ledger,
        )


def test_load_revalidates_manifest_and_all_database_head_anchors() -> None:
    ledger = _ledger(halted=True)
    manifest = ledger.manifest()
    rpc = FakeRpc({
        "brian_read_operational_risk_ledger": {
            "runtime_id": "runtime-risk",
            "version": 2,
            "ledger_hash": manifest["ledger_hash"],
            "manifest": manifest,
            "policy_hash": manifest["policy_hash"],
            "head_entry_id": manifest["head_entry_id"],
            "current_state": manifest["current_state"],
            "halt_latched": manifest["halt_latched"],
            "shadow_only": True,
            "live_execution": False,
        }
    })
    stored = OperationalRiskStore(rpc).load(runtime_id="runtime-risk")
    assert stored is not None
    assert stored.version == 2
    assert stored.ledger.manifest() == manifest
    assert stored.current_state == "HALTED"
    assert stored.halt_latched is True

    restored_governor = stored.ledger.governor()
    assert restored_governor.state == "HALTED"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ledger_hash", "0" * 64, "ledger_hash"),
        ("policy_hash", "0" * 64, "policy_hash"),
        ("head_entry_id", "0" * 64, "head_entry_id"),
        ("current_state", "REDUCING", "current_state"),
        ("halt_latched", False, "halt_latched"),
    ],
)
def test_load_rejects_database_anchor_mismatch(field, value, message) -> None:
    ledger = _ledger(halted=True)
    manifest = ledger.manifest()
    row = {
        "runtime_id": "runtime-risk",
        "version": 1,
        "ledger_hash": manifest["ledger_hash"],
        "manifest": manifest,
        "policy_hash": manifest["policy_hash"],
        "head_entry_id": manifest["head_entry_id"],
        "current_state": manifest["current_state"],
        "halt_latched": manifest["halt_latched"],
    }
    row[field] = value
    rpc = FakeRpc({"brian_read_operational_risk_ledger": row})
    with pytest.raises(OperationalRiskStoreError, match=message):
        OperationalRiskStore(rpc).load(runtime_id="runtime-risk")


def test_load_rejects_tampered_manifest_even_when_outer_head_is_changed_with_it() -> None:
    ledger = _ledger()
    manifest = copy.deepcopy(ledger.manifest())
    manifest["entries"][0]["receipt"]["reasons"] = ["forged"]
    rpc = FakeRpc({
        "brian_read_operational_risk_ledger": {
            "runtime_id": "runtime-risk",
            "version": 1,
            "ledger_hash": manifest["ledger_hash"],
            "manifest": manifest,
            "policy_hash": manifest["policy_hash"],
            "head_entry_id": manifest["head_entry_id"],
            "current_state": manifest["current_state"],
            "halt_latched": manifest["halt_latched"],
        }
    })
    with pytest.raises(OperationalRiskStoreError, match="failed validation"):
        OperationalRiskStore(rpc).load(runtime_id="runtime-risk")


def test_version_zero_head_returns_none() -> None:
    rpc = FakeRpc({
        "brian_read_operational_risk_ledger": {
            "runtime_id": "runtime-risk",
            "version": 0,
            "ledger_hash": None,
            "manifest": None,
            "policy_hash": None,
            "head_entry_id": None,
            "current_state": None,
            "halt_latched": None,
        }
    })
    assert OperationalRiskStore(rpc).load(runtime_id="runtime-risk") is None
