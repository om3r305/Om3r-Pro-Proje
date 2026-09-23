from __future__ import annotations

import copy

import pytest

from brian2026.phase57_shadow_execution_cycle import ShadowExecutionCycle
from brian2026.phase68_operational_risk_governor import (
    EquityPoint,
    OperationalRiskGovernor,
    OperationalRiskPolicy,
)
from brian2026.phase69_governed_shadow_execution import GovernedShadowExecution
from brian2026.phase70_durable_runtime_store import RuntimeLease
from brian2026.phase72_operational_risk_ledger import OperationalRiskLedger
from brian2026.phase73_operational_risk_store import StoredOperationalRiskLedger
from brian2026.phase74_governed_cycle_binding import (
    GovernedCycleBindingError,
    GovernedCycleBindingStore,
)


TS = 1_760_000_000.0


def _risk(runtime_id: str = "runtime-74") -> StoredOperationalRiskLedger:
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
        version=3,
        ledger=ledger,
        ledger_hash=str(manifest["ledger_hash"]),
        policy_hash=str(manifest["policy_hash"]),
        head_entry_id=str(manifest["head_entry_id"]),
        current_state=str(manifest["current_state"]),
        halt_latched=bool(manifest["halt_latched"]),
    )


def _lease(runtime_id: str = "runtime-74") -> RuntimeLease:
    return RuntimeLease(
        runtime_id=runtime_id,
        owner_token="owner-a",
        fencing_token=5,
        version=8,
        status="ACQUIRED",
        acquired=True,
        lease_until="2026-09-23T12:00:00Z",
    )


def _governed(risk: StoredOperationalRiskLedger) -> GovernedShadowExecution:
    receipt_id = risk.ledger.entries[-1].receipt.receipt_id
    cycle = ShadowExecutionCycle(
        source_plan_id="plan-phase74",
        items=(),
        initial_available_cash_usd=1000.0,
        reserved_new_risk_cash_usd=0.0,
        remaining_unreserved_cash_usd=1000.0,
        denied_assets=(),
        pending_reversal_assets=(),
        cycle_id="c" * 64,
    )
    return GovernedShadowExecution(
        operational_risk_receipt_id=receipt_id,
        trading_state="ACTIVE",
        blocked_new_risk_assets=(),
        policy_fingerprint="p" * 64,
        cycle=cycle,
        result_id="g" * 64,
    )


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


def _success_row(risk: StoredOperationalRiskLedger, governed: GovernedShadowExecution):
    return {
        "bound": True,
        "duplicate": False,
        "status": "BOUND",
        "runtime_id": "runtime-74",
        "cycle_id": governed.cycle.cycle_id,
        "runtime_version": 8,
        "risk_version": risk.version,
        "fencing_token": 5,
        "risk_ledger_hash": risk.ledger_hash,
        "risk_receipt_id": governed.operational_risk_receipt_id,
        "governed_result_id": governed.result_id,
        "policy_fingerprint": governed.policy_fingerprint,
    }


def test_bind_sends_exact_runtime_and_persisted_risk_anchors() -> None:
    risk = _risk()
    governed = _governed(risk)
    rpc = FakeRpc({
        "brian_bind_governed_shadow_cycle": _success_row(risk, governed),
    })
    receipt = GovernedCycleBindingStore(rpc).bind(
        _lease(),
        expected_runtime_version=8,
        risk=risk,
        governed=governed,
    )

    assert receipt.bound is True
    assert receipt.duplicate is False
    assert receipt.status == "BOUND"
    assert receipt.runtime_version == 8
    assert receipt.risk_version == 3

    params = rpc.calls[0][1]
    assert params["p_expected_runtime_version"] == 8
    assert params["p_risk_version"] == risk.version
    assert params["p_risk_ledger_hash"] == risk.ledger_hash
    assert params["p_risk_receipt_id"] == governed.operational_risk_receipt_id
    assert params["p_cycle_id"] == governed.cycle.cycle_id
    assert params["p_governed_result_id"] == governed.result_id
    assert params["p_policy_fingerprint"] == governed.policy_fingerprint


def test_bind_rejects_governed_cycle_from_nonpersisted_risk_receipt() -> None:
    risk = _risk()
    governed = _governed(risk)
    forged = GovernedShadowExecution(
        operational_risk_receipt_id="0" * 64,
        trading_state=governed.trading_state,
        blocked_new_risk_assets=governed.blocked_new_risk_assets,
        policy_fingerprint=governed.policy_fingerprint,
        cycle=governed.cycle,
        result_id=governed.result_id,
    )
    rpc = FakeRpc({"brian_bind_governed_shadow_cycle": {}})
    with pytest.raises(GovernedCycleBindingError, match="persisted risk head"):
        GovernedCycleBindingStore(rpc).bind(
            _lease(),
            expected_runtime_version=8,
            risk=risk,
            governed=forged,
        )
    assert rpc.calls == []


def test_bind_rejects_risk_ledger_from_different_runtime() -> None:
    risk = _risk(runtime_id="other-runtime")
    governed = _governed(risk)
    with pytest.raises(GovernedCycleBindingError, match="runtime_id"):
        GovernedCycleBindingStore(FakeRpc({})).bind(
            _lease(),
            expected_runtime_version=8,
            risk=risk,
            governed=governed,
        )


@pytest.mark.parametrize(
    "status",
    ["LEASE_LOST", "RUNTIME_VERSION_CONFLICT", "RISK_VERSION_CONFLICT"],
)
def test_fail_closed_database_statuses_cannot_be_misread_as_bound(status: str) -> None:
    risk = _risk()
    governed = _governed(risk)
    rpc = FakeRpc({
        "brian_bind_governed_shadow_cycle": {
            "bound": False,
            "duplicate": False,
            "status": status,
            "runtime_id": "runtime-74",
            "cycle_id": governed.cycle.cycle_id,
            "runtime_version": 9,
            "risk_version": risk.version + 1,
            "fencing_token": 5,
        }
    })
    receipt = GovernedCycleBindingStore(rpc).bind(
        _lease(),
        expected_runtime_version=8,
        risk=risk,
        governed=governed,
    )
    assert receipt.bound is False
    assert receipt.status == status


def test_duplicate_binding_requires_and_validates_same_evidence_anchors() -> None:
    risk = _risk()
    governed = _governed(risk)
    row = _success_row(risk, governed)
    row["status"] = "DUPLICATE"
    row["duplicate"] = True
    rpc = FakeRpc({"brian_bind_governed_shadow_cycle": row})

    receipt = GovernedCycleBindingStore(rpc).bind(
        _lease(),
        expected_runtime_version=8,
        risk=risk,
        governed=governed,
    )
    assert receipt.bound is True
    assert receipt.duplicate is True
    assert receipt.status == "DUPLICATE"

    forged = copy.deepcopy(row)
    forged["policy_fingerprint"] = "0" * 64
    with pytest.raises(GovernedCycleBindingError, match="policy_fingerprint"):
        GovernedCycleBindingStore(
            FakeRpc({"brian_bind_governed_shadow_cycle": forged})
        ).bind(
            _lease(),
            expected_runtime_version=8,
            risk=risk,
            governed=governed,
        )


def test_load_validates_stored_binding_shadow_boundary_and_hashes() -> None:
    risk = _risk()
    governed = _governed(risk)
    rpc = FakeRpc({
        "brian_read_governed_cycle_binding": {
            "runtime_id": "runtime-74",
            "cycle_id": governed.cycle.cycle_id,
            "governed_result_id": governed.result_id,
            "policy_fingerprint": governed.policy_fingerprint,
            "risk_version": risk.version,
            "risk_ledger_hash": risk.ledger_hash,
            "risk_receipt_id": governed.operational_risk_receipt_id,
            "runtime_version_before": 8,
            "fencing_token": 5,
            "bound_at": "2026-09-23T12:00:00Z",
            "shadow_only": True,
            "live_execution": False,
        }
    })
    binding = GovernedCycleBindingStore(rpc).load(
        runtime_id="runtime-74",
        cycle_id=governed.cycle.cycle_id,
    )
    assert binding is not None
    assert binding.risk_receipt_id == governed.operational_risk_receipt_id
    assert binding.runtime_version_before == 8

    bad = copy.deepcopy(rpc.responses["brian_read_governed_cycle_binding"])
    bad["live_execution"] = True
    with pytest.raises(GovernedCycleBindingError, match="live boundary"):
        GovernedCycleBindingStore(
            FakeRpc({"brian_read_governed_cycle_binding": bad})
        ).load(
            runtime_id="runtime-74",
            cycle_id=governed.cycle.cycle_id,
        )


def test_empty_risk_ledger_cannot_authorize_cycle() -> None:
    policy = OperationalRiskPolicy()
    ledger = OperationalRiskLedger(policy)
    manifest = ledger.manifest()
    risk = StoredOperationalRiskLedger(
        runtime_id="runtime-74",
        version=1,
        ledger=ledger,
        ledger_hash=str(manifest["ledger_hash"]),
        policy_hash=str(manifest["policy_hash"]),
        head_entry_id=None,
        current_state="ACTIVE",
        halt_latched=False,
    )
    cycle = ShadowExecutionCycle(
        source_plan_id="empty-risk",
        items=(),
        initial_available_cash_usd=1000.0,
        reserved_new_risk_cash_usd=0.0,
        remaining_unreserved_cash_usd=1000.0,
        denied_assets=(),
        pending_reversal_assets=(),
        cycle_id="e" * 64,
    )
    governed = GovernedShadowExecution(
        operational_risk_receipt_id="r" * 64,
        trading_state="ACTIVE",
        blocked_new_risk_assets=(),
        policy_fingerprint="p" * 64,
        cycle=cycle,
        result_id="g" * 64,
    )
    with pytest.raises(GovernedCycleBindingError, match="no persisted receipt"):
        GovernedCycleBindingStore(FakeRpc({})).bind(
            _lease(),
            expected_runtime_version=8,
            risk=risk,
            governed=governed,
        )
