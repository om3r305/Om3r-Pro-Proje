from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase71_persisted_runtime_supervisor import PersistedRuntimeStaleError
from brian2026.phase101_integrated_decision_shadow_runtime import (
    IntegratedDecisionShadowRuntime,
    IntegratedDecisionShadowRuntimeError,
)


RUNTIME = "runtime-101"
PIPELINE = "p" * 64
RISK_ID = "r" * 64
RISK_ENTRY = "e" * 64
GOVERNED_ID = "g" * 64


class _RiskReceipt:
    def __init__(self, *, receipt_id=RISK_ID, trading_state="ACTIVE", valid=True):
        self.receipt_id = receipt_id
        self.trading_state = trading_state
        self._valid = valid

    def verify_identity(self):
        return self._valid


def _stored_risk(*, valid=True, state="ACTIVE", entries=True):
    receipt = _RiskReceipt(trading_state=state, valid=valid)
    rows = (
        (SimpleNamespace(entry_id=RISK_ENTRY, receipt=receipt),)
        if entries
        else ()
    )
    return SimpleNamespace(
        version=7,
        head_entry_id=RISK_ENTRY if entries else None,
        current_state=state,
        ledger=SimpleNamespace(entries=rows),
    )


class _RiskStore:
    def __init__(self, stored):
        self.stored = stored
        self.calls = []

    def load(self, *, runtime_id):
        self.calls.append(runtime_id)
        return self.stored


class _Worker:
    def __init__(self):
        self.runtime_id = RUNTIME
        self.closed = False
        self.ready_for_normal_shadow = True
        self.session = SimpleNamespace(rpc=lambda name, params: {})
        self.calls = []
        self.error = None

    def process_governed_cycle(self, governed, **kwargs):
        self.calls.append((governed, kwargs))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            runtime_id=RUNTIME,
            status="NORMAL_SHADOW_PROCESSED",
            shadow_only=True,
            live_execution=False,
        )


def _decision(
    *,
    status="REBALANCE_PLANNED",
    timestamp=100.0,
    shadow_only=True,
    live_execution=False,
    automatic_promotion=False,
):
    btc_claims = (
        SimpleNamespace(
            support_evidence_ids=("btc-2", "btc-1"),
        ),
        SimpleNamespace(
            support_evidence_ids=("btc-1", "btc-3"),
        ),
    )
    eth_claims = (
        SimpleNamespace(
            support_evidence_ids=("eth-1",),
        ),
    )
    return SimpleNamespace(
        status=status,
        timestamp=timestamp,
        pipeline_id=PIPELINE,
        shadow_only=shadow_only,
        live_execution=live_execution,
        automatic_promotion=automatic_promotion,
        current_weights={"BTCUSDT": 0.10},
        portfolio_book=SimpleNamespace(
            blend=SimpleNamespace(
                convictions={
                    "BTCUSDT": -0.65,
                    "ETHUSDT": 0.40,
                }
            )
        ),
        asset_results={
            "BTCUSDT": SimpleNamespace(analyst_claims=btc_claims),
            "ETHUSDT": SimpleNamespace(analyst_claims=eth_claims),
        },
    )


def _governed(*, risk_id=RISK_ID, items=(object(),)):
    return SimpleNamespace(
        operational_risk_receipt_id=risk_id,
        result_id=GOVERNED_ID,
        cycle=SimpleNamespace(items=tuple(items)),
        shadow_only=True,
        live_execution=False,
    )


def _process(runtime, decision=None, **overrides):
    values = dict(
        expected_edge_bps_by_asset={
            "BTCUSDT": 35.0,
            "ETHUSDT": 25.0,
        },
        max_slippage_bps=20.0,
        ttl_seconds=60,
        equity_usd=1000.0,
        available_cash_usd=500.0,
        markets={
            "BTCUSDT": SimpleNamespace(reference_price=100.0),
            "ETHUSDT": SimpleNamespace(reference_price=50.0),
        },
        risk_limits_by_asset={},
        marks={
            "BTCUSDT": 100.5,
            "ETHUSDT": 50.5,
        },
        worker_token="worker-101",
        claim_seconds=45,
        observed_at=101.0,
        source_ref="phase101:test",
    )
    values.update(overrides)
    return runtime.process_integrated_decision(
        _decision() if decision is None else decision,
        **values,
    )


@pytest.mark.parametrize(
    "status",
    ["WAIT_NO_GROUNDED_SIGNALS", "HOLD_CURRENT_BOOK"],
)
def test_wait_or_hold_decision_has_zero_risk_or_execution_side_effects(status) -> None:
    worker = _Worker()
    risk_store = _RiskStore(_stored_risk())
    governed_calls = 0

    def governed_runner(*args, **kwargs):
        nonlocal governed_calls
        governed_calls += 1
        raise AssertionError("governed runner must not execute")

    runtime = IntegratedDecisionShadowRuntime(
        worker=worker,
        risk_store=risk_store,
        governed_runner=governed_runner,
    )
    receipt = _process(runtime, _decision(status=status))

    assert receipt.executed is False
    assert receipt.status == status
    assert receipt.risk_version is None
    assert receipt.risk_receipt_id is None
    assert receipt.governed_result_id is None
    assert receipt.execution is None
    assert risk_store.calls == []
    assert governed_calls == 0
    assert worker.calls == []


def test_rebalance_uses_persisted_risk_and_phase54_grounded_metadata() -> None:
    worker = _Worker()
    stored = _stored_risk()
    risk_store = _RiskStore(stored)
    seen = {}

    def governed_runner(decision, **kwargs):
        seen["decision"] = decision
        seen.update(kwargs)
        return _governed()

    runtime = IntegratedDecisionShadowRuntime(
        worker=worker,
        risk_store=risk_store,
        governed_runner=governed_runner,
    )
    decision = _decision()
    receipt = _process(runtime, decision)

    assert risk_store.calls == [RUNTIME]
    assert seen["decision"] is decision
    assert seen["operational_risk"] is stored.ledger.entries[-1].receipt
    assert seen["expected_edge_bps_by_asset"] == {
        "BTCUSDT": 35.0,
        "ETHUSDT": 25.0,
    }
    assert seen["confidence_by_asset"] == {
        "BTCUSDT": pytest.approx(0.65),
        "ETHUSDT": pytest.approx(0.40),
    }
    assert seen["evidence_ids_by_asset"] == {
        "BTCUSDT": ("btc-1", "btc-2", "btc-3"),
        "ETHUSDT": ("eth-1",),
    }
    assert seen["max_slippage_bps"] == pytest.approx(20.0)
    assert seen["ttl_seconds"] == 60
    assert seen["equity_usd"] == pytest.approx(1000.0)
    assert seen["available_cash_usd"] == pytest.approx(500.0)

    assert len(worker.calls) == 1
    governed, kwargs = worker.calls[0]
    assert governed.result_id == GOVERNED_ID
    assert kwargs == {
        "worker_token": "worker-101",
        "claim_seconds": 45,
        "marks": {
            "BTCUSDT": 100.5,
            "ETHUSDT": 50.5,
        },
        "observed_at": 101.0,
        "source_ref": "phase101:test",
    }

    assert receipt.executed is True
    assert receipt.status == "SHADOW_EXECUTED"
    assert receipt.risk_version == 7
    assert receipt.risk_receipt_id == RISK_ID
    assert receipt.governed_result_id == GOVERNED_ID
    assert receipt.execution.status == "NORMAL_SHADOW_PROCESSED"
    assert receipt.shadow_only is True
    assert receipt.live_execution is False


def test_rebalance_with_zero_executable_items_does_not_enter_phase100_execution() -> None:
    worker = _Worker()
    risk_store = _RiskStore(_stored_risk())
    runtime = IntegratedDecisionShadowRuntime(
        worker=worker,
        risk_store=risk_store,
        governed_runner=lambda *args, **kwargs: _governed(items=()),
    )

    receipt = _process(runtime)

    assert receipt.executed is False
    assert receipt.status == "NO_EXECUTABLE_INSTRUCTIONS"
    assert receipt.risk_version == 7
    assert receipt.risk_receipt_id == RISK_ID
    assert receipt.governed_result_id == GOVERNED_ID
    assert receipt.execution is None
    assert worker.calls == []


@pytest.mark.parametrize(
    ("stored", "message"),
    [
        (None, "persisted operational-risk head"),
        (_stored_risk(entries=False), "no head receipt"),
        (_stored_risk(valid=False), "receipt identity mismatch"),
    ],
)
def test_missing_or_invalid_persisted_risk_fails_before_governance(stored, message) -> None:
    worker = _Worker()
    risk_store = _RiskStore(stored)
    governed_calls = 0

    def governed_runner(*args, **kwargs):
        nonlocal governed_calls
        governed_calls += 1
        raise AssertionError("governed runner must not execute")

    runtime = IntegratedDecisionShadowRuntime(
        worker=worker,
        risk_store=risk_store,
        governed_runner=governed_runner,
    )

    with pytest.raises(IntegratedDecisionShadowRuntimeError, match=message):
        _process(runtime)

    assert governed_calls == 0
    assert worker.calls == []


def test_persisted_risk_state_drift_fails_closed() -> None:
    stored = _stored_risk(state="ACTIVE")
    stored.current_state = "REDUCING"
    runtime = IntegratedDecisionShadowRuntime(
        worker=_Worker(),
        risk_store=_RiskStore(stored),
        governed_runner=lambda *args, **kwargs: pytest.fail("must not run"),
    )

    with pytest.raises(
        IntegratedDecisionShadowRuntimeError,
        match="state differs",
    ):
        _process(runtime)


def test_phase69_result_must_bind_exact_persisted_risk_head() -> None:
    worker = _Worker()
    runtime = IntegratedDecisionShadowRuntime(
        worker=worker,
        risk_store=_RiskStore(_stored_risk()),
        governed_runner=lambda *args, **kwargs: _governed(risk_id="x" * 64),
    )

    with pytest.raises(
        IntegratedDecisionShadowRuntimeError,
        match="not bound to persisted risk head",
    ):
        _process(runtime)

    assert worker.calls == []


def test_phase75_risk_race_failure_propagates_without_false_phase101_success() -> None:
    worker = _Worker()
    worker.error = PersistedRuntimeStaleError("risk head changed before authorization")
    runtime = IntegratedDecisionShadowRuntime(
        worker=worker,
        risk_store=_RiskStore(_stored_risk()),
        governed_runner=lambda *args, **kwargs: _governed(),
    )

    with pytest.raises(PersistedRuntimeStaleError, match="risk head changed"):
        _process(runtime)

    assert len(worker.calls) == 1


@pytest.mark.parametrize(
    ("decision", "message"),
    [
        (_decision(status="UNKNOWN"), "unsupported Phase54 decision status"),
        (_decision(shadow_only=False), "shadow-only boundary"),
        (_decision(live_execution=True), "shadow-only boundary"),
        (_decision(automatic_promotion=True), "automatic promotion"),
    ],
)
def test_invalid_phase54_decision_contract_is_rejected(decision, message) -> None:
    runtime = IntegratedDecisionShadowRuntime(
        worker=_Worker(),
        risk_store=_RiskStore(_stored_risk()),
        governed_runner=lambda *args, **kwargs: pytest.fail("must not run"),
    )

    with pytest.raises(IntegratedDecisionShadowRuntimeError, match=message):
        _process(runtime, decision)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"observed_at": 99.0}, "cannot precede"),
        ({"worker_token": ""}, "worker_token"),
        ({"claim_seconds": 9}, "claim_seconds"),
        ({"claim_seconds": 301}, "claim_seconds"),
        ({"source_ref": ""}, "source_ref"),
        ({"max_slippage_bps": float("nan")}, "max_slippage_bps"),
        ({"ttl_seconds": 0}, "ttl_seconds"),
        (
            {"expected_edge_bps_by_asset": {"BTCUSDT": float("nan")}},
            "expected_edge_bps_by_asset",
        ),
        (
            {"marks": {"BTCUSDT": float("inf")}},
            "marks",
        ),
    ],
)
def test_invalid_execution_inputs_fail_before_risk_load(overrides, message) -> None:
    risk_store = _RiskStore(_stored_risk())
    worker = _Worker()
    runtime = IntegratedDecisionShadowRuntime(
        worker=worker,
        risk_store=risk_store,
        governed_runner=lambda *args, **kwargs: pytest.fail("must not run"),
    )

    with pytest.raises(ValueError, match=message):
        _process(runtime, **overrides)

    assert risk_store.calls == []
    assert worker.calls == []


def test_constructor_requires_ready_open_phase100_worker() -> None:
    worker = _Worker()
    worker.ready_for_normal_shadow = False

    with pytest.raises(
        IntegratedDecisionShadowRuntimeError,
        match="has not released",
    ):
        IntegratedDecisionShadowRuntime(
            worker=worker,
            risk_store=_RiskStore(_stored_risk()),
        )


def test_worker_closing_after_constructor_blocks_future_decisions() -> None:
    worker = _Worker()
    runtime = IntegratedDecisionShadowRuntime(
        worker=worker,
        risk_store=_RiskStore(_stored_risk()),
        governed_runner=lambda *args, **kwargs: pytest.fail("must not run"),
    )
    worker.closed = True

    with pytest.raises(
        IntegratedDecisionShadowRuntimeError,
        match="closed before integrated decision",
    ):
        _process(runtime)
