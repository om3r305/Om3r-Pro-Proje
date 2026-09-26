from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase102_grounded_decision_worker_cycle import (
    GroundedDecisionWorkerCycle,
    GroundedDecisionWorkerCycleError,
)


RUNTIME = "runtime-102"
STATE_ID = "s" * 64
PIPELINE = "p" * 64


def _head(
    *,
    state_id=STATE_ID,
    observed_at=100.0,
    equity=1200.0,
    cash=700.0,
    weights=(("BTCUSDT", 0.20), ("ETHUSDT", -0.10)),
):
    return SimpleNamespace(
        state_id=state_id,
        observed_at=observed_at,
        equity_usd=equity,
        available_cash_usd=cash,
        position_weights=tuple(weights),
    )


class _Worker:
    def __init__(self, head=None):
        self.runtime_id = RUNTIME
        self.closed = False
        self.ready_for_normal_shadow = True
        runtime = SimpleNamespace(
            ledger=SimpleNamespace(head_state=head or _head()),
        )
        self.session = SimpleNamespace(
            runtime_supervisor=SimpleNamespace(runtime=runtime),
        )


class _IntegratedRuntime:
    def __init__(self, worker):
        self.worker = worker
        self.calls = []
        self.result = SimpleNamespace(
            runtime_id=RUNTIME,
            decision_pipeline_id=PIPELINE,
            status="SHADOW_EXECUTED",
            shadow_only=True,
            live_execution=False,
        )

    def process_integrated_decision(self, decision, **kwargs):
        self.calls.append((decision, kwargs))
        return self.result


def _decision(
    *,
    timestamp=101.0,
    weights=None,
    shadow_only=True,
    live_execution=False,
    automatic_promotion=False,
    pipeline_id=PIPELINE,
):
    return SimpleNamespace(
        status="REBALANCE_PLANNED",
        timestamp=timestamp,
        current_weights=(
            {"BTCUSDT": 0.20, "ETHUSDT": -0.10}
            if weights is None
            else dict(weights)
        ),
        pipeline_id=pipeline_id,
        shadow_only=shadow_only,
        live_execution=live_execution,
        automatic_promotion=automatic_promotion,
    )


def _process(runtime, **overrides):
    values = dict(
        asset_inputs={"BTCUSDT": "prefetched-btc"},
        timestamp=101.0,
        model_weights={"news_analyst": 1.0},
        returns_by_asset={"BTCUSDT": (0.01, -0.01, 0.02)},
        config="phase54-config",
        expected_edge_bps_by_asset={"BTCUSDT": 30.0},
        max_slippage_bps=20.0,
        ttl_seconds=60,
        markets={"BTCUSDT": "market"},
        risk_limits_by_asset={"BTCUSDT": "risk-limit"},
        marks={"BTCUSDT": 101.0, "ETHUSDT": 49.0},
        worker_token="worker-102",
        claim_seconds=45,
        observed_at=102.0,
        source_ref="phase102:test",
    )
    values.update(overrides)
    return runtime.process(**values)


def test_phase102_runs_phase54_from_exact_phase60_account_head_then_phase101() -> None:
    worker = _Worker()
    integrated = _IntegratedRuntime(worker)
    seen = {}

    def decision_runner(asset_inputs, **kwargs):
        seen["asset_inputs"] = asset_inputs
        seen.update(kwargs)
        return _decision(timestamp=kwargs["timestamp"])

    runtime = GroundedDecisionWorkerCycle(
        worker=worker,
        integrated_runtime=integrated,
        decision_runner=decision_runner,
    )
    receipt = _process(runtime)

    assert seen["asset_inputs"] == {"BTCUSDT": "prefetched-btc"}
    assert seen["timestamp"] == pytest.approx(101.0)
    assert seen["model_weights"] == {"news_analyst": 1.0}
    assert seen["returns_by_asset"] == {
        "BTCUSDT": (0.01, -0.01, 0.02)
    }
    assert seen["config"] == "phase54-config"
    assert seen["current_weights"] == {
        "BTCUSDT": pytest.approx(0.20),
        "ETHUSDT": pytest.approx(-0.10),
    }

    assert len(integrated.calls) == 1
    decision, kwargs = integrated.calls[0]
    assert decision.pipeline_id == PIPELINE
    assert kwargs["equity_usd"] == pytest.approx(1200.0)
    assert kwargs["available_cash_usd"] == pytest.approx(700.0)
    assert kwargs["expected_edge_bps_by_asset"] == {"BTCUSDT": 30.0}
    assert kwargs["marks"] == {"BTCUSDT": 101.0, "ETHUSDT": 49.0}
    assert kwargs["worker_token"] == "worker-102"
    assert kwargs["claim_seconds"] == 45
    assert kwargs["observed_at"] == pytest.approx(102.0)
    assert kwargs["source_ref"] == "phase102:test"

    assert receipt.runtime_id == RUNTIME
    assert receipt.account_state_id == STATE_ID
    assert receipt.decision is decision
    assert receipt.execution is integrated.result
    assert receipt.status == "SHADOW_EXECUTED"
    assert receipt.shadow_only is True
    assert receipt.live_execution is False


def test_caller_cannot_override_current_weights_equity_or_cash() -> None:
    worker = _Worker(
        _head(
            equity=4321.0,
            cash=1234.0,
            weights=(("SOLUSDT", 0.15),),
        )
    )
    integrated = _IntegratedRuntime(worker)
    seen = {}

    def decision_runner(asset_inputs, **kwargs):
        seen.update(kwargs)
        return _decision(
            timestamp=kwargs["timestamp"],
            weights={"SOLUSDT": 0.15},
        )

    runtime = GroundedDecisionWorkerCycle(
        worker=worker,
        integrated_runtime=integrated,
        decision_runner=decision_runner,
    )
    _process(
        runtime,
        asset_inputs={"SOLUSDT": "prefetched-sol"},
        markets={"SOLUSDT": "market"},
        marks={"SOLUSDT": 120.0},
        expected_edge_bps_by_asset={"SOLUSDT": 25.0},
        risk_limits_by_asset={"SOLUSDT": "risk-limit"},
    )

    assert seen["current_weights"] == {"SOLUSDT": pytest.approx(0.15)}
    _, kwargs = integrated.calls[0]
    assert kwargs["equity_usd"] == pytest.approx(4321.0)
    assert kwargs["available_cash_usd"] == pytest.approx(1234.0)


def test_phase60_head_change_during_phase54_fails_before_phase101() -> None:
    worker = _Worker()
    integrated = _IntegratedRuntime(worker)

    def decision_runner(asset_inputs, **kwargs):
        worker.session.runtime_supervisor.runtime.ledger.head_state = _head(
            state_id="n" * 64,
        )
        return _decision(timestamp=kwargs["timestamp"])

    runtime = GroundedDecisionWorkerCycle(
        worker=worker,
        integrated_runtime=integrated,
        decision_runner=decision_runner,
    )

    with pytest.raises(
        GroundedDecisionWorkerCycleError,
        match="head changed while Phase54",
    ):
        _process(runtime)

    assert integrated.calls == []


def test_stale_decision_timestamp_is_rejected_before_phase54() -> None:
    worker = _Worker(_head(observed_at=105.0))
    integrated = _IntegratedRuntime(worker)
    calls = 0

    def decision_runner(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("must not run")

    runtime = GroundedDecisionWorkerCycle(
        worker=worker,
        integrated_runtime=integrated,
        decision_runner=decision_runner,
    )

    with pytest.raises(
        GroundedDecisionWorkerCycleError,
        match="predates authoritative",
    ):
        _process(runtime, timestamp=101.0)

    assert calls == 0
    assert integrated.calls == []


@pytest.mark.parametrize(
    ("decision", "message"),
    [
        (_decision(timestamp=102.0), "timestamp differs"),
        (
            _decision(weights={"BTCUSDT": 0.30, "ETHUSDT": -0.10}),
            "current_weights differ",
        ),
        (_decision(shadow_only=False), "shadow-only boundary"),
        (_decision(live_execution=True), "shadow-only boundary"),
        (_decision(automatic_promotion=True), "automatic promotion"),
        (_decision(pipeline_id="short"), "pipeline_id"),
    ],
)
def test_phase54_output_contract_drift_is_rejected(decision, message) -> None:
    worker = _Worker()
    integrated = _IntegratedRuntime(worker)
    runtime = GroundedDecisionWorkerCycle(
        worker=worker,
        integrated_runtime=integrated,
        decision_runner=lambda *args, **kwargs: decision,
    )

    with pytest.raises(GroundedDecisionWorkerCycleError, match=message):
        _process(runtime)

    assert integrated.calls == []


def test_phase101_wait_or_hold_status_is_preserved_by_phase102_receipt() -> None:
    worker = _Worker()
    integrated = _IntegratedRuntime(worker)
    integrated.result = SimpleNamespace(
        runtime_id=RUNTIME,
        decision_pipeline_id=PIPELINE,
        status="HOLD_CURRENT_BOOK",
        shadow_only=True,
        live_execution=False,
    )
    decision = _decision()
    decision.status = "HOLD_CURRENT_BOOK"
    runtime = GroundedDecisionWorkerCycle(
        worker=worker,
        integrated_runtime=integrated,
        decision_runner=lambda *args, **kwargs: decision,
    )

    receipt = _process(runtime)

    assert receipt.status == "HOLD_CURRENT_BOOK"
    assert receipt.execution.status == "HOLD_CURRENT_BOOK"


def test_constructor_rejects_closed_not_ready_or_cross_worker_runtime() -> None:
    closed = _Worker()
    closed.closed = True
    with pytest.raises(GroundedDecisionWorkerCycleError, match="closed"):
        GroundedDecisionWorkerCycle(worker=closed)

    blocked = _Worker()
    blocked.ready_for_normal_shadow = False
    with pytest.raises(GroundedDecisionWorkerCycleError, match="has not released"):
        GroundedDecisionWorkerCycle(worker=blocked)

    worker = _Worker()
    other = _Worker()
    with pytest.raises(
        GroundedDecisionWorkerCycleError,
        match="different Phase100 worker",
    ):
        GroundedDecisionWorkerCycle(
            worker=worker,
            integrated_runtime=_IntegratedRuntime(other),
        )


def test_missing_or_invalid_phase60_head_fails_closed() -> None:
    worker = _Worker()
    worker.session.runtime_supervisor.runtime.ledger.head_state = None
    integrated = _IntegratedRuntime(worker)
    runtime = GroundedDecisionWorkerCycle(
        worker=worker,
        integrated_runtime=integrated,
        decision_runner=lambda *args, **kwargs: pytest.fail("must not run"),
    )

    with pytest.raises(
        GroundedDecisionWorkerCycleError,
        match="does not expose authoritative",
    ):
        _process(runtime)

    worker = _Worker(_head(state_id="bad"))
    integrated = _IntegratedRuntime(worker)
    runtime = GroundedDecisionWorkerCycle(
        worker=worker,
        integrated_runtime=integrated,
        decision_runner=lambda *args, **kwargs: pytest.fail("must not run"),
    )
    with pytest.raises(
        GroundedDecisionWorkerCycleError,
        match="state_id is invalid",
    ):
        _process(runtime)


def test_worker_closing_after_constructor_blocks_cycle() -> None:
    worker = _Worker()
    integrated = _IntegratedRuntime(worker)
    runtime = GroundedDecisionWorkerCycle(
        worker=worker,
        integrated_runtime=integrated,
        decision_runner=lambda *args, **kwargs: pytest.fail("must not run"),
    )
    worker.closed = True

    with pytest.raises(
        GroundedDecisionWorkerCycleError,
        match="closed before Phase102",
    ):
        _process(runtime)


@pytest.mark.parametrize(
    ("timestamp", "observed_at", "message"),
    [
        (float("nan"), 102.0, "timestamp"),
        (101.0, float("inf"), "observed_at"),
    ],
)
def test_nonfinite_times_fail_before_phase54(timestamp, observed_at, message) -> None:
    worker = _Worker()
    integrated = _IntegratedRuntime(worker)
    calls = 0

    def decision_runner(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("must not run")

    runtime = GroundedDecisionWorkerCycle(
        worker=worker,
        integrated_runtime=integrated,
        decision_runner=decision_runner,
    )

    with pytest.raises(ValueError, match=message):
        _process(runtime, timestamp=timestamp, observed_at=observed_at)

    assert calls == 0
    assert integrated.calls == []
