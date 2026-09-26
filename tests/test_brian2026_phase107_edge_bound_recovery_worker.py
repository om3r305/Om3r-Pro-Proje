from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase105_lagged_prospective_edge import LaggedReliabilityEvidence
from brian2026.phase106_decision_bound_lagged_edge import (
    AssetLaggedEdgeContext,
    DecisionBoundLaggedEdgeRuntime,
)
from brian2026.phase107_edge_bound_recovery_worker import (
    EdgeBoundRecoveryWorker,
    EdgeBoundRecoveryWorkerError,
    PrefetchedLaggedEdgeGroundedCycle,
)


RUNTIME = "runtime-107"
TS = 1_790_000_000.0


def _startup(*, ready: bool, status: str):
    return SimpleNamespace(
        runtime_id=RUNTIME,
        ready_for_normal_shadow=ready,
        status=status,
    )


def _edge_context():
    return AssetLaggedEdgeContext(
        reliability=(
            LaggedReliabilityEvidence(
                group="price_structure",
                sample_count=250,
                bayesian_hit_rate=0.62,
                avg_signed_bps=30.0,
                avg_cost_adjusted_signed_bps=25.0,
                outcome_horizon_seconds=900,
                snapshot_window_end=TS - 3600,
                snapshot_generated_at=TS - 1800,
            ),
        ),
        round_trip_cost_bps=4.0,
        cost_observed_at=TS - 1,
    )


def _bundle():
    return PrefetchedLaggedEdgeGroundedCycle(
        bundle_ref="bundle-107",
        asset_inputs={"BTCUSDT": "asset-input"},
        timestamp=TS,
        model_weights={"market_snapshot_analyst": 1.0},
        returns_by_asset={"BTCUSDT": (0.01, -0.01, 0.02)},
        config="phase54-config",
        edge_contexts_by_asset={"BTCUSDT": _edge_context()},
        max_slippage_bps=20.0,
        ttl_seconds=60,
        markets={"BTCUSDT": "market"},
        risk_limits_by_asset={"BTCUSDT": "risk-limit"},
        marks={"BTCUSDT": 101.0},
        observed_at=TS + 1,
        source_ref="phase107:test",
    )


class _Worker:
    def __init__(self, startups=()):
        self.runtime_id = RUNTIME
        self.closed = False
        self.ready_for_normal_shadow = False
        self.startup = None
        self.startups = list(startups)
        self.recovery_calls = []
        self.close_calls = 0

    def run_recovery_gate(self, **kwargs):
        self.recovery_calls.append(kwargs)
        if not self.startups:
            raise AssertionError("unexpected recovery gate call")
        startup = self.startups.pop(0)
        self.startup = startup
        self.ready_for_normal_shadow = startup.ready_for_normal_shadow
        return startup

    def close(self):
        self.close_calls += 1
        self.closed = True
        return True


class _BaseRuntime:
    def __init__(self, worker):
        self.worker = worker
        self.calls = []

    def process_integrated_decision(self, decision, **kwargs):
        self.calls.append((decision, kwargs))
        return SimpleNamespace(status="SHADOW_EXECUTED")


class _CycleRunner:
    def __init__(self, worker, integrated_runtime):
        self.worker = worker
        self.integrated_runtime = integrated_runtime
        self.calls = []

    def process(self, asset_inputs, **kwargs):
        self.calls.append((asset_inputs, kwargs))
        return SimpleNamespace(
            runtime_id=RUNTIME,
            status="SHADOW_EXECUTED",
            shadow_only=True,
            live_execution=False,
        )


def _service(worker, seen):
    def base_runtime_factory(*, worker):
        seen["base_worker"] = worker
        base = _BaseRuntime(worker)
        seen["base"] = base
        return base

    def cycle_factory(*, worker, integrated_runtime):
        seen["cycle_worker"] = worker
        seen["integrated_runtime"] = integrated_runtime
        cycle = _CycleRunner(worker, integrated_runtime)
        seen["cycle"] = cycle
        return cycle

    return EdgeBoundRecoveryWorker(
        worker=worker,
        base_runtime_factory=base_runtime_factory,
        cycle_factory=cycle_factory,
    )


def _run(service, prefetch_provider):
    return service.run_once(
        prefetch_provider=prefetch_provider,
        recovery_max_items=4,
        recovery_worker_token="recovery-107",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        recovery_source_ref="phase107:recovery",
        normal_worker_token="normal-107",
        normal_claim_seconds=45,
        clock=lambda: 100.0,
    )


def test_recovery_blocked_stops_before_prefetch_and_edge_runtime() -> None:
    worker = _Worker([
        _startup(ready=False, status="RECOVERY_BLOCKED"),
    ])
    seen = {}
    service = _service(worker, seen)
    prefetch_calls = 0

    def prefetch():
        nonlocal prefetch_calls
        prefetch_calls += 1
        raise AssertionError("must not prefetch")

    receipt = _run(service, prefetch)

    assert receipt.status == "RECOVERY_BLOCKED"
    assert receipt.prefetched is False
    assert receipt.bundle_ref is None
    assert receipt.cycle is None
    assert prefetch_calls == 0
    assert seen == {}


def test_ready_recovery_builds_phase106_runtime_then_phase102_with_empty_raw_edge_map() -> None:
    worker = _Worker([
        _startup(ready=True, status="READY_FOR_NORMAL_SHADOW"),
    ])
    seen = {}
    service = _service(worker, seen)
    bundle = _bundle()

    receipt = _run(service, lambda: bundle)

    assert receipt.status == "SHADOW_EXECUTED"
    assert receipt.prefetched is True
    assert receipt.bundle_ref == "bundle-107"
    assert seen["base_worker"] is worker
    assert seen["cycle_worker"] is worker
    assert isinstance(seen["integrated_runtime"], DecisionBoundLaggedEdgeRuntime)
    assert seen["integrated_runtime"].base_runtime is seen["base"]
    assert set(seen["integrated_runtime"].contexts_by_asset) == {"BTCUSDT"}

    asset_inputs, kwargs = seen["cycle"].calls[0]
    assert asset_inputs == {"BTCUSDT": "asset-input"}
    assert kwargs["expected_edge_bps_by_asset"] == {}
    assert kwargs["worker_token"] == "normal-107"
    assert kwargs["claim_seconds"] == 45
    assert worker.recovery_calls[0]["recovery_worker_token"] == "recovery-107"
    assert worker.recovery_calls[0]["recovery_claim_seconds"] == 30


def test_already_ready_worker_does_not_rerun_recovery() -> None:
    worker = _Worker()
    worker.ready_for_normal_shadow = True
    worker.startup = _startup(
        ready=True,
        status="READY_FOR_NORMAL_SHADOW",
    )
    seen = {}
    service = _service(worker, seen)

    receipt = _run(service, _bundle)

    assert receipt.prefetched is True
    assert worker.recovery_calls == []
    assert len(seen["cycle"].calls) == 1


def test_invalid_bundle_type_fails_after_ready_without_cycle_factory() -> None:
    worker = _Worker([
        _startup(ready=True, status="READY_FOR_NORMAL_SHADOW"),
    ])
    seen = {}
    service = _service(worker, seen)

    with pytest.raises(
        EdgeBoundRecoveryWorkerError,
        match="invalid Phase107 bundle type",
    ):
        _run(service, lambda: {"not": "typed"})

    assert len(worker.recovery_calls) == 1
    assert seen == {}


@pytest.mark.parametrize(
    ("token", "seconds", "message"),
    [
        ("", 30, "normal_worker_token"),
        ("normal", 9, "normal_claim_seconds"),
        ("normal", 301, "normal_claim_seconds"),
    ],
)
def test_invalid_normal_claim_fails_before_recovery_and_prefetch(
    token,
    seconds,
    message,
) -> None:
    worker = _Worker([
        _startup(ready=True, status="READY_FOR_NORMAL_SHADOW"),
    ])
    service = _service(worker, {})
    prefetch_calls = 0

    def prefetch():
        nonlocal prefetch_calls
        prefetch_calls += 1
        return _bundle()

    with pytest.raises(ValueError, match=message):
        service.run_once(
            prefetch_provider=prefetch,
            recovery_max_items=4,
            recovery_worker_token="recovery-107",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            recovery_source_ref="phase107:recovery",
            normal_worker_token=token,
            normal_claim_seconds=seconds,
        )

    assert worker.recovery_calls == []
    assert prefetch_calls == 0


def test_owned_from_env_worker_is_closed_on_context_exit() -> None:
    worker = _Worker()
    seen = {}

    def worker_factory(**kwargs):
        seen.update(kwargs)
        return worker

    with EdgeBoundRecoveryWorker.from_env(
        env={"BRIAN_RUNTIME_ID": RUNTIME},
        initial_runtime="initial",
        foreign_cycle_aborter="aborter",
        client="client",
        worker_factory=worker_factory,
        base_runtime_factory=lambda **kwargs: _BaseRuntime(worker),
        cycle_factory=lambda **kwargs: _CycleRunner(
            worker,
            kwargs["integrated_runtime"],
        ),
    ) as service:
        assert service.closed is False
        assert worker.close_calls == 0

    assert service.closed is True
    assert worker.close_calls == 1
    assert seen == {
        "env": {"BRIAN_RUNTIME_ID": RUNTIME},
        "initial_runtime": "initial",
        "foreign_cycle_aborter": "aborter",
        "client": "client",
    }


def test_external_worker_is_not_closed_by_phase107_wrapper() -> None:
    worker = _Worker()
    service = EdgeBoundRecoveryWorker(worker=worker)

    assert service.close() is True
    assert service.close() is True
    assert worker.close_calls == 0
    assert worker.closed is False


def test_constructor_failure_closes_worker_opened_from_env() -> None:
    worker = _Worker()

    with pytest.raises(TypeError, match="base_runtime_factory"):
        EdgeBoundRecoveryWorker.from_env(
            env={"BRIAN_RUNTIME_ID": RUNTIME},
            worker_factory=lambda **kwargs: worker,
            base_runtime_factory=None,
        )

    assert worker.close_calls == 1
    assert worker.closed is True
