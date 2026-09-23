from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase104_recovery_first_grounded_worker import (
    PrefetchedGroundedCycle,
    RecoveryFirstGroundedWorker,
    RecoveryFirstGroundedWorkerError,
)


RUNTIME = "runtime-104"


def _startup(*, ready: bool, status: str):
    return SimpleNamespace(
        runtime_id=RUNTIME,
        ready_for_normal_shadow=ready,
        status=status,
    )


def _bundle():
    return PrefetchedGroundedCycle(
        bundle_ref="bundle-104",
        asset_inputs={"BTCUSDT": "asset-input"},
        timestamp=1_790_000_000.0,
        model_weights={"market_snapshot_analyst": 1.0},
        returns_by_asset={"BTCUSDT": (0.01, -0.01, 0.02)},
        config="phase54-config",
        expected_edge_bps_by_asset={"BTCUSDT": 30.0},
        max_slippage_bps=20.0,
        ttl_seconds=60,
        markets={"BTCUSDT": "market"},
        risk_limits_by_asset={"BTCUSDT": "risk-limit"},
        marks={"BTCUSDT": 101.0},
        observed_at=1_790_000_001.0,
        source_ref="phase104:test",
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


class _CycleRunner:
    def __init__(self, worker, result_status="SHADOW_EXECUTED"):
        self.worker = worker
        self.calls = []
        self.result_status = result_status

    def process(self, asset_inputs, **kwargs):
        self.calls.append((asset_inputs, kwargs))
        return SimpleNamespace(
            runtime_id=RUNTIME,
            status=self.result_status,
            shadow_only=True,
            live_execution=False,
        )


def _run(worker, *, prefetch_provider, cycle_factory, **overrides):
    values = dict(
        prefetch_provider=prefetch_provider,
        recovery_max_items=4,
        recovery_worker_token="recovery-104",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        recovery_source_ref="phase104:recovery",
        normal_worker_token="normal-104",
        normal_claim_seconds=45,
        clock=lambda: 100.0,
    )
    values.update(overrides)
    return RecoveryFirstGroundedWorker(
        worker=worker,
        cycle_factory=cycle_factory,
    ).run_once(**values)


def test_recovery_blocked_returns_before_prefetch() -> None:
    worker = _Worker([
        _startup(ready=False, status="RECOVERY_BLOCKED"),
    ])
    prefetch_calls = 0
    cycle_calls = 0

    def prefetch():
        nonlocal prefetch_calls
        prefetch_calls += 1
        raise AssertionError("prefetch must not run")

    def cycle_factory(**kwargs):
        nonlocal cycle_calls
        cycle_calls += 1
        raise AssertionError("cycle must not be built")

    receipt = _run(
        worker,
        prefetch_provider=prefetch,
        cycle_factory=cycle_factory,
    )

    assert receipt.status == "RECOVERY_BLOCKED"
    assert receipt.prefetched is False
    assert receipt.bundle_ref is None
    assert receipt.cycle is None
    assert prefetch_calls == 0
    assert cycle_calls == 0
    assert len(worker.recovery_calls) == 1


def test_ready_recovery_prefetches_once_then_runs_phase102_on_same_worker() -> None:
    worker = _Worker([
        _startup(ready=True, status="READY_FOR_NORMAL_SHADOW"),
    ])
    bundle = _bundle()
    events = []
    cycle_runner = _CycleRunner(worker)

    def prefetch():
        events.append("prefetch")
        assert worker.ready_for_normal_shadow is True
        return bundle

    def cycle_factory(*, worker: object):
        events.append("cycle_factory")
        assert worker is globals_worker
        return cycle_runner

    globals_worker = worker
    receipt = _run(
        worker,
        prefetch_provider=prefetch,
        cycle_factory=cycle_factory,
    )

    assert events == ["prefetch", "cycle_factory"]
    assert receipt.status == "SHADOW_EXECUTED"
    assert receipt.prefetched is True
    assert receipt.bundle_ref == "bundle-104"
    assert receipt.cycle.status == "SHADOW_EXECUTED"
    assert len(cycle_runner.calls) == 1

    asset_inputs, kwargs = cycle_runner.calls[0]
    assert asset_inputs == {"BTCUSDT": "asset-input"}
    assert kwargs["timestamp"] == pytest.approx(1_790_000_000.0)
    assert kwargs["worker_token"] == "normal-104"
    assert kwargs["claim_seconds"] == 45
    assert kwargs["source_ref"] == "phase104:test"
    assert worker.recovery_calls[0]["recovery_worker_token"] == "recovery-104"
    assert worker.recovery_calls[0]["recovery_claim_seconds"] == 30


def test_already_ready_worker_does_not_rerun_recovery_gate() -> None:
    worker = _Worker()
    worker.ready_for_normal_shadow = True
    worker.startup = _startup(
        ready=True,
        status="READY_FOR_NORMAL_SHADOW",
    )
    cycle_runner = _CycleRunner(worker, result_status="HOLD_CURRENT_BOOK")

    receipt = _run(
        worker,
        prefetch_provider=_bundle,
        cycle_factory=lambda **kwargs: cycle_runner,
    )

    assert worker.recovery_calls == []
    assert receipt.status == "HOLD_CURRENT_BOOK"
    assert receipt.prefetched is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"normal_worker_token": ""}, "normal_worker_token"),
        ({"normal_claim_seconds": 9}, "normal_claim_seconds"),
        ({"normal_claim_seconds": 301}, "normal_claim_seconds"),
    ],
)
def test_invalid_normal_claim_inputs_fail_before_recovery_or_prefetch(
    overrides,
    message,
) -> None:
    worker = _Worker([
        _startup(ready=True, status="READY_FOR_NORMAL_SHADOW"),
    ])
    prefetch_calls = 0

    def prefetch():
        nonlocal prefetch_calls
        prefetch_calls += 1
        return _bundle()

    with pytest.raises(ValueError, match=message):
        _run(
            worker,
            prefetch_provider=prefetch,
            cycle_factory=lambda **kwargs: _CycleRunner(worker),
            **overrides,
        )

    assert worker.recovery_calls == []
    assert prefetch_calls == 0


def test_invalid_prefetch_bundle_type_fails_after_recovery_without_cycle() -> None:
    worker = _Worker([
        _startup(ready=True, status="READY_FOR_NORMAL_SHADOW"),
    ])
    cycle_calls = 0

    def cycle_factory(**kwargs):
        nonlocal cycle_calls
        cycle_calls += 1
        return _CycleRunner(worker)

    with pytest.raises(
        RecoveryFirstGroundedWorkerError,
        match="invalid bundle type",
    ):
        _run(
            worker,
            prefetch_provider=lambda: {"not": "typed"},
            cycle_factory=cycle_factory,
        )

    assert len(worker.recovery_calls) == 1
    assert cycle_calls == 0


def test_prefetch_failure_propagates_without_false_cycle_receipt() -> None:
    worker = _Worker([
        _startup(ready=True, status="READY_FOR_NORMAL_SHADOW"),
    ])
    cycle_calls = 0

    def cycle_factory(**kwargs):
        nonlocal cycle_calls
        cycle_calls += 1
        return _CycleRunner(worker)

    with pytest.raises(RuntimeError, match="prefetch unavailable"):
        _run(
            worker,
            prefetch_provider=lambda: (_ for _ in ()).throw(
                RuntimeError("prefetch unavailable")
            ),
            cycle_factory=cycle_factory,
        )

    assert len(worker.recovery_calls) == 1
    assert cycle_calls == 0


def test_from_env_owned_worker_is_closed_by_phase104_context() -> None:
    worker = _Worker()
    seen = {}

    def worker_factory(**kwargs):
        seen.update(kwargs)
        return worker

    with RecoveryFirstGroundedWorker.from_env(
        env={"BRIAN_RUNTIME_ID": RUNTIME},
        initial_runtime="initial",
        foreign_cycle_aborter="aborter",
        client="client",
        worker_factory=worker_factory,
        cycle_factory=lambda **kwargs: _CycleRunner(worker),
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


def test_external_worker_is_not_closed_by_phase104_wrapper() -> None:
    worker = _Worker()
    service = RecoveryFirstGroundedWorker(worker=worker)

    assert service.close() is True
    assert service.close() is True
    assert service.closed is True
    assert worker.close_calls == 0
    assert worker.closed is False


def test_constructor_failure_closes_owned_worker_opened_from_env() -> None:
    worker = _Worker()

    with pytest.raises(TypeError, match="cycle_factory"):
        RecoveryFirstGroundedWorker.from_env(
            env={"BRIAN_RUNTIME_ID": RUNTIME},
            worker_factory=lambda **kwargs: worker,
            cycle_factory=None,
        )

    assert worker.close_calls == 1
    assert worker.closed is True


def test_closed_phase104_worker_rejects_run() -> None:
    worker = _Worker()
    service = RecoveryFirstGroundedWorker(worker=worker)
    service.close()

    with pytest.raises(RecoveryFirstGroundedWorkerError, match="closed"):
        service.run_once(
            prefetch_provider=_bundle,
            recovery_max_items=1,
            recovery_worker_token="recovery-104",
            recovery_claim_seconds=30,
            recovery_ttl_seconds=60,
            recovery_source_ref="phase104:recovery",
            normal_worker_token="normal-104",
            normal_claim_seconds=30,
        )
