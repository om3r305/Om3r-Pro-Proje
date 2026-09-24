from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase112_edge_bound_crypto_shadow_service import (
    EdgeBoundCryptoShadowService,
    EdgeBoundCryptoShadowServiceError,
)


class _Worker:
    def __init__(self):
        self.runtime_id = "runtime-112"
        self.closed = False
        self.calls = []
        self.close_calls = 0

    def run_once(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            runtime_id=self.runtime_id,
            status="SHADOW_EXECUTED",
            shadow_only=True,
            live_execution=False,
        )

    def close(self):
        self.close_calls += 1
        self.closed = True
        return True


class _Prefetch:
    def __init__(self):
        self.closed = False
        self.calls = 0
        self.close_calls = 0

    def __call__(self):
        self.calls += 1
        return SimpleNamespace(bundle_ref="bundle-112")

    def close(self):
        self.close_calls += 1
        self.closed = True
        return True


def _run(service):
    return service.run_once(
        recovery_max_items=4,
        recovery_worker_token="recovery-112",
        recovery_claim_seconds=30,
        recovery_ttl_seconds=60,
        recovery_source_ref="phase112:recovery",
        normal_worker_token="normal-112",
        normal_claim_seconds=45,
        recovery_provider_factory="recovery-provider-factory",
        clock=lambda: 100.0,
    )


def test_service_passes_phase111_as_only_normal_prefetch_provider() -> None:
    worker = _Worker()
    prefetch = _Prefetch()
    service = EdgeBoundCryptoShadowService(
        worker=worker,
        prefetch_provider=prefetch,
    )

    receipt = _run(service)

    assert receipt.status == "SHADOW_EXECUTED"
    assert service.runtime_id == "runtime-112"
    assert len(worker.calls) == 1
    call = worker.calls[0]
    assert call["prefetch_provider"] is prefetch
    assert call["recovery_max_items"] == 4
    assert call["recovery_worker_token"] == "recovery-112"
    assert call["recovery_claim_seconds"] == 30
    assert call["recovery_ttl_seconds"] == 60
    assert call["recovery_source_ref"] == "phase112:recovery"
    assert call["normal_worker_token"] == "normal-112"
    assert call["normal_claim_seconds"] == 45
    assert call["recovery_provider_factory"] == "recovery-provider-factory"
    assert callable(call["clock"])


def test_service_does_not_prefetch_outside_phase107_authority() -> None:
    worker = _Worker()
    prefetch = _Prefetch()
    service = EdgeBoundCryptoShadowService(
        worker=worker,
        prefetch_provider=prefetch,
    )

    _run(service)

    # Phase112 only hands the callable to Phase107. It never calls Phase111
    # itself, preserving Phase107's recovery-before-prefetch ordering.
    assert prefetch.calls == 0


def test_owned_service_closes_prefetch_then_worker_once() -> None:
    events = []

    class OrderedWorker(_Worker):
        def close(self):
            events.append("worker")
            return super().close()

    class OrderedPrefetch(_Prefetch):
        def close(self):
            events.append("prefetch")
            return super().close()

    worker = OrderedWorker()
    prefetch = OrderedPrefetch()
    service = EdgeBoundCryptoShadowService(
        worker=worker,
        prefetch_provider=prefetch,
        owns_components=True,
    )

    assert service.close() is True
    assert service.close() is True
    assert events == ["prefetch", "worker"]
    assert prefetch.close_calls == 1
    assert worker.close_calls == 1
    assert service.closed is True


def test_external_components_are_not_closed() -> None:
    worker = _Worker()
    prefetch = _Prefetch()
    service = EdgeBoundCryptoShadowService(
        worker=worker,
        prefetch_provider=prefetch,
        owns_components=False,
    )

    service.close()

    assert worker.close_calls == 0
    assert prefetch.close_calls == 0
    assert worker.closed is False
    assert prefetch.closed is False


def test_closed_service_rejects_run() -> None:
    service = EdgeBoundCryptoShadowService(
        worker=_Worker(),
        prefetch_provider=_Prefetch(),
    )
    service.close()

    with pytest.raises(
        EdgeBoundCryptoShadowServiceError,
        match="closed",
    ):
        _run(service)


def test_constructor_rejects_closed_components() -> None:
    worker = _Worker()
    worker.closed = True
    with pytest.raises(
        EdgeBoundCryptoShadowServiceError,
        match="Phase107 worker is closed",
    ):
        EdgeBoundCryptoShadowService(
            worker=worker,
            prefetch_provider=_Prefetch(),
        )

    worker = _Worker()
    prefetch = _Prefetch()
    prefetch.closed = True
    with pytest.raises(
        EdgeBoundCryptoShadowServiceError,
        match="Phase111 prefetch provider is closed",
    ):
        EdgeBoundCryptoShadowService(
            worker=worker,
            prefetch_provider=prefetch,
        )


def test_from_env_builds_owned_components_and_forwards_configuration() -> None:
    worker = _Worker()
    prefetch = _Prefetch()
    seen = {}

    def worker_factory(**kwargs):
        seen["worker"] = kwargs
        return worker

    def prefetch_factory(**kwargs):
        seen["prefetch"] = kwargs
        return prefetch

    service = EdgeBoundCryptoShadowService.from_env(
        asset_ids=("crypto:BTCUSDT",),
        model_weights={"market_snapshot_analyst": 1.0},
        config="phase54-config",
        max_slippage_bps=20.0,
        ttl_seconds=60,
        minimum_net_margin_bps=3.0,
        env={"SUPABASE_URL": "https://example.supabase.co"},
        initial_runtime="initial-runtime",
        foreign_cycle_aborter="aborter",
        client="rpc-client",
        worker_factory=worker_factory,
        prefetch_factory=prefetch_factory,
        clock=lambda: 123.0,
    )

    assert seen["worker"] == {
        "env": {"SUPABASE_URL": "https://example.supabase.co"},
        "initial_runtime": "initial-runtime",
        "foreign_cycle_aborter": "aborter",
        "client": "rpc-client",
    }
    assert seen["prefetch"]["asset_ids"] == ("crypto:BTCUSDT",)
    assert seen["prefetch"]["model_weights"] == {
        "market_snapshot_analyst": 1.0,
    }
    assert seen["prefetch"]["config"] == "phase54-config"
    assert seen["prefetch"]["max_slippage_bps"] == pytest.approx(20.0)
    assert seen["prefetch"]["ttl_seconds"] == 60
    assert seen["prefetch"]["minimum_net_margin_bps"] == pytest.approx(3.0)
    assert seen["prefetch"]["env"] == {
        "SUPABASE_URL": "https://example.supabase.co",
    }
    assert callable(seen["prefetch"]["clock"])

    assert service.close() is True
    assert prefetch.close_calls == 1
    assert worker.close_calls == 1


def test_from_env_closes_worker_if_prefetch_factory_fails() -> None:
    worker = _Worker()

    with pytest.raises(RuntimeError, match="prefetch unavailable"):
        EdgeBoundCryptoShadowService.from_env(
            asset_ids=("crypto:BTCUSDT",),
            model_weights={"market_snapshot_analyst": 1.0},
            config="phase54-config",
            max_slippage_bps=20.0,
            ttl_seconds=60,
            worker_factory=lambda **kwargs: worker,
            prefetch_factory=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("prefetch unavailable")
            ),
        )

    assert worker.close_calls == 1
    assert worker.closed is True
