from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

from .phase54_integrated_shadow_decision import IntegratedShadowConfig
from .phase94_binance_spot_recovery_evidence import (
    BinanceSpotRecoveryEvidenceProvider,
)
from .phase97_bounded_auto_recovery_drain import ProviderFactory
from .phase107_edge_bound_recovery_worker import (
    EdgeBoundRecoveryWorker,
    EdgeBoundRecoveryWorkerReceipt,
)
from .phase111_crypto_edge_bound_prefetch import (
    CryptoEdgeBoundPrefetchProvider,
)

PHASE112_SCHEMA_VERSION = "brian.phase112-edge-bound-crypto-shadow-service.v1"


class EdgeBoundCryptoShadowServiceError(RuntimeError):
    pass


class EdgeBoundCryptoShadowService:
    """Own the complete recovery-first current-data crypto shadow composition.

    Phase107 remains the runtime/recovery authority. Phase111 is the only normal
    prefetch provider, so expected edge is derived from Phase105/106 rather than
    supplied by the caller. This service adds lifecycle composition only: it
    does not schedule itself, change SQL, place orders, or enable live trading.
    """

    def __init__(
        self,
        *,
        worker: EdgeBoundRecoveryWorker,
        prefetch_provider: CryptoEdgeBoundPrefetchProvider,
        owns_components: bool = False,
    ) -> None:
        if getattr(worker, "closed", False):
            raise EdgeBoundCryptoShadowServiceError(
                "Phase107 worker is closed"
            )
        if getattr(prefetch_provider, "closed", False):
            raise EdgeBoundCryptoShadowServiceError(
                "Phase111 prefetch provider is closed"
            )
        if not callable(prefetch_provider):
            raise TypeError("prefetch_provider must be callable")
        if not callable(getattr(worker, "run_once", None)):
            raise TypeError("worker must expose callable run_once")
        self.worker = worker
        self.prefetch_provider = prefetch_provider
        self._owns_components = bool(owns_components)
        self._closed = False

    @classmethod
    def from_env(
        cls,
        *,
        asset_ids: Sequence[str],
        model_weights: Mapping[str, float],
        config: IntegratedShadowConfig,
        max_slippage_bps: float,
        ttl_seconds: int,
        minimum_net_margin_bps: float = 2.0,
        env: Mapping[str, str] | None = None,
        initial_runtime=None,
        foreign_cycle_aborter=None,
        client=None,
        worker_factory=EdgeBoundRecoveryWorker.from_env,
        prefetch_factory=CryptoEdgeBoundPrefetchProvider.from_env,
        clock: Callable[[], float] = time.time,
    ) -> "EdgeBoundCryptoShadowService":
        if not callable(worker_factory):
            raise TypeError("worker_factory must be callable")
        if not callable(prefetch_factory):
            raise TypeError("prefetch_factory must be callable")
        worker = None
        prefetch = None
        try:
            worker = worker_factory(
                env=env,
                initial_runtime=initial_runtime,
                foreign_cycle_aborter=foreign_cycle_aborter,
                client=client,
            )
            prefetch = prefetch_factory(
                asset_ids=asset_ids,
                model_weights=model_weights,
                config=config,
                max_slippage_bps=max_slippage_bps,
                ttl_seconds=ttl_seconds,
                minimum_net_margin_bps=minimum_net_margin_bps,
                env=env,
                clock=clock,
            )
            return cls(
                worker=worker,
                prefetch_provider=prefetch,
                owns_components=True,
            )
        except Exception:
            for component in (prefetch, worker):
                close = getattr(component, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            raise

    @property
    def runtime_id(self) -> str:
        return self.worker.runtime_id

    @property
    def closed(self) -> bool:
        return self._closed

    def run_once(
        self,
        *,
        recovery_max_items: int,
        recovery_worker_token: str,
        recovery_claim_seconds: int,
        recovery_ttl_seconds: int,
        recovery_source_ref: str,
        normal_worker_token: str,
        normal_claim_seconds: int,
        recovery_provider_factory: ProviderFactory = BinanceSpotRecoveryEvidenceProvider,
        clock=time.time,
    ) -> EdgeBoundRecoveryWorkerReceipt:
        if self._closed:
            raise EdgeBoundCryptoShadowServiceError(
                "Phase112 service is closed"
            )
        return self.worker.run_once(
            prefetch_provider=self.prefetch_provider,
            recovery_max_items=recovery_max_items,
            recovery_worker_token=recovery_worker_token,
            recovery_claim_seconds=recovery_claim_seconds,
            recovery_ttl_seconds=recovery_ttl_seconds,
            recovery_source_ref=recovery_source_ref,
            normal_worker_token=normal_worker_token,
            normal_claim_seconds=normal_claim_seconds,
            recovery_provider_factory=recovery_provider_factory,
            clock=clock,
        )

    def close(self) -> bool:
        if self._closed:
            return True
        released = True
        try:
            if self._owns_components:
                # Stop normal prefetch I/O first, then release runtime/session.
                self.prefetch_provider.close()
                released = bool(self.worker.close())
        finally:
            self._closed = True
        return released

    def __enter__(self) -> "EdgeBoundCryptoShadowService":
        if self._closed:
            raise EdgeBoundCryptoShadowServiceError(
                "Phase112 service is closed"
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
