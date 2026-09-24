from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace

from .phase54_integrated_shadow_decision import IntegratedShadowConfig
from .phase94_binance_spot_recovery_evidence import (
    BinanceSpotRecoveryEvidenceBundle,
    BinanceSpotRecoveryEvidenceProvider,
)
from .phase107_edge_bound_recovery_worker import PrefetchedLaggedEdgeGroundedCycle
from .phase108_supabase_lagged_edge_reader import SupabaseLaggedEdgeReader
from .phase109_pit_edge_prefetch_builder import build_pit_edge_prefetch_bundle
from .phase110_supabase_grounded_market_prefetch import (
    GroundedMarketPrefetch,
    SupabaseGroundedMarketPrefetchReader,
)

PHASE111_SCHEMA_VERSION = "brian.phase111-crypto-edge-bound-prefetch.v1"
_CRYPTO_ASSET = re.compile(r"^crypto:([A-Z0-9]{2,20}USDT)$")


class CryptoEdgeBoundPrefetchError(RuntimeError):
    pass


def _sha(payload: object) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _crypto_symbol(asset_id: str) -> str:
    text = str(asset_id).strip()
    match = _CRYPTO_ASSET.fullmatch(text)
    if match is None:
        raise CryptoEdgeBoundPrefetchError(
            f"Phase111 currently supports crypto:*USDT only: {text!r}"
        )
    return match.group(1)


def _bundle_identity_payload(
    *,
    decision_timestamp: float,
    market: GroundedMarketPrefetch,
    execution: BinanceSpotRecoveryEvidenceBundle,
    bundle: PrefetchedLaggedEdgeGroundedCycle,
) -> dict[str, object]:
    return {
        "schema_version": PHASE111_SCHEMA_VERSION,
        "decision_timestamp": float(decision_timestamp),
        "assets": sorted(bundle.asset_inputs),
        "market": {
            "common_return_buckets": list(market.common_return_buckets),
            "marks": dict(sorted(market.marks.items())),
            "asset_inputs": {
                asset: {
                    "snapshot": dict(sorted(item.snapshot.items())),
                    "observation_ids": sorted(
                        row.observation_id for row in item.observations
                    ),
                    "source_kind_by_eye": dict(
                        sorted(item.source_kind_by_eye.items())
                    ),
                }
                for asset, item in sorted(market.asset_inputs.items())
            },
            "returns": {
                asset: {
                    "observed_from": row.observed_from,
                    "observed_until": row.observed_until,
                    "values": list(row.values),
                    "source_ids": list(row.source_ids),
                }
                for asset, row in sorted(market.return_series_by_asset.items())
            },
        },
        "decision_policy": {
            "model_weights": dict(sorted(bundle.model_weights.items())),
            "config": asdict(bundle.config),
            "max_slippage_bps": bundle.max_slippage_bps,
            "ttl_seconds": bundle.ttl_seconds,
        },
        "edge_contexts": {
            asset: {
                "round_trip_cost_bps": context.round_trip_cost_bps,
                "cost_observed_at": context.cost_observed_at,
                "minimum_net_margin_bps": context.minimum_net_margin_bps,
                "reliability": [
                    {
                        "group": row.group,
                        "sample_count": row.sample_count,
                        "bayesian_hit_rate": row.bayesian_hit_rate,
                        "avg_signed_bps": row.avg_signed_bps,
                        "avg_cost_adjusted_signed_bps":
                            row.avg_cost_adjusted_signed_bps,
                        "outcome_horizon_seconds":
                            row.outcome_horizon_seconds,
                        "snapshot_window_end": row.snapshot_window_end,
                        "snapshot_generated_at": row.snapshot_generated_at,
                    }
                    for row in context.reliability
                ],
            }
            for asset, context in sorted(bundle.edge_contexts_by_asset.items())
        },
        "execution": [
            {
                "symbol": row.asset_id,
                "observed_at": row.observed_at,
                "depth_last_update_id": row.depth_last_update_id,
                "source_host": row.source_host,
                "exchange_status": row.exchange_status,
                "mark": row.mark,
                "reference_price": row.market.reference_price,
                "tick_size": row.market.tick_size,
                "min_notional": row.risk_limits.min_notional,
            }
            for row in execution.assets
        ],
        "shadow_only": True,
        "live_execution": False,
    }


class CryptoEdgeBoundPrefetchProvider:
    """Zero-argument Phase107 prefetch provider for canonical crypto assets.

    Phase110 freezes grounded sensor/price/return history at one decision time.
    Phase108 supplies only lagged PIT reliability/cost evidence at that time.
    Phase94 then collects fresh public Binance depth/exchange-info evidence for
    paper execution. Canonical crypto:BTCUSDT ids are mapped to Binance BTCUSDT
    only at this public-market boundary.

    Non-crypto assets fail closed because there is no proven venue depth/risk
    adapter for them in this path. No synthetic book is manufactured.
    """

    def __init__(
        self,
        *,
        asset_ids: Sequence[str],
        model_weights: Mapping[str, float],
        config: IntegratedShadowConfig,
        market_reader: SupabaseGroundedMarketPrefetchReader,
        edge_reader: SupabaseLaggedEdgeReader,
        execution_provider: BinanceSpotRecoveryEvidenceProvider,
        max_slippage_bps: float,
        ttl_seconds: int,
        minimum_net_margin_bps: float = 2.0,
        clock: Callable[[], float] = time.time,
        owns_resources: bool = False,
    ) -> None:
        assets = tuple(sorted({str(value).strip() for value in asset_ids}))
        if not assets:
            raise ValueError("asset_ids are required")
        for asset in assets:
            _crypto_symbol(asset)
        if not model_weights:
            raise ValueError("model_weights are required")
        clean_weights = {
            str(name): float(value)
            for name, value in model_weights.items()
        }
        if any(
            not name.strip() or not math.isfinite(value) or value < 0
            for name, value in clean_weights.items()
        ):
            raise ValueError(
                "model_weights require non-empty names and finite non-negative values"
            )
        if not math.isfinite(float(max_slippage_bps)) or max_slippage_bps < 0:
            raise ValueError("max_slippage_bps must be finite and non-negative")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if (
            not math.isfinite(float(minimum_net_margin_bps))
            or minimum_net_margin_bps < 0
        ):
            raise ValueError(
                "minimum_net_margin_bps must be finite and non-negative"
            )
        for label, value, method in (
            ("market_reader", market_reader, "load"),
            ("edge_reader", edge_reader, "load_contexts"),
            ("execution_provider", execution_provider, "collect"),
        ):
            if not callable(getattr(value, method, None)):
                raise TypeError(f"{label} must expose callable {method}")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self.asset_ids = assets
        self.model_weights = clean_weights
        self.config = config
        self.market_reader = market_reader
        self.edge_reader = edge_reader
        self.execution_provider = execution_provider
        self.max_slippage_bps = float(max_slippage_bps)
        self.ttl_seconds = int(ttl_seconds)
        self.minimum_net_margin_bps = float(minimum_net_margin_bps)
        self.clock = clock
        self._owns_resources = bool(owns_resources)
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
        market_reader_factory=SupabaseGroundedMarketPrefetchReader.from_env,
        edge_reader_factory=SupabaseLaggedEdgeReader.from_env,
        execution_provider_factory=BinanceSpotRecoveryEvidenceProvider,
        clock: Callable[[], float] = time.time,
    ) -> "CryptoEdgeBoundPrefetchProvider":
        market_reader = None
        edge_reader = None
        execution_provider = None
        try:
            market_reader = market_reader_factory(env=env)
            edge_reader = edge_reader_factory(env=env)
            execution_provider = execution_provider_factory()
            return cls(
                asset_ids=asset_ids,
                model_weights=model_weights,
                config=config,
                market_reader=market_reader,
                edge_reader=edge_reader,
                execution_provider=execution_provider,
                max_slippage_bps=max_slippage_bps,
                ttl_seconds=ttl_seconds,
                minimum_net_margin_bps=minimum_net_margin_bps,
                clock=clock,
                owns_resources=True,
            )
        except Exception:
            for resource in (execution_provider, edge_reader, market_reader):
                close = getattr(resource, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def __call__(self) -> PrefetchedLaggedEdgeGroundedCycle:
        if self._closed:
            raise CryptoEdgeBoundPrefetchError(
                "Phase111 prefetch provider is closed"
            )
        decision_timestamp = float(self.clock())
        if not math.isfinite(decision_timestamp):
            raise ValueError("clock must return a finite timestamp")

        market = self.market_reader.load(
            asset_ids=self.asset_ids,
            decision_timestamp=decision_timestamp,
        )
        if not isinstance(market, GroundedMarketPrefetch):
            raise CryptoEdgeBoundPrefetchError(
                "Phase110 reader returned invalid prefetch type"
            )
        if not math.isclose(
            market.decision_timestamp,
            decision_timestamp,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise CryptoEdgeBoundPrefetchError(
                "Phase110 decision timestamp drift"
            )
        if set(market.asset_inputs) != set(self.asset_ids):
            raise CryptoEdgeBoundPrefetchError(
                "Phase110 asset coverage differs from Phase111 request"
            )

        symbols = tuple(_crypto_symbol(asset) for asset in self.asset_ids)
        execution = self.execution_provider.collect(symbols)
        if not isinstance(execution, BinanceSpotRecoveryEvidenceBundle):
            raise CryptoEdgeBoundPrefetchError(
                "Phase94 provider returned invalid evidence type"
            )
        by_symbol = {row.asset_id: row for row in execution.assets}
        if set(by_symbol) != set(symbols):
            raise CryptoEdgeBoundPrefetchError(
                "Phase94 execution evidence does not cover requested assets exactly"
            )

        canonical_by_symbol = {
            _crypto_symbol(asset): asset for asset in self.asset_ids
        }
        markets = {
            canonical_by_symbol[row.asset_id]: row.market
            for row in execution.assets
        }
        risk_limits = {
            canonical_by_symbol[row.asset_id]: row.risk_limits
            for row in execution.assets
        }
        marks = {
            canonical_by_symbol[row.asset_id]: row.mark
            for row in execution.assets
        }
        execution_observed_at = max(row.observed_at for row in execution.assets)
        if execution_observed_at + 1e-9 < decision_timestamp:
            raise CryptoEdgeBoundPrefetchError(
                "Phase94 execution evidence predates Phase111 decision snapshot"
            )
        if any(
            not row.shadow_only or row.live_execution
            for row in execution.assets
        ):
            raise CryptoEdgeBoundPrefetchError(
                "Phase94 execution evidence crossed shadow-only boundary"
            )

        provisional_ref = _sha({
            "schema_version": PHASE111_SCHEMA_VERSION,
            "decision_timestamp": decision_timestamp,
            "assets": self.asset_ids,
            "execution": [
                (
                    row.asset_id,
                    row.depth_last_update_id,
                    row.observed_at,
                    row.mark,
                )
                for row in execution.assets
            ],
            "common_return_buckets": market.common_return_buckets,
        })
        bundle = build_pit_edge_prefetch_bundle(
            edge_reader=self.edge_reader,
            bundle_ref=provisional_ref,
            asset_inputs=market.asset_inputs,
            decision_timestamp=decision_timestamp,
            model_weights=self.model_weights,
            return_series_by_asset=market.return_series_by_asset,
            config=self.config,
            max_slippage_bps=self.max_slippage_bps,
            ttl_seconds=self.ttl_seconds,
            markets=markets,
            risk_limits_by_asset=risk_limits,
            marks=marks,
            observed_at=execution_observed_at,
            source_ref=f"phase111:{provisional_ref}",
            cost_asset_id_by_asset=market.cost_asset_id_by_asset,
            minimum_net_margin_bps=self.minimum_net_margin_bps,
        )

        identity = _sha(_bundle_identity_payload(
            decision_timestamp=decision_timestamp,
            market=market,
            execution=execution,
            bundle=bundle,
        ))
        return replace(
            bundle,
            bundle_ref=identity,
            source_ref=f"phase111:{identity}",
        )

    def close(self) -> bool:
        if self._closed:
            return True
        try:
            if self._owns_resources:
                for resource in (
                    self.execution_provider,
                    self.edge_reader,
                    self.market_reader,
                ):
                    close = getattr(resource, "close", None)
                    if callable(close):
                        close()
        finally:
            self._closed = True
        return True

    def __enter__(self) -> "CryptoEdgeBoundPrefetchProvider":
        if self._closed:
            raise CryptoEdgeBoundPrefetchError(
                "Phase111 prefetch provider is closed"
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
