from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.global_sensor_mesh import SensorObservation
from brian2026.phase44_portfolio_brain import PortfolioRiskLimits
from brian2026.phase46_execution_simulator import (
    LiquidityLevel,
    OrderBookSnapshot,
)
from brian2026.phase52_covariance_risk import CovarianceRiskConfig
from brian2026.phase53_turnover_rebalance import TurnoverConfig
from brian2026.phase54_integrated_shadow_decision import (
    AssetDecisionInput,
    IntegratedShadowConfig,
)
from brian2026.phase56_pretrade_risk_engine import InstrumentRiskLimits
from brian2026.phase57_shadow_execution_cycle import ExecutionMarketInput
from brian2026.phase94_binance_spot_recovery_evidence import (
    BinanceSpotRecoveryAssetEvidence,
    BinanceSpotRecoveryEvidenceBundle,
    PUBLIC_MARKET_HOSTS,
)
from brian2026.phase106_decision_bound_lagged_edge import (
    AssetLaggedEdgeContext,
)
from brian2026.phase109_pit_edge_prefetch_builder import PointInTimeReturnSeries
from brian2026.phase110_supabase_grounded_market_prefetch import (
    GroundedMarketPrefetch,
)
from brian2026.phase111_crypto_edge_bound_prefetch import (
    CryptoEdgeBoundPrefetchError,
    CryptoEdgeBoundPrefetchProvider,
)


TS = 1_789_999_950.0
ASSET = "crypto:BTCUSDT"
SYMBOL = "BTCUSDT"


def _config() -> IntegratedShadowConfig:
    return IntegratedShadowConfig(
        gross_target=0.5,
        position_limits=PortfolioRiskLimits(
            max_position_pct=0.3,
            max_gross_exposure=0.5,
        ),
        covariance=CovarianceRiskConfig(
            min_observations=30,
            max_period_volatility=1.0,
        ),
        turnover=TurnoverConfig(
            max_l1_turnover=1.0,
            risk_reduction_bypass=True,
        ),
    )


def _market_prefetch() -> GroundedMarketPrefetch:
    observation = SensorObservation(
        eye_id="eye-btc-structure",
        asset_id=ASSET,
        observed_at=TS - 30,
        direction=1,
        strength=0.8,
        confidence=0.8,
        reliability=0.7,
        available=True,
        independent_group="price_structure",
        source_ids=("obs-btc", "raw-btc"),
        horizon="FAST_5_30M",
        reason="phase111 fixture",
    )
    asset_input = AssetDecisionInput(
        snapshot={
            "structure_state": 1.0,
            "return_1": 0.001,
            "ema_slope": 0.0005,
        },
        observations=(observation,),
        source_kind_by_eye={
            "eye-btc-structure": "market_snapshot",
        },
    )
    returns = PointInTimeReturnSeries(
        asset_id=ASSET,
        values=tuple(0.001 * ((index % 5) - 2) for index in range(30)),
        observed_from=TS - 31 * 300,
        observed_until=TS - 300,
        source_ids=tuple(f"tick-{index:02d}" for index in range(31)),
    )
    return GroundedMarketPrefetch(
        decision_timestamp=TS,
        asset_inputs={ASSET: asset_input},
        return_series_by_asset={ASSET: returns},
        marks={ASSET: 100.0},
        cost_asset_id_by_asset={ASSET: ASSET},
        common_return_buckets=tuple(
            TS - (31 - index) * 300
            for index in range(31)
        ),
    )


def _execution_bundle(*, observed_at=TS + 1, include=True):
    if not include:
        return BinanceSpotRecoveryEvidenceBundle(())
    book = OrderBookSnapshot(
        timestamp=observed_at,
        bids=(
            LiquidityLevel(100.0, 10.0),
            LiquidityLevel(99.9, 10.0),
        ),
        asks=(
            LiquidityLevel(100.1, 10.0),
            LiquidityLevel(100.2, 10.0),
        ),
    )
    market = ExecutionMarketInput(
        reference_price=100.05,
        tick_size=0.01,
        snapshots=(book,),
    )
    row = BinanceSpotRecoveryAssetEvidence(
        asset_id=SYMBOL,
        market=market,
        risk_limits=InstrumentRiskLimits(min_notional=5.0),
        mark=100.05,
        observed_at=observed_at,
        depth_last_update_id="123456",
        source_host=PUBLIC_MARKET_HOSTS[0],
        exchange_status="TRADING",
    )
    return BinanceSpotRecoveryEvidenceBundle((row,))


class _MarketReader:
    def __init__(self, result=None):
        self.result = _market_prefetch() if result is None else result
        self.calls = []
        self.close_calls = 0

    def load(self, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def close(self):
        self.close_calls += 1


class _EdgeReader:
    def __init__(self):
        self.calls = []
        self.close_calls = 0

    def load_contexts(self, **kwargs):
        self.calls.append(kwargs)
        return {
            asset: AssetLaggedEdgeContext(
                reliability=(),
                round_trip_cost_bps=None,
                cost_observed_at=None,
                minimum_net_margin_bps=kwargs["minimum_net_margin_bps"],
            )
            for asset in kwargs["groups_by_asset"]
        }

    def close(self):
        self.close_calls += 1


class _ExecutionProvider:
    def __init__(self, result=None):
        self.result = _execution_bundle() if result is None else result
        self.calls = []
        self.close_calls = 0

    def collect(self, assets):
        self.calls.append(tuple(assets))
        return self.result

    def close(self):
        self.close_calls += 1


def _provider(
    *,
    market=None,
    edge=None,
    execution=None,
    clock=lambda: TS,
    owns_resources=False,
):
    return CryptoEdgeBoundPrefetchProvider(
        asset_ids=(ASSET,),
        model_weights={"market_snapshot_analyst": 1.0},
        config=_config(),
        market_reader=market or _MarketReader(),
        edge_reader=edge or _EdgeReader(),
        execution_provider=execution or _ExecutionProvider(),
        max_slippage_bps=20.0,
        ttl_seconds=60,
        minimum_net_margin_bps=2.0,
        clock=clock,
        owns_resources=owns_resources,
    )


def test_phase111_composes_phase110_108_109_and_public_phase94_evidence() -> None:
    market = _MarketReader()
    edge = _EdgeReader()
    execution = _ExecutionProvider()
    provider = _provider(
        market=market,
        edge=edge,
        execution=execution,
    )

    bundle = provider()

    assert market.calls == [{
        "asset_ids": (ASSET,),
        "decision_timestamp": TS,
    }]
    assert execution.calls == [(SYMBOL,)]
    assert len(edge.calls) == 1
    assert edge.calls[0]["groups_by_asset"] == {
        ASSET: ("price_structure",),
    }
    assert edge.calls[0]["decision_timestamp"] == TS
    assert edge.calls[0]["cost_asset_id_by_asset"] == {
        ASSET: ASSET,
    }

    assert set(bundle.asset_inputs) == {ASSET}
    assert set(bundle.returns_by_asset) == {ASSET}
    assert set(bundle.edge_contexts_by_asset) == {ASSET}
    assert set(bundle.markets) == {ASSET}
    assert set(bundle.risk_limits_by_asset) == {ASSET}
    assert bundle.marks == {ASSET: pytest.approx(100.05)}
    assert bundle.timestamp == TS
    assert bundle.observed_at == pytest.approx(TS + 1)
    assert len(bundle.bundle_ref) == 64
    assert bundle.source_ref == f"phase111:{bundle.bundle_ref}"
    assert bundle.shadow_only is True
    assert bundle.live_execution is False


def test_phase111_identity_is_deterministic_for_identical_frozen_inputs() -> None:
    provider = _provider()

    first = provider()
    second = provider()

    assert first.bundle_ref == second.bundle_ref
    assert first.source_ref == second.source_ref


def test_noncrypto_assets_fail_before_any_reader_or_provider_is_used() -> None:
    market = _MarketReader()
    edge = _EdgeReader()
    execution = _ExecutionProvider()

    with pytest.raises(
        CryptoEdgeBoundPrefetchError,
        match="crypto:\\*USDT only",
    ):
        CryptoEdgeBoundPrefetchProvider(
            asset_ids=("fx:EURUSD",),
            model_weights={"market_snapshot_analyst": 1.0},
            config=_config(),
            market_reader=market,
            edge_reader=edge,
            execution_provider=execution,
            max_slippage_bps=20.0,
            ttl_seconds=60,
        )

    assert market.calls == []
    assert edge.calls == []
    assert execution.calls == []


def test_phase94_must_cover_requested_crypto_assets_exactly() -> None:
    provider = _provider(
        execution=_ExecutionProvider(
            _execution_bundle(include=False)
        )
    )

    with pytest.raises(
        CryptoEdgeBoundPrefetchError,
        match="does not cover requested assets exactly",
    ):
        provider()


def test_phase94_execution_snapshot_cannot_predate_decision() -> None:
    provider = _provider(
        execution=_ExecutionProvider(
            _execution_bundle(observed_at=TS - 1)
        )
    )

    with pytest.raises(
        CryptoEdgeBoundPrefetchError,
        match="predates Phase111 decision",
    ):
        provider()


def test_phase110_timestamp_drift_is_rejected() -> None:
    original = _market_prefetch()
    drifted = GroundedMarketPrefetch(
        decision_timestamp=TS - 1,
        asset_inputs=original.asset_inputs,
        return_series_by_asset=original.return_series_by_asset,
        marks=original.marks,
        cost_asset_id_by_asset=original.cost_asset_id_by_asset,
        common_return_buckets=original.common_return_buckets,
    )
    provider = _provider(
        market=_MarketReader(drifted),
    )

    with pytest.raises(
        CryptoEdgeBoundPrefetchError,
        match="decision timestamp drift",
    ):
        provider()


def test_closed_provider_rejects_prefetch() -> None:
    provider = _provider()
    provider.close()

    with pytest.raises(
        CryptoEdgeBoundPrefetchError,
        match="closed",
    ):
        provider()


def test_owned_resources_are_closed_in_reverse_runtime_order() -> None:
    market = _MarketReader()
    edge = _EdgeReader()
    execution = _ExecutionProvider()
    provider = _provider(
        market=market,
        edge=edge,
        execution=execution,
        owns_resources=True,
    )

    assert provider.close() is True
    assert provider.close() is True
    assert execution.close_calls == 1
    assert edge.close_calls == 1
    assert market.close_calls == 1


def test_external_resources_are_not_closed() -> None:
    market = _MarketReader()
    edge = _EdgeReader()
    execution = _ExecutionProvider()
    provider = _provider(
        market=market,
        edge=edge,
        execution=execution,
        owns_resources=False,
    )

    provider.close()

    assert execution.close_calls == 0
    assert edge.close_calls == 0
    assert market.close_calls == 0


def test_from_env_closes_partially_opened_resources_on_factory_failure() -> None:
    market = _MarketReader()
    edge = _EdgeReader()

    def market_factory(**kwargs):
        return market

    def edge_factory(**kwargs):
        return edge

    def execution_factory():
        raise RuntimeError("execution adapter unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        CryptoEdgeBoundPrefetchProvider.from_env(
            asset_ids=(ASSET,),
            model_weights={"market_snapshot_analyst": 1.0},
            config=_config(),
            max_slippage_bps=20.0,
            ttl_seconds=60,
            env={"SUPABASE_URL": "https://example.supabase.co"},
            market_reader_factory=market_factory,
            edge_reader_factory=edge_factory,
            execution_provider_factory=execution_factory,
            clock=lambda: TS,
        )

    assert edge.close_calls == 1
    assert market.close_calls == 1
