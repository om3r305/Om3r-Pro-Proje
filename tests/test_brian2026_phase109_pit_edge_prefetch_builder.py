from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.global_sensor_mesh import SensorObservation
from brian2026.phase44_portfolio_brain import PortfolioRiskLimits
from brian2026.phase52_covariance_risk import CovarianceRiskConfig
from brian2026.phase53_turnover_rebalance import TurnoverConfig
from brian2026.phase54_integrated_shadow_decision import (
    AssetDecisionInput,
    IntegratedShadowConfig,
)
from brian2026.phase106_decision_bound_lagged_edge import AssetLaggedEdgeContext
from brian2026.phase109_pit_edge_prefetch_builder import (
    PointInTimePrefetchError,
    PointInTimeReturnSeries,
    build_pit_edge_prefetch_bundle,
)
from brian2026.portfolio import DEVELOPMENT_CUTOFF


TS = 1_790_000_000.0


def _obs(
    asset: str,
    eye: str,
    group: str,
    *,
    observed_at: float = TS - 60,
) -> SensorObservation:
    return SensorObservation(
        eye_id=f"{asset}-{eye}",
        asset_id=asset,
        observed_at=observed_at,
        direction=1,
        strength=0.8,
        confidence=0.8,
        reliability=0.7,
        available=True,
        independent_group=group,
        source_ids=(f"src-{asset}-{eye}",),
        horizon="FAST_5_30M",
        reason="phase109 fixture",
    )


def _asset_input(asset: str) -> AssetDecisionInput:
    return AssetDecisionInput(
        snapshot={"structure_state": 1.0},
        observations=(
            _obs(asset, "price", "price_structure"),
            _obs(asset, "deriv", "derivatives", observed_at=TS - 90),
        ),
        source_kind_by_eye={
            f"{asset}-price": "market_snapshot",
            f"{asset}-deriv": "derivatives",
        },
    )


def _returns(asset: str, *, until: float = TS - 1) -> PointInTimeReturnSeries:
    return PointInTimeReturnSeries(
        asset_id=asset,
        values=(0.01, -0.005, 0.002, 0.004, -0.001),
        observed_from=TS - 3600,
        observed_until=until,
        source_ids=(f"returns-{asset}",),
    )


def _config() -> IntegratedShadowConfig:
    return IntegratedShadowConfig(
        gross_target=0.5,
        position_limits=PortfolioRiskLimits(
            max_position_pct=0.3,
            max_gross_exposure=0.5,
        ),
        covariance=CovarianceRiskConfig(
            min_observations=5,
            max_period_volatility=1.0,
        ),
        turnover=TurnoverConfig(
            max_l1_turnover=1.0,
            risk_reduction_bypass=True,
        ),
    )


class _Reader:
    def __init__(self, *, missing=()):
        self.calls = []
        self.missing = set(missing)

    def load_contexts(self, **kwargs):
        self.calls.append(kwargs)
        return {
            asset: AssetLaggedEdgeContext(
                reliability=(),
                round_trip_cost_bps=None,
                cost_observed_at=None,
            )
            for asset in kwargs["groups_by_asset"]
            if asset not in self.missing
        }


def _build(reader, **overrides):
    asset_inputs = {
        "BTCUSDT": _asset_input("BTCUSDT"),
        "ETHUSDT": _asset_input("ETHUSDT"),
    }
    values = dict(
        edge_reader=reader,
        bundle_ref="bundle-109",
        asset_inputs=asset_inputs,
        decision_timestamp=TS,
        model_weights={"market_snapshot_analyst": 1.0},
        return_series_by_asset={
            "BTCUSDT": _returns("BTCUSDT"),
            "ETHUSDT": _returns("ETHUSDT"),
        },
        config=_config(),
        max_slippage_bps=20.0,
        ttl_seconds=60,
        markets={"BTCUSDT": "market-btc", "ETHUSDT": "market-eth"},
        risk_limits_by_asset={},
        marks={"BTCUSDT": 100.0, "ETHUSDT": 50.0},
        observed_at=TS + 1,
        source_ref="phase109:test",
        cost_asset_id_by_asset={
            "BTCUSDT": "crypto:BTCUSDT",
            "ETHUSDT": "crypto:ETHUSDT",
        },
        minimum_net_margin_bps=3.0,
    )
    values.update(overrides)
    return build_pit_edge_prefetch_bundle(**values)


def test_builder_freezes_causal_returns_and_derives_edge_groups_from_observations() -> None:
    reader = _Reader()

    bundle = _build(reader)

    assert bundle.bundle_ref == "bundle-109"
    assert bundle.timestamp == TS
    assert bundle.returns_by_asset == {
        "BTCUSDT": (0.01, -0.005, 0.002, 0.004, -0.001),
        "ETHUSDT": (0.01, -0.005, 0.002, 0.004, -0.001),
    }
    assert set(bundle.edge_contexts_by_asset) == {"BTCUSDT", "ETHUSDT"}
    assert len(reader.calls) == 1
    call = reader.calls[0]
    assert call["groups_by_asset"] == {
        "BTCUSDT": ("derivatives", "price_structure"),
        "ETHUSDT": ("derivatives", "price_structure"),
    }
    assert call["decision_timestamp"] == TS
    assert call["cost_asset_id_by_asset"] == {
        "BTCUSDT": "crypto:BTCUSDT",
        "ETHUSDT": "crypto:ETHUSDT",
    }
    assert call["minimum_net_margin_bps"] == pytest.approx(3.0)
    assert bundle.shadow_only is True
    assert bundle.live_execution is False


def test_future_sensor_observation_fails_before_edge_reader() -> None:
    reader = _Reader()
    bad = AssetDecisionInput(
        snapshot={},
        observations=(
            _obs(
                "BTCUSDT",
                "future",
                "price_structure",
                observed_at=TS + 1,
            ),
        ),
        source_kind_by_eye={},
    )

    with pytest.raises(
        PointInTimePrefetchError,
        match="after decision timestamp",
    ):
        _build(
            reader,
            asset_inputs={"BTCUSDT": bad},
            return_series_by_asset={"BTCUSDT": _returns("BTCUSDT")},
        )

    assert reader.calls == []


def test_pre_cutoff_sensor_reuse_is_rejected() -> None:
    reader = _Reader()
    bad = AssetDecisionInput(
        snapshot={},
        observations=(
            _obs(
                "BTCUSDT",
                "old",
                "price_structure",
                observed_at=DEVELOPMENT_CUTOFF - 1,
            ),
        ),
        source_kind_by_eye={},
    )

    with pytest.raises(
        PointInTimePrefetchError,
        match="pre-cutoff development evidence",
    ):
        _build(
            reader,
            asset_inputs={"BTCUSDT": bad},
            return_series_by_asset={"BTCUSDT": _returns("BTCUSDT")},
        )

    assert reader.calls == []


def test_return_series_cannot_extend_past_decision_time() -> None:
    reader = _Reader()

    with pytest.raises(
        PointInTimePrefetchError,
        match="post-decision data",
    ):
        _build(
            reader,
            return_series_by_asset={
                "BTCUSDT": _returns("BTCUSDT", until=TS + 0.1),
                "ETHUSDT": _returns("ETHUSDT"),
            },
        )

    assert reader.calls == []


def test_return_series_assets_must_exactly_match_decision_assets() -> None:
    reader = _Reader()

    with pytest.raises(
        PointInTimePrefetchError,
        match="exactly match",
    ):
        _build(
            reader,
            return_series_by_asset={
                "BTCUSDT": _returns("BTCUSDT"),
            },
        )

    assert reader.calls == []


def test_observation_asset_identity_mismatch_fails_closed() -> None:
    reader = _Reader()
    bad = AssetDecisionInput(
        snapshot={},
        observations=(_obs("ETHUSDT", "price", "price_structure"),),
        source_kind_by_eye={},
    )

    with pytest.raises(
        PointInTimePrefetchError,
        match="asset identity mismatch",
    ):
        _build(
            reader,
            asset_inputs={"BTCUSDT": bad},
            return_series_by_asset={"BTCUSDT": _returns("BTCUSDT")},
        )

    assert reader.calls == []


def test_edge_reader_must_return_context_for_every_decision_asset() -> None:
    reader = _Reader(missing=("ETHUSDT",))

    with pytest.raises(
        PointInTimePrefetchError,
        match="exactly one context",
    ):
        _build(reader)

    assert len(reader.calls) == 1


def test_execution_observation_cannot_precede_decision() -> None:
    reader = _Reader()

    with pytest.raises(
        PointInTimePrefetchError,
        match="cannot precede decision",
    ):
        _build(reader, observed_at=TS - 1)

    assert reader.calls == []


def test_phase109_requires_post_cutoff_decision_time() -> None:
    reader = _Reader()

    with pytest.raises(
        PointInTimePrefetchError,
        match="post-cutoff prospective",
    ):
        _build(
            reader,
            decision_timestamp=DEVELOPMENT_CUTOFF - 1,
        )

    assert reader.calls == []


def test_return_series_requires_lineage() -> None:
    with pytest.raises(ValueError, match="source lineage"):
        PointInTimeReturnSeries(
            asset_id="BTCUSDT",
            values=(0.01,),
            observed_from=TS - 60,
            observed_until=TS - 1,
            source_ids=(),
        )

def test_return_history_shorter_than_covariance_minimum_fails_before_edge_reader() -> None:
    reader = _Reader()
    short = PointInTimeReturnSeries(
        asset_id="BTCUSDT",
        values=(0.01, -0.01, 0.005, 0.002),
        observed_from=TS - 3600,
        observed_until=TS - 1,
        source_ids=("short-returns",),
    )

    with pytest.raises(
        PointInTimePrefetchError,
        match="shorter than covariance minimum",
    ):
        _build(
            reader,
            asset_inputs={"BTCUSDT": _asset_input("BTCUSDT")},
            return_series_by_asset={"BTCUSDT": short},
        )

    assert reader.calls == []

