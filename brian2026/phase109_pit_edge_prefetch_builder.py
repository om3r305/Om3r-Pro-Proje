from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .global_sensor_mesh import PROSPECTIVE_EVIDENCE_CLASS
from .phase54_integrated_shadow_decision import (
    AssetDecisionInput,
    IntegratedShadowConfig,
)
from .phase56_pretrade_risk_engine import InstrumentRiskLimits
from .phase57_shadow_execution_cycle import ExecutionMarketInput
from .phase107_edge_bound_recovery_worker import (
    PrefetchedLaggedEdgeGroundedCycle,
)
from .portfolio import DEVELOPMENT_CUTOFF

PHASE109_SCHEMA_VERSION = "brian.phase109-pit-edge-prefetch-builder.v1"


class PointInTimePrefetchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PointInTimeReturnSeries:
    asset_id: str
    values: tuple[float, ...]
    observed_from: float
    observed_until: float
    source_ids: tuple[str, ...]
    schema_version: str = PHASE109_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("asset_id is required")
        if not self.values:
            raise ValueError("return series cannot be empty")
        if any(not math.isfinite(float(value)) for value in self.values):
            raise ValueError("return series values must be finite")
        if not all(
            math.isfinite(float(value))
            for value in (self.observed_from, self.observed_until)
        ):
            raise ValueError("return series timestamps must be finite")
        if self.observed_until < self.observed_from:
            raise ValueError("return series time order is invalid")
        if not self.source_ids:
            raise ValueError("return series requires source lineage")
        if not self.shadow_only or self.live_execution:
            raise ValueError("return series must remain shadow-only")


def _validate_asset_inputs(
    asset_inputs: Mapping[str, AssetDecisionInput],
    *,
    decision_timestamp: float,
) -> dict[str, tuple[str, ...]]:
    groups_by_asset: dict[str, tuple[str, ...]] = {}
    if not asset_inputs:
        raise PointInTimePrefetchError("asset_inputs are required")
    for raw_asset, item in asset_inputs.items():
        asset = str(raw_asset).strip()
        if not asset:
            raise PointInTimePrefetchError("asset key cannot be blank")
        observations = tuple(item.observations)
        if not observations:
            raise PointInTimePrefetchError(
                f"{asset} requires prospective observations"
            )
        observed_assets = {str(row.asset_id) for row in observations}
        if observed_assets != {asset}:
            raise PointInTimePrefetchError(
                f"{asset} observation asset identity mismatch"
            )
        groups: set[str] = set()
        for row in observations:
            if row.evidence_class != PROSPECTIVE_EVIDENCE_CLASS:
                raise PointInTimePrefetchError(
                    f"{asset} observation has wrong evidence class"
                )
            if not row.shadow_only or row.live_execution:
                raise PointInTimePrefetchError(
                    f"{asset} observation crossed shadow-only boundary"
                )
            if float(row.observed_at) < DEVELOPMENT_CUTOFF:
                raise PointInTimePrefetchError(
                    f"{asset} reuses pre-cutoff development evidence"
                )
            if float(row.observed_at) > decision_timestamp:
                raise PointInTimePrefetchError(
                    f"{asset} observation is after decision timestamp"
                )
            group = str(row.independent_group).strip()
            if not group:
                raise PointInTimePrefetchError(
                    f"{asset} observation group is blank"
                )
            groups.add(group)
        groups_by_asset[asset] = tuple(sorted(groups))
    return groups_by_asset


def _validate_returns(
    asset_inputs: Mapping[str, AssetDecisionInput],
    return_series_by_asset: Mapping[str, PointInTimeReturnSeries],
    *,
    decision_timestamp: float,
) -> dict[str, tuple[float, ...]]:
    expected_assets = set(asset_inputs)
    if set(return_series_by_asset) != expected_assets:
        raise PointInTimePrefetchError(
            "return-series assets must exactly match decision assets"
        )
    result: dict[str, tuple[float, ...]] = {}
    for asset in sorted(expected_assets):
        row = return_series_by_asset[asset]
        if row.asset_id != asset:
            raise PointInTimePrefetchError(
                f"{asset} return-series identity mismatch"
            )
        if row.observed_until > decision_timestamp:
            raise PointInTimePrefetchError(
                f"{asset} return series contains post-decision data"
            )
        result[asset] = tuple(float(value) for value in row.values)
    return result


def build_pit_edge_prefetch_bundle(
    *,
    edge_reader,
    bundle_ref: str,
    asset_inputs: Mapping[str, AssetDecisionInput],
    decision_timestamp: float,
    model_weights: Mapping[str, float],
    return_series_by_asset: Mapping[str, PointInTimeReturnSeries],
    config: IntegratedShadowConfig,
    max_slippage_bps: float,
    ttl_seconds: int,
    markets: Mapping[str, ExecutionMarketInput],
    risk_limits_by_asset: Mapping[str, InstrumentRiskLimits],
    marks: Mapping[str, float],
    observed_at: float,
    source_ref: str,
    cost_asset_id_by_asset: Mapping[str, str] | None = None,
    minimum_net_margin_bps: float = 2.0,
) -> PrefetchedLaggedEdgeGroundedCycle:
    """Build the Phase107 bundle from causal observations and lagged edge data.

    This is the point-in-time boundary for covariance returns: unlike the legacy
    raw Sequence[float] input, every series must declare an observation window
    that ends no later than the Phase54 decision timestamp.
    """
    timestamp = float(decision_timestamp)
    if not math.isfinite(timestamp):
        raise ValueError("decision_timestamp must be finite")
    if timestamp < DEVELOPMENT_CUTOFF:
        raise PointInTimePrefetchError(
            "Phase109 requires post-cutoff prospective decision time"
        )
    execution_observed_at = float(observed_at)
    if not math.isfinite(execution_observed_at):
        raise ValueError("observed_at must be finite")
    if execution_observed_at < timestamp:
        raise PointInTimePrefetchError(
            "execution observation cannot precede decision timestamp"
        )
    if not hasattr(edge_reader, "load_contexts") or not callable(
        edge_reader.load_contexts
    ):
        raise TypeError("edge_reader must expose callable load_contexts")

    groups_by_asset = _validate_asset_inputs(
        asset_inputs,
        decision_timestamp=timestamp,
    )
    returns = _validate_returns(
        asset_inputs,
        return_series_by_asset,
        decision_timestamp=timestamp,
    )
    too_short = tuple(sorted(
        asset
        for asset, values in returns.items()
        if len(values) < config.covariance.min_observations
    ))
    if too_short:
        raise PointInTimePrefetchError(
            "return history is shorter than covariance minimum for "
            f"{too_short}: need {config.covariance.min_observations}"
        )
    contexts = edge_reader.load_contexts(
        groups_by_asset=groups_by_asset,
        decision_timestamp=timestamp,
        cost_asset_id_by_asset=cost_asset_id_by_asset,
        minimum_net_margin_bps=minimum_net_margin_bps,
    )
    if set(contexts) != set(asset_inputs):
        raise PointInTimePrefetchError(
            "edge reader did not return exactly one context per decision asset"
        )

    return PrefetchedLaggedEdgeGroundedCycle(
        bundle_ref=bundle_ref,
        asset_inputs=asset_inputs,
        timestamp=timestamp,
        model_weights=model_weights,
        returns_by_asset=returns,
        config=config,
        edge_contexts_by_asset=contexts,
        max_slippage_bps=float(max_slippage_bps),
        ttl_seconds=int(ttl_seconds),
        markets=markets,
        risk_limits_by_asset=risk_limits_by_asset,
        marks=marks,
        observed_at=execution_observed_at,
        source_ref=source_ref,
    )
