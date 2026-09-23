from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .expert_reasoner import reason_market_prospective
from .global_sensor_mesh import (
    PROSPECTIVE_EVIDENCE_CLASS,
    SensorObservation,
)
from .phase43_grounded_analysts import (
    Phase43GroundedResult,
    run_grounded_phase43,
)
from .phase54_integrated_shadow_decision import (
    AssetDecisionInput,
    IntegratedShadowConfig,
    IntegratedShadowDecision,
    run_integrated_shadow_decision,
)
from .portfolio import DEVELOPMENT_CUTOFF

PHASE103_SCHEMA_VERSION = "brian.phase103-prospective-grounded-runtime.v1"


class ProspectiveGroundedRuntimeError(RuntimeError):
    pass


def _validate_prospective_observations(
    observations: Sequence[SensorObservation],
    *,
    timestamp: float,
) -> tuple[SensorObservation, ...]:
    rows = tuple(observations)
    if not rows:
        raise ProspectiveGroundedRuntimeError(
            "prospective grounded runtime requires sensor observations"
        )
    for row in rows:
        if row.evidence_class != PROSPECTIVE_EVIDENCE_CLASS:
            raise ProspectiveGroundedRuntimeError(
                "prospective observation has wrong evidence class"
            )
        if not row.shadow_only or row.live_execution:
            raise ProspectiveGroundedRuntimeError(
                "prospective observation crossed shadow-only boundary"
            )
        if float(row.observed_at) < DEVELOPMENT_CUTOFF:
            raise ProspectiveGroundedRuntimeError(
                "prospective runtime cannot reuse pre-cutoff development evidence"
            )
        if float(row.observed_at) > float(timestamp):
            raise ProspectiveGroundedRuntimeError(
                "prospective observation is from the future"
            )
    return rows


def run_prospective_grounded_phase43(
    snapshot: Mapping[str, object],
    observations: Sequence[SensorObservation],
    *,
    timestamp: float,
    source_kind_by_eye: Mapping[str, str] | None = None,
) -> Phase43GroundedResult:
    """Run Phase43 on post-cutoff prospective shadow observations only.

    The original Phase43 default path still calls the frozen pre-2026 reasoner.
    This wrapper is the explicit post-cutoff runtime lane: observations must be
    prospectively captured, shadow-only, timestamp-safe, and cannot be reused
    from the frozen development period.
    """
    if not math.isfinite(float(timestamp)):
        raise ValueError("timestamp must be finite")
    if float(timestamp) < DEVELOPMENT_CUTOFF:
        raise ProspectiveGroundedRuntimeError(
            "prospective Phase43 requires a post-cutoff decision timestamp"
        )
    rows = _validate_prospective_observations(
        observations,
        timestamp=float(timestamp),
    )
    result = run_grounded_phase43(
        snapshot,
        rows,
        timestamp=float(timestamp),
        source_kind_by_eye=source_kind_by_eye,
        reasoner=reason_market_prospective,
    )
    if not result.shadow_only or result.live_execution:
        raise ProspectiveGroundedRuntimeError(
            "prospective Phase43 result crossed shadow-only boundary"
        )
    if result.automatic_promotion:
        raise ProspectiveGroundedRuntimeError(
            "prospective Phase43 cannot auto-promote"
        )
    return result


def run_prospective_integrated_shadow_decision(
    asset_inputs: Mapping[str, AssetDecisionInput],
    *,
    timestamp: float,
    model_weights: Mapping[str, float],
    current_weights: Mapping[str, float],
    returns_by_asset: Mapping[str, Sequence[float]],
    config: IntegratedShadowConfig,
) -> IntegratedShadowDecision:
    """Run the real Phase43→44→52→53→54 path on current prospective data."""
    if not math.isfinite(float(timestamp)):
        raise ValueError("timestamp must be finite")
    if float(timestamp) < DEVELOPMENT_CUTOFF:
        raise ProspectiveGroundedRuntimeError(
            "prospective Phase54 requires a post-cutoff decision timestamp"
        )
    result = run_integrated_shadow_decision(
        asset_inputs,
        timestamp=float(timestamp),
        model_weights=model_weights,
        current_weights=current_weights,
        returns_by_asset=returns_by_asset,
        config=config,
        grounded_runner=run_prospective_grounded_phase43,
    )
    if not result.shadow_only or result.live_execution:
        raise ProspectiveGroundedRuntimeError(
            "prospective Phase54 result crossed shadow-only boundary"
        )
    if result.automatic_promotion:
        raise ProspectiveGroundedRuntimeError(
            "prospective Phase54 cannot auto-promote"
        )
    return result
