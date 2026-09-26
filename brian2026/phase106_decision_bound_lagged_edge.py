from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .phase54_integrated_shadow_decision import IntegratedShadowDecision
from .phase101_integrated_decision_shadow_runtime import (
    IntegratedDecisionShadowRuntime,
)
from .phase105_lagged_prospective_edge import (
    EvidenceFreshness,
    LaggedExpectedEdgeEstimate,
    LaggedReliabilityEvidence,
    eligible_expected_edge_bps_by_asset,
    estimate_lagged_expected_edge,
)

PHASE106_SCHEMA_VERSION = "brian.phase106-decision-bound-lagged-edge.v1"


class DecisionBoundLaggedEdgeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AssetLaggedEdgeContext:
    reliability: tuple[LaggedReliabilityEvidence, ...]
    round_trip_cost_bps: float | None
    cost_observed_at: float | None
    minimum_net_margin_bps: float = 2.0
    schema_version: str = PHASE106_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if self.round_trip_cost_bps is not None:
            if (
                not math.isfinite(float(self.round_trip_cost_bps))
                or float(self.round_trip_cost_bps) < 0
            ):
                raise ValueError(
                    "round_trip_cost_bps must be finite and non-negative"
                )
        if self.cost_observed_at is not None and not math.isfinite(
            float(self.cost_observed_at)
        ):
            raise ValueError("cost_observed_at must be finite")
        if (
            not math.isfinite(float(self.minimum_net_margin_bps))
            or float(self.minimum_net_margin_bps) < 0
        ):
            raise ValueError(
                "minimum_net_margin_bps must be finite and non-negative"
            )
        if not self.shadow_only or self.live_execution:
            raise ValueError("Phase106 edge context must remain shadow-only")


@dataclass(frozen=True, slots=True)
class DecisionBoundEdgeResolution:
    decision_pipeline_id: str
    estimates: tuple[tuple[str, LaggedExpectedEdgeEstimate], ...]
    expected_edge_bps_by_asset: tuple[tuple[str, float], ...]
    blocked_new_risk_assets: tuple[str, ...]
    required_new_risk_assets: tuple[str, ...]
    reasons_by_asset: tuple[tuple[str, tuple[str, ...]], ...]
    schema_version: str = PHASE106_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False
    automatic_promotion: bool = False

    def __post_init__(self) -> None:
        if len(self.decision_pipeline_id) != 64:
            raise ValueError("decision_pipeline_id must be a content hash")
        required = set(self.required_new_risk_assets)
        blocked = set(self.blocked_new_risk_assets)
        if not blocked <= required:
            raise ValueError("blocked assets must be required new-risk assets")
        edges = dict(self.expected_edge_bps_by_asset)
        if set(edges) & blocked:
            raise ValueError("blocked assets cannot carry expected edge")
        if not set(edges) <= required:
            raise ValueError("edge assets must be required new-risk assets")
        if not self.shadow_only or self.live_execution or self.automatic_promotion:
            raise ValueError("Phase106 resolution must remain hard shadow-only")

    @property
    def estimate_map(self) -> dict[str, LaggedExpectedEdgeEstimate]:
        return dict(self.estimates)

    @property
    def edge_map(self) -> dict[str, float]:
        return dict(self.expected_edge_bps_by_asset)


def _sign(value: float, *, eps: float = 1e-12) -> int:
    return 1 if value > eps else -1 if value < -eps else 0


def _required_new_risk_directions(
    decision: IntegratedShadowDecision,
) -> dict[str, int]:
    current = {
        str(asset): float(weight)
        for asset, weight in decision.current_weights.items()
    }
    planned = {
        str(asset): float(weight)
        for asset, weight in decision.final_planned_weights.items()
    }
    result: dict[str, int] = {}
    for asset in sorted(set(current) | set(planned)):
        c = current.get(asset, 0.0)
        p = planned.get(asset, 0.0)
        cs = _sign(c)
        ps = _sign(p)
        if ps == 0:
            continue
        if cs == 0:
            result[asset] = ps
            continue
        if ps == cs and abs(p) > abs(c) + 1e-12:
            result[asset] = ps
            continue
        if ps == -cs:
            result[asset] = ps
    return result


def _support_freshness(
    decision: IntegratedShadowDecision,
    asset: str,
    direction: int,
) -> tuple[tuple[str, ...], tuple[EvidenceFreshness, ...]]:
    result = decision.asset_results.get(asset)
    if result is None:
        return (), ()

    support_ids: set[str] = set()
    support_groups: set[str] = set()
    for claim in result.analyst_claims:
        if (
            int(claim.direction) == direction
            and float(claim.grounded_confidence) > 0
        ):
            support_ids.update(str(value) for value in claim.support_evidence_ids)
            support_groups.update(
                str(value) for value in claim.independent_support_groups
            )

    freshness: list[EvidenceFreshness] = []
    seen: set[tuple[str, float, str]] = set()
    for block in result.packet.blocks:
        if block.evidence_id not in support_ids:
            continue
        key = (
            str(block.independent_group),
            float(block.observed_at),
            str(block.horizon),
        )
        if key in seen:
            continue
        seen.add(key)
        freshness.append(EvidenceFreshness(
            group=key[0],
            observed_at=key[1],
            horizon=key[2],
        ))
    return tuple(sorted(support_groups)), tuple(
        sorted(
            freshness,
            key=lambda row: (row.group, row.observed_at, row.horizon),
        )
    )


def resolve_decision_bound_edges(
    decision: IntegratedShadowDecision,
    *,
    contexts_by_asset: Mapping[str, AssetLaggedEdgeContext],
) -> DecisionBoundEdgeResolution:
    if not decision.shadow_only or decision.live_execution:
        raise DecisionBoundLaggedEdgeError(
            "Phase54 decision crossed shadow-only boundary"
        )
    if decision.automatic_promotion:
        raise DecisionBoundLaggedEdgeError(
            "Phase54 automatic promotion is forbidden in Phase106"
        )
    if len(str(decision.pipeline_id)) != 64:
        raise DecisionBoundLaggedEdgeError(
            "Phase54 pipeline_id must be a content hash"
        )

    required = _required_new_risk_directions(decision)
    if not required:
        return DecisionBoundEdgeResolution(
            decision_pipeline_id=decision.pipeline_id,
            estimates=(),
            expected_edge_bps_by_asset=(),
            blocked_new_risk_assets=(),
            required_new_risk_assets=(),
            reasons_by_asset=(),
        )

    if decision.portfolio_book is None:
        raise DecisionBoundLaggedEdgeError(
            "new-risk decision requires Phase44 portfolio book"
        )

    estimates: dict[str, LaggedExpectedEdgeEstimate] = {}
    blocked: set[str] = set()
    reasons: dict[str, tuple[str, ...]] = {}

    for asset, direction in sorted(required.items()):
        context = contexts_by_asset.get(asset)
        if context is None:
            blocked.add(asset)
            reasons[asset] = ("lagged expected-edge context unavailable",)
            continue

        if (
            context.cost_observed_at is not None
            and float(context.cost_observed_at) > float(decision.timestamp)
        ):
            blocked.add(asset)
            reasons[asset] = ("post-decision execution cost rejected",)
            continue

        support_groups, freshness = _support_freshness(
            decision,
            asset,
            direction,
        )
        if not support_groups:
            blocked.add(asset)
            reasons[asset] = (
                "no grounded support groups align with planned new-risk direction",
            )
            continue

        reliability = tuple(
            row
            for row in context.reliability
            if row.group in support_groups
        )
        conviction = abs(float(
            decision.portfolio_book.blend.convictions.get(asset, 0.0)
        ))
        if not math.isfinite(conviction) or not 0 <= conviction <= 1:
            raise DecisionBoundLaggedEdgeError(
                f"Phase44 conviction for {asset} is outside [0,1]"
            )

        estimate = estimate_lagged_expected_edge(
            decision_timestamp=float(decision.timestamp),
            direction=direction,
            evidence_score=conviction,
            round_trip_cost_bps=context.round_trip_cost_bps,
            reliability=reliability,
            freshness=freshness,
            minimum_net_margin_bps=context.minimum_net_margin_bps,
        )
        estimates[asset] = estimate
        reasons[asset] = estimate.reasons
        if not estimate.eligible:
            blocked.add(asset)

    edge_map = eligible_expected_edge_bps_by_asset(estimates)
    for asset in required:
        if asset not in edge_map:
            blocked.add(asset)

    return DecisionBoundEdgeResolution(
        decision_pipeline_id=decision.pipeline_id,
        estimates=tuple(sorted(estimates.items())),
        expected_edge_bps_by_asset=tuple(sorted(edge_map.items())),
        blocked_new_risk_assets=tuple(sorted(blocked)),
        required_new_risk_assets=tuple(sorted(required)),
        reasons_by_asset=tuple(sorted(reasons.items())),
    )


class DecisionBoundLaggedEdgeRuntime:
    """Phase105 expected-edge gate in front of Phase101 durable execution.

    The caller may not inject arbitrary edge values. Phase106 resolves new-risk
    edges from the completed Phase54 decision's actual support groups plus
    lagged PIT reliability/cost context. Assets without an eligible edge are
    passed to Phase55 as blocked new risk, while reduce-only risk can continue.
    """

    def __init__(
        self,
        *,
        base_runtime: IntegratedDecisionShadowRuntime,
        contexts_by_asset: Mapping[str, AssetLaggedEdgeContext],
    ) -> None:
        if not isinstance(base_runtime, IntegratedDecisionShadowRuntime):
            # Keep production wiring strict while tests can use a compatible
            # subclass of the real Phase101 runtime.
            if not (
                hasattr(base_runtime, "process_integrated_decision")
                and hasattr(base_runtime, "worker")
            ):
                raise TypeError("base_runtime must be Phase101-compatible")
        self.base_runtime = base_runtime
        self.worker = base_runtime.worker
        self.contexts_by_asset = dict(contexts_by_asset)
        self.last_resolution: DecisionBoundEdgeResolution | None = None

    def process_integrated_decision(
        self,
        decision: IntegratedShadowDecision,
        *,
        expected_edge_bps_by_asset: Mapping[str, float],
        **kwargs,
    ):
        if expected_edge_bps_by_asset:
            raise DecisionBoundLaggedEdgeError(
                "caller-provided edge injection is forbidden in Phase106"
            )
        resolution = resolve_decision_bound_edges(
            decision,
            contexts_by_asset=self.contexts_by_asset,
        )
        self.last_resolution = resolution
        return self.base_runtime.process_integrated_decision(
            decision,
            expected_edge_bps_by_asset=resolution.edge_map,
            blocked_new_risk_assets=resolution.blocked_new_risk_assets,
            **kwargs,
        )
