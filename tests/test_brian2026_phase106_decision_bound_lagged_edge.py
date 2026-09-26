from __future__ import annotations

from types import SimpleNamespace

import pytest

from brian2026.phase105_lagged_prospective_edge import (
    LaggedReliabilityEvidence,
)
from brian2026.phase106_decision_bound_lagged_edge import (
    AssetLaggedEdgeContext,
    DecisionBoundLaggedEdgeError,
    DecisionBoundLaggedEdgeRuntime,
    resolve_decision_bound_edges,
)


TS = 1_790_000_000.0
PIPELINE = "p" * 64


def _row(group: str, *, samples=250, signed=30.0):
    return LaggedReliabilityEvidence(
        group=group,
        sample_count=samples,
        bayesian_hit_rate=0.62,
        avg_signed_bps=signed,
        avg_cost_adjusted_signed_bps=signed - 5.0,
        outcome_horizon_seconds=900,
        snapshot_window_end=TS - 3600,
        snapshot_generated_at=TS - 1800,
    )


def _block(evidence_id: str, group: str, *, observed_at=TS - 60):
    return SimpleNamespace(
        evidence_id=evidence_id,
        independent_group=group,
        observed_at=observed_at,
        horizon="FAST_5_30M",
    )


def _claim(direction: int, *pairs: tuple[str, str]):
    return SimpleNamespace(
        direction=direction,
        grounded_confidence=0.8,
        support_evidence_ids=tuple(evidence_id for evidence_id, _ in pairs),
        independent_support_groups=tuple(group for _, group in pairs),
    )


def _asset_result(direction: int = 1):
    pairs = (("ev-price", "price_structure"), ("ev-deriv", "derivatives"))
    return SimpleNamespace(
        analyst_claims=(_claim(direction, *pairs),),
        packet=SimpleNamespace(
            blocks=(
                _block("ev-price", "price_structure"),
                _block("ev-deriv", "derivatives", observed_at=TS - 90),
            )
        ),
    )


def _decision(
    *,
    current=None,
    planned=None,
    direction=1,
    convictions=None,
    asset_results=None,
):
    return SimpleNamespace(
        shadow_only=True,
        live_execution=False,
        automatic_promotion=False,
        pipeline_id=PIPELINE,
        timestamp=TS,
        current_weights={} if current is None else dict(current),
        final_planned_weights=(
            {"BTCUSDT": 0.20} if planned is None else dict(planned)
        ),
        portfolio_book=SimpleNamespace(
            blend=SimpleNamespace(
                convictions=(
                    {"BTCUSDT": 0.80}
                    if convictions is None
                    else dict(convictions)
                )
            )
        ),
        asset_results=(
            {"BTCUSDT": _asset_result(direction)}
            if asset_results is None
            else dict(asset_results)
        ),
    )


def _context(*, rows=None, cost=4.0, cost_at=TS - 1):
    return AssetLaggedEdgeContext(
        reliability=tuple(
            rows
            if rows is not None
            else (
                _row("price_structure", signed=32.0),
                _row("derivatives", signed=26.0),
            )
        ),
        round_trip_cost_bps=cost,
        cost_observed_at=cost_at,
        minimum_net_margin_bps=2.0,
    )


def test_decision_support_groups_bind_lagged_edge_and_allow_new_risk() -> None:
    resolution = resolve_decision_bound_edges(
        _decision(),
        contexts_by_asset={"BTCUSDT": _context()},
    )

    assert resolution.required_new_risk_assets == ("BTCUSDT",)
    assert resolution.blocked_new_risk_assets == ()
    assert set(resolution.edge_map) == {"BTCUSDT"}
    assert resolution.edge_map["BTCUSDT"] > 2.0
    estimate = resolution.estimate_map["BTCUSDT"]
    assert estimate.eligible is True
    assert estimate.recommendation == "ALLOW_EDGE"
    assert {row.group for row in estimate.contributions} == {
        "price_structure",
        "derivatives",
    }


def test_only_actual_direction_support_groups_can_contribute() -> None:
    result = _asset_result(direction=-1)
    decision = _decision(
        direction=1,
        asset_results={"BTCUSDT": result},
    )

    resolution = resolve_decision_bound_edges(
        decision,
        contexts_by_asset={"BTCUSDT": _context()},
    )

    assert resolution.blocked_new_risk_assets == ("BTCUSDT",)
    assert resolution.edge_map == {}
    assert resolution.estimate_map == {}
    assert "no grounded support groups align" in resolution.reasons_by_asset[0][1][0]


def test_insufficient_mature_support_blocks_new_risk_without_fabricated_edge() -> None:
    resolution = resolve_decision_bound_edges(
        _decision(),
        contexts_by_asset={
            "BTCUSDT": _context(
                rows=(
                    _row("price_structure", samples=250),
                    _row("derivatives", samples=99),
                )
            )
        },
    )

    assert resolution.blocked_new_risk_assets == ("BTCUSDT",)
    assert resolution.edge_map == {}
    estimate = resolution.estimate_map["BTCUSDT"]
    assert estimate.eligible is False
    assert estimate.recommendation == "INSUFFICIENT_LAGGED_EVIDENCE"


def test_future_cost_is_rejected_before_edge_estimation() -> None:
    resolution = resolve_decision_bound_edges(
        _decision(),
        contexts_by_asset={
            "BTCUSDT": _context(cost_at=TS + 0.001),
        },
    )

    assert resolution.blocked_new_risk_assets == ("BTCUSDT",)
    assert resolution.edge_map == {}
    assert resolution.estimate_map == {}
    assert resolution.reasons_by_asset == (
        ("BTCUSDT", ("post-decision execution cost rejected",)),
    )


def test_pure_risk_reduction_does_not_require_edge_context() -> None:
    decision = _decision(
        current={"BTCUSDT": 0.30},
        planned={"BTCUSDT": 0.10},
    )

    resolution = resolve_decision_bound_edges(
        decision,
        contexts_by_asset={},
    )

    assert resolution.required_new_risk_assets == ()
    assert resolution.blocked_new_risk_assets == ()
    assert resolution.edge_map == {}
    assert resolution.estimate_map == {}


def test_reversal_requires_edge_for_opposite_side_open() -> None:
    decision = _decision(
        current={"BTCUSDT": 0.30},
        planned={"BTCUSDT": -0.20},
        direction=-1,
        convictions={"BTCUSDT": -0.75},
        asset_results={"BTCUSDT": _asset_result(-1)},
    )

    resolution = resolve_decision_bound_edges(
        decision,
        contexts_by_asset={},
    )

    assert resolution.required_new_risk_assets == ("BTCUSDT",)
    assert resolution.blocked_new_risk_assets == ("BTCUSDT",)
    assert resolution.edge_map == {}


class _BaseRuntime:
    def __init__(self):
        self.worker = SimpleNamespace(runtime_id="runtime-106")
        self.calls = []

    def process_integrated_decision(self, decision, **kwargs):
        self.calls.append((decision, kwargs))
        return SimpleNamespace(status="SHADOW_EXECUTED")


def test_runtime_forbids_caller_edge_injection() -> None:
    runtime = DecisionBoundLaggedEdgeRuntime(
        base_runtime=_BaseRuntime(),
        contexts_by_asset={"BTCUSDT": _context()},
    )

    with pytest.raises(
        DecisionBoundLaggedEdgeError,
        match="caller-provided edge injection",
    ):
        runtime.process_integrated_decision(
            _decision(),
            expected_edge_bps_by_asset={"BTCUSDT": 999.0},
        )


def test_runtime_passes_only_resolved_edges_and_block_set_to_phase101() -> None:
    base = _BaseRuntime()
    runtime = DecisionBoundLaggedEdgeRuntime(
        base_runtime=base,
        contexts_by_asset={"BTCUSDT": _context()},
    )
    decision = _decision()

    result = runtime.process_integrated_decision(
        decision,
        expected_edge_bps_by_asset={},
        max_slippage_bps=20.0,
        ttl_seconds=60,
        equity_usd=1000.0,
        available_cash_usd=500.0,
        markets={},
        risk_limits_by_asset={},
        marks={},
        worker_token="normal-106",
        claim_seconds=30,
        observed_at=TS + 1,
        source_ref="phase106:test",
    )

    assert result.status == "SHADOW_EXECUTED"
    assert len(base.calls) == 1
    seen_decision, kwargs = base.calls[0]
    assert seen_decision is decision
    assert set(kwargs["expected_edge_bps_by_asset"]) == {"BTCUSDT"}
    assert kwargs["expected_edge_bps_by_asset"]["BTCUSDT"] > 2.0
    assert kwargs["blocked_new_risk_assets"] == ()
    assert kwargs["worker_token"] == "normal-106"


def test_runtime_passes_blocked_asset_when_edge_is_not_mature() -> None:
    base = _BaseRuntime()
    runtime = DecisionBoundLaggedEdgeRuntime(
        base_runtime=base,
        contexts_by_asset={
            "BTCUSDT": _context(
                rows=(
                    _row("price_structure", samples=50),
                    _row("derivatives", samples=50),
                )
            )
        },
    )

    runtime.process_integrated_decision(
        _decision(),
        expected_edge_bps_by_asset={},
    )

    _, kwargs = base.calls[0]
    assert kwargs["expected_edge_bps_by_asset"] == {}
    assert kwargs["blocked_new_risk_assets"] == ("BTCUSDT",)
