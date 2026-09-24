from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from brian2026.global_sensor_mesh import SensorObservation
from brian2026.phase44_portfolio_brain import PortfolioRiskLimits
from brian2026.phase46_execution_simulator import LiquidityLevel, OrderBookSnapshot
from brian2026.phase52_covariance_risk import CovarianceRiskConfig
from brian2026.phase53_turnover_rebalance import TurnoverConfig
from brian2026.phase54_integrated_shadow_decision import (
    AssetDecisionInput,
    IntegratedShadowConfig,
)
from brian2026.phase56_pretrade_risk_engine import InstrumentRiskLimits
from brian2026.phase57_shadow_execution_cycle import ExecutionMarketInput
from brian2026.phase86_recovery_admission_interlock import RecoveryAdmissionState
from brian2026.phase94_binance_spot_recovery_evidence import (
    BinanceSpotRecoveryAssetEvidence,
    BinanceSpotRecoveryEvidenceBundle,
    PUBLIC_MARKET_HOSTS,
)
from brian2026.phase105_lagged_prospective_edge import LaggedReliabilityEvidence
from brian2026.phase106_decision_bound_lagged_edge import AssetLaggedEdgeContext
from brian2026.phase109_pit_edge_prefetch_builder import PointInTimeReturnSeries
from brian2026.phase110_supabase_grounded_market_prefetch import GroundedMarketPrefetch
from brian2026.phase115_crypto_shadow_readiness_gate import (
    CryptoShadowReadinessGate,
    PersistedReadinessProbe,
    PersistedReadinessState,
    SupabaseReadinessRpcResponseError,
    SupabaseReadinessRpcTransport,
)


TS = 1_790_000_000.0
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
            nan_policy="reject",
        ),
        turnover=TurnoverConfig(
            max_l1_turnover=1.0,
            risk_reduction_bypass=True,
        ),
    )


def _observation(
    eye: str,
    group: str,
    *,
    observed_at: float = TS - 30,
    horizon: str = "FAST_5_30M",
) -> SensorObservation:
    return SensorObservation(
        eye_id=eye,
        asset_id=ASSET,
        observed_at=observed_at,
        direction=1,
        strength=0.8,
        confidence=0.8,
        reliability=0.7,
        available=True,
        independent_group=group,
        source_ids=(f"src-{eye}",),
        horizon=horizon,
        reason="phase115 fixture",
    )


def _market_prefetch(
    *,
    observations=None,
    return_count: int = 30,
) -> GroundedMarketPrefetch:
    obs = tuple(observations or (
        _observation("eye-structure", "price_structure"),
        _observation("eye-momentum", "price_momentum"),
    ))
    item = AssetDecisionInput(
        snapshot={"structure_state": 1.0},
        observations=obs,
        source_kind_by_eye={
            row.eye_id: "market_snapshot"
            for row in obs
        },
    )
    returns = PointInTimeReturnSeries(
        asset_id=ASSET,
        values=tuple(
            0.001 * ((index % 5) - 2)
            for index in range(return_count)
        ),
        observed_from=TS - (return_count + 1) * 300,
        observed_until=TS - 300,
        source_ids=tuple(
            f"kline-{index:03d}"
            for index in range(return_count + 1)
        ),
    )
    return GroundedMarketPrefetch(
        decision_timestamp=TS,
        asset_inputs={ASSET: item},
        return_series_by_asset={ASSET: returns},
        marks={ASSET: 100.0},
        cost_asset_id_by_asset={ASSET: ASSET},
        common_return_buckets=tuple(
            TS - (return_count + 1 - index) * 300
            for index in range(return_count + 1)
        ),
    )


def _reliability(
    group: str,
    *,
    samples: int = 150,
    generated_at: float = TS - 900,
    window_end: float = TS - 1200,
) -> LaggedReliabilityEvidence:
    return LaggedReliabilityEvidence(
        group=group,
        sample_count=samples,
        bayesian_hit_rate=0.60,
        avg_signed_bps=25.0,
        avg_cost_adjusted_signed_bps=20.0,
        outcome_horizon_seconds=900,
        snapshot_window_end=window_end,
        snapshot_generated_at=generated_at,
    )


def _edge_context(
    *,
    reliability=None,
    cost: float | None = 4.0,
    cost_observed_at: float | None = TS - 30,
) -> AssetLaggedEdgeContext:
    return AssetLaggedEdgeContext(
        reliability=tuple(reliability or (
            _reliability("price_structure"),
            _reliability("price_momentum"),
        )),
        round_trip_cost_bps=cost,
        cost_observed_at=cost_observed_at,
        minimum_net_margin_bps=2.0,
    )


def _execution_bundle() -> BinanceSpotRecoveryEvidenceBundle:
    book = OrderBookSnapshot(
        timestamp=TS + 1,
        bids=(LiquidityLevel(100.0, 10.0),),
        asks=(LiquidityLevel(100.1, 10.0),),
    )
    return BinanceSpotRecoveryEvidenceBundle((
        BinanceSpotRecoveryAssetEvidence(
            asset_id=SYMBOL,
            market=ExecutionMarketInput(
                reference_price=100.05,
                tick_size=0.01,
                snapshots=(book,),
            ),
            risk_limits=InstrumentRiskLimits(min_notional=5.0),
            mark=100.05,
            observed_at=TS + 1,
            depth_last_update_id="12345",
            source_host=PUBLIC_MARKET_HOSTS[0],
            exchange_status="TRADING",
        ),
    ))


class _MarketReader:
    def __init__(self, result=None, error=None):
        self.result = _market_prefetch() if result is None else result
        self.error = error
        self.calls = []
        self.close_calls = 0

    def load(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result

    def close(self):
        self.close_calls += 1


class _EdgeReader:
    def __init__(self, context=None, error=None):
        self.context = _edge_context() if context is None else context
        self.error = error
        self.calls = []
        self.close_calls = 0

    def load_contexts(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {ASSET: self.context}

    def close(self):
        self.close_calls += 1


class _ExecutionProvider:
    def __init__(self, result=None, error=None):
        self.result = _execution_bundle() if result is None else result
        self.error = error
        self.calls = []
        self.close_calls = 0

    def collect(self, symbols):
        self.calls.append(tuple(symbols))
        if self.error is not None:
            raise self.error
        return self.result

    def close(self):
        self.close_calls += 1


class _PersistedProbe:
    def __init__(
        self,
        *,
        runtime=True,
        risk=True,
        blocked=False,
        risk_state="ACTIVE",
        pending_cycle_id=None,
        error=None,
    ):
        self.runtime = runtime
        self.risk = risk
        self.blocked = blocked
        self.risk_state = risk_state
        self.pending_cycle_id = pending_cycle_id
        self.error = error
        self.calls = []

    def read(self, *, runtime_id):
        self.calls.append(runtime_id)
        if self.error is not None:
            raise self.error
        admission = (
            RecoveryAdmissionState(
                runtime_id=runtime_id,
                status="RECOVERY_BARRIER",
                blocked=True,
                original_cycle_id="a" * 64,
                cancel_risk_receipt_id="b" * 64,
                reason="unresolved after-start recovery",
            )
            if self.blocked
            else RecoveryAdmissionState(
                runtime_id=runtime_id,
                status="OPEN",
                blocked=False,
            )
        )
        return PersistedReadinessState(
            runtime=(
                SimpleNamespace(
                    version=4,
                    pending_cycle_id=self.pending_cycle_id,
                )
                if self.runtime
                else None
            ),
            risk=(
                SimpleNamespace(
                    version=3,
                    current_state=self.risk_state,
                )
                if self.risk
                else None
            ),
            admission=admission,
        )


def _gate(
    *,
    market=None,
    edge=None,
    execution=None,
    persisted=None,
    reliability_max_age_seconds=7200.0,
    owns_resources=False,
):
    return CryptoShadowReadinessGate(
        market_reader=market or _MarketReader(),
        edge_reader=edge or _EdgeReader(),
        execution_provider=execution or _ExecutionProvider(),
        persisted_probe=persisted or _PersistedProbe(),
        reliability_max_age_seconds=reliability_max_age_seconds,
        clock=lambda: TS,
        owns_resources=owns_resources,
    )


def _check_map(report):
    return {row.code: row for row in report.checks}


def test_all_green_inputs_produce_ready_edge_bound_shadow_report() -> None:
    gate = _gate()

    report = gate.run(
        runtime_id="runtime-115",
        asset_ids=(ASSET,),
        config=_config(),
        decision_timestamp=TS,
    )

    assert report.status == "READY_FOR_EDGE_BOUND_SHADOW"
    assert report.safe_to_invoke_shadow_worker is True
    assert report.new_risk_ready is True
    assert len(report.report_id) == 64
    checks = _check_map(report)
    assert all(row.status == "PASS" for row in checks.values())
    assert set(checks) == {
        "PERSISTED_RUNTIME",
        "PERSISTED_RISK",
        "RECOVERY_ADMISSION",
        "RUNTIME_CONTINUITY",
        "RISK_STATE",
        "MARKET_PREFETCH",
        "SENSOR_FRESHNESS",
        "COVARIANCE_HISTORY",
        "RELIABILITY_FRESHNESS",
        "RELIABILITY_MATURITY",
        "DYNAMIC_COST",
        "PUBLIC_EXECUTION_EVIDENCE",
    }
    assert report.read_only is True
    assert report.shadow_only is True
    assert report.live_execution is False


def test_missing_persisted_runtime_or_risk_is_not_schedule_ready() -> None:
    report = _gate(
        persisted=_PersistedProbe(runtime=False, risk=False),
    ).run(
        runtime_id="runtime-115",
        asset_ids=(ASSET,),
        config=_config(),
        decision_timestamp=TS,
    )

    assert report.status == "NOT_READY"
    assert report.safe_to_invoke_shadow_worker is False
    assert report.new_risk_ready is False
    checks = _check_map(report)
    assert checks["PERSISTED_RUNTIME"].status == "FAIL"
    assert checks["PERSISTED_RISK"].status == "FAIL"
    assert checks["RECOVERY_ADMISSION"].status == "PASS"


def test_recovery_barrier_blocks_core_readiness() -> None:
    report = _gate(
        persisted=_PersistedProbe(blocked=True),
    ).run(
        runtime_id="runtime-115",
        asset_ids=(ASSET,),
        config=_config(),
        decision_timestamp=TS,
    )

    assert report.status == "NOT_READY"
    assert _check_map(report)["RECOVERY_ADMISSION"].status == "FAIL"


def test_stale_reliability_and_missing_cost_are_safe_fail_closed_only() -> None:
    stale = _edge_context(
        reliability=(
            _reliability(
                "price_structure",
                generated_at=TS - 20000,
                window_end=TS - 21000,
            ),
            _reliability(
                "price_momentum",
                generated_at=TS - 20000,
                window_end=TS - 21000,
            ),
        ),
        cost=None,
        cost_observed_at=None,
    )
    report = _gate(
        edge=_EdgeReader(context=stale),
    ).run(
        runtime_id="runtime-115",
        asset_ids=(ASSET,),
        config=_config(),
        decision_timestamp=TS,
    )

    assert report.status == "SAFE_FAIL_CLOSED_ONLY"
    assert report.safe_to_invoke_shadow_worker is True
    assert report.new_risk_ready is False
    checks = _check_map(report)
    assert checks["RELIABILITY_FRESHNESS"].status == "FAIL"
    assert checks["DYNAMIC_COST"].status == "FAIL"
    assert checks["RELIABILITY_MATURITY"].status == "PASS"


def test_immature_reliability_is_reported_without_fabricating_edge() -> None:
    immature = _edge_context(
        reliability=(
            _reliability("price_structure", samples=30),
            _reliability("price_momentum", samples=20),
        )
    )
    report = _gate(
        edge=_EdgeReader(context=immature),
    ).run(
        runtime_id="runtime-115",
        asset_ids=(ASSET,),
        config=_config(),
        decision_timestamp=TS,
    )

    assert report.status == "SAFE_FAIL_CLOSED_ONLY"
    assert _check_map(report)["RELIABILITY_MATURITY"].status == "FAIL"


def test_stale_sensor_and_short_covariance_history_block_new_risk_only() -> None:
    market = _market_prefetch(
        observations=(
            _observation(
                "eye-old",
                "price_structure",
                observed_at=TS - 7200,
            ),
        ),
        return_count=10,
    )
    report = _gate(
        market=_MarketReader(result=market),
    ).run(
        runtime_id="runtime-115",
        asset_ids=(ASSET,),
        config=_config(),
        decision_timestamp=TS,
    )

    assert report.status == "SAFE_FAIL_CLOSED_ONLY"
    checks = _check_map(report)
    assert checks["SENSOR_FRESHNESS"].status == "FAIL"
    assert checks["COVARIANCE_HISTORY"].status == "FAIL"


def test_market_or_execution_transport_failure_blocks_core_readiness() -> None:
    market_report = _gate(
        market=_MarketReader(error=RuntimeError("sensor source unavailable")),
    ).run(
        runtime_id="runtime-115",
        asset_ids=(ASSET,),
        config=_config(),
        decision_timestamp=TS,
    )
    assert market_report.status == "NOT_READY"
    assert _check_map(market_report)["MARKET_PREFETCH"].status == "FAIL"

    execution_report = _gate(
        execution=_ExecutionProvider(error=RuntimeError("depth unavailable")),
    ).run(
        runtime_id="runtime-115",
        asset_ids=(ASSET,),
        config=_config(),
        decision_timestamp=TS,
    )
    assert execution_report.status == "NOT_READY"
    assert (
        _check_map(execution_report)["PUBLIC_EXECUTION_EVIDENCE"].status
        == "FAIL"
    )


def test_report_identity_is_deterministic_for_same_frozen_inputs() -> None:
    gate = _gate()
    first = gate.run(
        runtime_id="runtime-115",
        asset_ids=(ASSET,),
        config=_config(),
        decision_timestamp=TS,
    )
    second = gate.run(
        runtime_id="runtime-115",
        asset_ids=(ASSET,),
        config=_config(),
        decision_timestamp=TS,
    )
    assert first.report_id == second.report_id


def test_readiness_rpc_transport_is_read_only_allowlisted_and_accepts_null_heads() -> None:
    secret = "sb_secret_phase115_abcdefghijklmnopqrstuvwxyz"
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.method == "POST"
        assert request.headers["apikey"] == secret
        assert "authorization" not in request.headers
        if request.url.path.endswith(
            "/rpc/brian_read_shadow_recovery_admission"
        ):
            return httpx.Response(
                200,
                json={
                    "runtime_id": "runtime-115",
                    "status": "OPEN",
                    "blocked": False,
                },
            )
        return httpx.Response(200, json=None)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = SupabaseReadinessRpcTransport(
        project_url="https://runtime.supabase.co",
        key_source="SUPABASE_SECRET_KEY",
        api_key=secret,
        client=client,
    )
    try:
        state = PersistedReadinessProbe(transport).read(
            runtime_id="runtime-115"
        )
        assert state.runtime is None
        assert state.risk is None
        assert state.admission.status == "OPEN"

        with pytest.raises(
            Exception,
            match="not allowed",
        ):
            transport("brian_commit_shadow_runtime_checkpoint", {})
    finally:
        client.close()

    assert len(seen) == 3


def test_readiness_rpc_http_error_is_sanitized() -> None:
    secret = "sb_secret_phase115_never_echo"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "code": "PGRST202",
                "message": "function not found",
                "hint": "schema cache",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = SupabaseReadinessRpcTransport(
        project_url="https://runtime.supabase.co",
        key_source="SUPABASE_SECRET_KEY",
        api_key=secret,
        client=client,
    )
    try:
        with pytest.raises(
            SupabaseReadinessRpcResponseError,
            match="HTTP 404",
        ) as exc:
            transport(
                "brian_read_shadow_runtime_checkpoint",
                {"p_runtime_id": "runtime-115"},
            )
    finally:
        client.close()

    assert secret not in str(exc.value)


def test_owned_gate_closes_owned_external_resources_once() -> None:
    market = _MarketReader()
    edge = _EdgeReader()
    execution = _ExecutionProvider()
    gate = _gate(
        market=market,
        edge=edge,
        execution=execution,
        owns_resources=True,
    )

    assert gate.close() is True
    assert gate.close() is True
    assert market.close_calls == 1
    assert edge.close_calls == 1
    assert execution.close_calls == 1

@pytest.mark.parametrize("risk_state", ["REDUCING", "HALTED"])
def test_nonactive_persisted_risk_blocks_new_risk_but_not_safe_invocation(
    risk_state,
) -> None:
    report = _gate(
        persisted=_PersistedProbe(risk_state=risk_state),
    ).run(
        runtime_id="runtime-115",
        asset_ids=(ASSET,),
        config=_config(),
        decision_timestamp=TS,
    )

    assert report.status == "SAFE_FAIL_CLOSED_ONLY"
    assert report.safe_to_invoke_shadow_worker is True
    assert report.new_risk_ready is False
    assert _check_map(report)["RISK_STATE"].status == "FAIL"


def test_pending_durable_cycle_blocks_new_risk_until_recovery_resolves_it() -> None:
    report = _gate(
        persisted=_PersistedProbe(pending_cycle_id="c" * 64),
    ).run(
        runtime_id="runtime-115",
        asset_ids=(ASSET,),
        config=_config(),
        decision_timestamp=TS,
    )

    assert report.status == "SAFE_FAIL_CLOSED_ONLY"
    assert report.safe_to_invoke_shadow_worker is True
    assert report.new_risk_ready is False
    assert _check_map(report)["RUNTIME_CONTINUITY"].status == "FAIL"

