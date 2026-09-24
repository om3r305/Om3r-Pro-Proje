from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import httpx

from .phase43_grounded_analysts import HORIZON_MAX_AGE_SECONDS
from .phase54_integrated_shadow_decision import IntegratedShadowConfig
from .phase70_durable_runtime_store import (
    DurableRuntimeStore,
    StoredRuntimeCheckpoint,
)
from .phase73_operational_risk_store import (
    OperationalRiskStore,
    StoredOperationalRiskLedger,
)
from .phase86_recovery_admission_interlock import (
    RecoveryAdmissionInterlockStore,
    RecoveryAdmissionState,
)
from .phase91_supabase_rpc_transport import (
    SupabaseRecoveryRpcConfigurationError,
    _read_scoped_project_url,
    _read_scoped_secret_key_from_env,
    _sanitize_error_payload,
    _validate_server_key,
)
from .phase94_binance_spot_recovery_evidence import (
    BinanceSpotRecoveryEvidenceBundle,
    BinanceSpotRecoveryEvidenceProvider,
)
from .phase105_lagged_prospective_edge import (
    MIN_MATURE_GROUPS,
    MIN_MATURE_SAMPLES,
)
from .phase106_decision_bound_lagged_edge import AssetLaggedEdgeContext
from .phase108_supabase_lagged_edge_reader import SupabaseLaggedEdgeReader
from .phase110_supabase_grounded_market_prefetch import GroundedMarketPrefetch
from .phase113_binance_grounded_market_prefetch import (
    BinanceGroundedMarketPrefetchReader,
)

PHASE115_SCHEMA_VERSION = "brian.phase115-crypto-shadow-readiness-gate.v1"

CheckStatus = Literal["PASS", "FAIL"]
CheckScope = Literal["CORE", "NEW_RISK"]

_READ_RPC_ALLOWLIST = frozenset({
    "brian_read_shadow_runtime_checkpoint",
    "brian_read_operational_risk_ledger",
    "brian_read_shadow_recovery_admission",
})
_CRYPTO_ASSET = re.compile(r"^crypto:([A-Z0-9]{2,20}USDT)$")


class CryptoShadowReadinessError(RuntimeError):
    pass


class SupabaseReadinessRpcError(CryptoShadowReadinessError):
    pass


class SupabaseReadinessRpcResponseError(SupabaseReadinessRpcError):
    pass


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    code: str
    scope: CheckScope
    status: CheckStatus
    detail: str
    assets: tuple[str, ...] = ()
    schema_version: str = PHASE115_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("readiness check code is required")
        if self.scope not in ("CORE", "NEW_RISK"):
            raise ValueError("invalid readiness check scope")
        if self.status not in ("PASS", "FAIL"):
            raise ValueError("invalid readiness check status")
        if not self.detail.strip():
            raise ValueError("readiness check detail is required")
        if tuple(sorted(set(self.assets))) != self.assets:
            raise ValueError("readiness check assets must be unique and sorted")


@dataclass(frozen=True, slots=True)
class CryptoShadowReadinessReport:
    runtime_id: str
    observed_at: float
    status: str
    safe_to_invoke_shadow_worker: bool
    new_risk_ready: bool
    checks: tuple[ReadinessCheck, ...]
    report_id: str = field(init=False)
    schema_version: str = PHASE115_SCHEMA_VERSION
    read_only: bool = True
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError("runtime_id is required")
        if not math.isfinite(float(self.observed_at)):
            raise ValueError("observed_at must be finite")
        if not self.checks:
            raise ValueError("readiness report requires checks")
        expected_safe = all(
            row.status == "PASS"
            for row in self.checks
            if row.scope == "CORE"
        )
        expected_new_risk = expected_safe and all(
            row.status == "PASS"
            for row in self.checks
            if row.scope == "NEW_RISK"
        )
        if self.safe_to_invoke_shadow_worker != expected_safe:
            raise ValueError("safe_to_invoke_shadow_worker disagrees with CORE checks")
        if self.new_risk_ready != expected_new_risk:
            raise ValueError("new_risk_ready disagrees with readiness checks")
        expected_status = (
            "READY_FOR_EDGE_BOUND_SHADOW"
            if expected_new_risk
            else "SAFE_FAIL_CLOSED_ONLY"
            if expected_safe
            else "NOT_READY"
        )
        if self.status != expected_status:
            raise ValueError("readiness status disagrees with checks")
        if not self.read_only or not self.shadow_only or self.live_execution:
            raise ValueError("Phase115 report must remain read-only shadow-only")
        identity = self.identity_payload()
        object.__setattr__(
            self,
            "report_id",
            hashlib.sha256(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "runtime_id": self.runtime_id,
            "observed_at": float(self.observed_at),
            "status": self.status,
            "safe_to_invoke_shadow_worker": self.safe_to_invoke_shadow_worker,
            "new_risk_ready": self.new_risk_ready,
            "checks": [asdict(row) for row in self.checks],
            "read_only": self.read_only,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["report_id"] = self.report_id
        return payload


@dataclass(frozen=True, slots=True)
class PersistedReadinessState:
    runtime: StoredRuntimeCheckpoint | None
    risk: StoredOperationalRiskLedger | None
    admission: RecoveryAdmissionState
    schema_version: str = PHASE115_SCHEMA_VERSION
    read_only: bool = True
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not self.read_only or not self.shadow_only or self.live_execution:
            raise ValueError("persisted readiness state must remain read-only shadow-only")


class SupabaseReadinessRpcTransport:
    """GET-equivalent, read-RPC-only PostgREST transport for Phase115.

    Supabase exposes Postgres functions through POST /rpc even when the
    underlying functions are pure reads. The allowlist contains only the three
    existing read functions needed to validate Phase70, Phase73 and Phase86.
    There is no acquire/commit/claim/cancel/dispatch/order surface here.
    """

    def __init__(
        self,
        *,
        project_url: str,
        key_source: str,
        api_key: str,
        timeout_seconds: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not project_url.startswith("https://") and not (
            project_url.startswith("http://127.0.0.1")
            or project_url.startswith("http://localhost")
        ):
            raise ValueError("project_url must use https outside localhost")
        if not key_source.strip():
            raise ValueError("key_source is required")
        _validate_server_key(api_key, key_source)
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.project_url = project_url.rstrip("/")
        self.key_source = key_source
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=float(timeout_seconds),
            follow_redirects=False,
            trust_env=False,
        )

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> "SupabaseReadinessRpcTransport":
        source = os.environ if env is None else env
        project_url, _ = _read_scoped_project_url(
            source,
            "BRIAN_RUNTIME",
        )
        api_key, key_source = _read_scoped_secret_key_from_env(
            source,
            "BRIAN_RUNTIME",
        )
        raw_timeout = source.get(
            "BRIAN_READINESS_RPC_TIMEOUT_SECONDS",
            "10",
        )
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError) as exc:
            raise SupabaseRecoveryRpcConfigurationError(
                "BRIAN_READINESS_RPC_TIMEOUT_SECONDS must be numeric"
            ) from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise SupabaseRecoveryRpcConfigurationError(
                "BRIAN_READINESS_RPC_TIMEOUT_SECONDS must be positive"
            )
        return cls(
            project_url=project_url,
            key_source=key_source,
            api_key=api_key,
            timeout_seconds=timeout,
            client=client,
        )

    def __call__(
        self,
        function_name: str,
        params: Mapping[str, object],
    ) -> object:
        if function_name not in _READ_RPC_ALLOWLIST:
            raise SupabaseRecoveryRpcConfigurationError(
                f"RPC function is not allowed by Phase115: {function_name}"
            )
        if not isinstance(params, Mapping):
            raise TypeError("RPC params must be a mapping")
        try:
            response = self._client.post(
                f"{self.project_url}/rest/v1/rpc/{function_name}",
                headers={
                    "apikey": self._api_key,
                    "accept": "application/json",
                    "content-type": "application/json",
                    "user-agent": "brian-phase115-readiness/1",
                },
                json=dict(params),
            )
        except httpx.TimeoutException as exc:
            raise SupabaseReadinessRpcError(
                f"Supabase readiness RPC timeout for {function_name}"
            ) from exc
        except httpx.TransportError as exc:
            raise SupabaseReadinessRpcError(
                f"Supabase readiness RPC transport failure for {function_name}"
            ) from exc
        if response.status_code < 200 or response.status_code >= 300:
            detail = _sanitize_error_payload(response)
            raise SupabaseReadinessRpcResponseError(
                f"Supabase readiness RPC {function_name} returned HTTP "
                f"{response.status_code}: {detail}"
            )
        if not response.content.strip():
            return None
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise SupabaseReadinessRpcResponseError(
                f"Supabase readiness RPC {function_name} returned invalid JSON"
            ) from exc
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise SupabaseReadinessRpcResponseError(
                f"Supabase readiness RPC {function_name} must return object or null"
            )
        return {str(key): value for key, value in payload.items()}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class PersistedReadinessProbe:
    def __init__(self, rpc: Callable[[str, Mapping[str, object]], object]) -> None:
        if not callable(rpc):
            raise TypeError("rpc must be callable")
        self.runtime_store = DurableRuntimeStore(rpc)
        self.risk_store = OperationalRiskStore(rpc)
        self.admission_store = RecoveryAdmissionInterlockStore(rpc)

    def read(self, *, runtime_id: str) -> PersistedReadinessState:
        if not runtime_id.strip():
            raise ValueError("runtime_id is required")
        runtime = self.runtime_store.load(runtime_id=runtime_id)
        risk = self.risk_store.load(runtime_id=runtime_id)
        admission = self.admission_store.read(runtime_id=runtime_id)
        return PersistedReadinessState(
            runtime=runtime,
            risk=risk,
            admission=admission,
        )


def _crypto_symbol(asset_id: str) -> str:
    match = _CRYPTO_ASSET.fullmatch(str(asset_id).strip())
    if match is None:
        raise ValueError(
            f"Phase115 supports canonical crypto:*USDT only: {asset_id!r}"
        )
    return match.group(1)


def _check(
    code: str,
    scope: CheckScope,
    passed: bool,
    detail: str,
    assets: Sequence[str] = (),
) -> ReadinessCheck:
    return ReadinessCheck(
        code=code,
        scope=scope,
        status="PASS" if passed else "FAIL",
        detail=str(detail)[:500],
        assets=tuple(sorted(set(str(value) for value in assets))),
    )


class CryptoShadowReadinessGate:
    """Read-only end-to-end gate before scheduling Phase114.

    The gate intentionally performs no runtime bootstrap and no recovery/risk
    mutation. It verifies that the already-persisted authority needed by
    Phase101/107 exists, that Phase86 is OPEN, that current grounded inputs and
    covariance history are causal, that Phase105 can see mature lagged evidence
    plus decision-time cost, and that fresh public Binance execution evidence is
    available.

    The reliability producer in the repository is scheduled hourly
    (brian-sensor-reliability-shadow-hourly). A two-hour default freshness
    budget allows one missed refresh without silently accepting multi-day stale
    expected-edge calibration.
    """

    def __init__(
        self,
        *,
        market_reader,
        edge_reader: SupabaseLaggedEdgeReader,
        execution_provider: BinanceSpotRecoveryEvidenceProvider,
        persisted_probe: PersistedReadinessProbe,
        reliability_max_age_seconds: float = 2 * 60 * 60.0,
        clock: Callable[[], float] = time.time,
        owns_resources: bool = False,
    ) -> None:
        for label, value, method in (
            ("market_reader", market_reader, "load"),
            ("edge_reader", edge_reader, "load_contexts"),
            ("execution_provider", execution_provider, "collect"),
            ("persisted_probe", persisted_probe, "read"),
        ):
            if not callable(getattr(value, method, None)):
                raise TypeError(f"{label} must expose callable {method}")
        if (
            not math.isfinite(float(reliability_max_age_seconds))
            or reliability_max_age_seconds <= 0
        ):
            raise ValueError("reliability_max_age_seconds must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.market_reader = market_reader
        self.edge_reader = edge_reader
        self.execution_provider = execution_provider
        self.persisted_probe = persisted_probe
        self.reliability_max_age_seconds = float(
            reliability_max_age_seconds
        )
        self.clock = clock
        self._owns_resources = bool(owns_resources)
        self._closed = False

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        market_reader_factory=BinanceGroundedMarketPrefetchReader.from_env,
        edge_reader_factory=SupabaseLaggedEdgeReader.from_env,
        execution_provider_factory=BinanceSpotRecoveryEvidenceProvider,
        rpc_transport_factory=SupabaseReadinessRpcTransport.from_env,
        clock: Callable[[], float] = time.time,
    ) -> "CryptoShadowReadinessGate":
        source = os.environ if env is None else env
        raw_age = source.get(
            "BRIAN_READINESS_RELIABILITY_MAX_AGE_SECONDS",
            str(2 * 60 * 60),
        )
        try:
            max_age = float(raw_age)
        except (TypeError, ValueError) as exc:
            raise SupabaseRecoveryRpcConfigurationError(
                "BRIAN_READINESS_RELIABILITY_MAX_AGE_SECONDS must be numeric"
            ) from exc
        if not math.isfinite(max_age) or max_age <= 0:
            raise SupabaseRecoveryRpcConfigurationError(
                "BRIAN_READINESS_RELIABILITY_MAX_AGE_SECONDS must be positive"
            )

        market_reader = None
        edge_reader = None
        execution_provider = None
        rpc_transport = None
        try:
            market_reader = market_reader_factory(env=source)
            edge_reader = edge_reader_factory(env=source)
            execution_provider = execution_provider_factory()
            rpc_transport = rpc_transport_factory(env=source)
            return cls(
                market_reader=market_reader,
                edge_reader=edge_reader,
                execution_provider=execution_provider,
                persisted_probe=PersistedReadinessProbe(rpc_transport),
                reliability_max_age_seconds=max_age,
                clock=clock,
                owns_resources=True,
            )
        except Exception:
            for resource in (
                rpc_transport,
                execution_provider,
                edge_reader,
                market_reader,
            ):
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

    def run(
        self,
        *,
        runtime_id: str,
        asset_ids: Sequence[str],
        config: IntegratedShadowConfig,
        minimum_net_margin_bps: float = 2.0,
        decision_timestamp: float | None = None,
    ) -> CryptoShadowReadinessReport:
        if self._closed:
            raise CryptoShadowReadinessError("Phase115 gate is closed")
        if not runtime_id.strip():
            raise ValueError("runtime_id is required")
        assets = tuple(sorted({str(value).strip() for value in asset_ids}))
        if not assets:
            raise ValueError("asset_ids are required")
        for asset in assets:
            _crypto_symbol(asset)
        if (
            not math.isfinite(float(minimum_net_margin_bps))
            or minimum_net_margin_bps < 0
        ):
            raise ValueError(
                "minimum_net_margin_bps must be finite and non-negative"
            )
        observed_at = (
            float(self.clock())
            if decision_timestamp is None
            else float(decision_timestamp)
        )
        if not math.isfinite(observed_at):
            raise ValueError("decision_timestamp must be finite")

        checks: list[ReadinessCheck] = []

        persisted: PersistedReadinessState | None = None
        try:
            persisted = self.persisted_probe.read(runtime_id=runtime_id)
        except Exception as exc:
            checks.extend((
                _check(
                    "PERSISTED_RUNTIME",
                    "CORE",
                    False,
                    f"runtime readiness read failed: {type(exc).__name__}: {str(exc)[:240]}",
                ),
                _check(
                    "PERSISTED_RISK",
                    "CORE",
                    False,
                    "risk readiness unavailable because persisted read failed",
                ),
                _check(
                    "RECOVERY_ADMISSION",
                    "CORE",
                    False,
                    "recovery admission unavailable because persisted read failed",
                ),
            ))
        else:
            checks.append(_check(
                "PERSISTED_RUNTIME",
                "CORE",
                persisted.runtime is not None,
                (
                    f"runtime checkpoint version={persisted.runtime.version}"
                    if persisted.runtime is not None
                    else "no persisted Phase70 runtime checkpoint"
                ),
            ))
            checks.append(_check(
                "PERSISTED_RISK",
                "CORE",
                persisted.risk is not None,
                (
                    f"risk ledger version={persisted.risk.version} "
                    f"state={persisted.risk.current_state}"
                    if persisted.risk is not None
                    else "no persisted Phase73 operational-risk ledger"
                ),
            ))
            admission_ok = (
                persisted.admission.status == "OPEN"
                and not persisted.admission.blocked
            )
            checks.append(_check(
                "RECOVERY_ADMISSION",
                "CORE",
                admission_ok,
                (
                    "Phase86 admission OPEN"
                    if admission_ok
                    else (
                        "Phase86 recovery barrier active: "
                        f"{persisted.admission.reason or persisted.admission.status}"
                    )
                ),
            ))

        market: GroundedMarketPrefetch | None = None
        try:
            market = self.market_reader.load(
                asset_ids=assets,
                decision_timestamp=observed_at,
            )
            if not isinstance(market, GroundedMarketPrefetch):
                raise CryptoShadowReadinessError(
                    "market reader returned invalid prefetch type"
                )
            if set(market.asset_inputs) != set(assets):
                raise CryptoShadowReadinessError(
                    "market prefetch asset coverage mismatch"
                )
            checks.append(_check(
                "MARKET_PREFETCH",
                "CORE",
                True,
                "grounded sensors plus completed PIT market history loaded",
                assets,
            ))
        except Exception as exc:
            checks.append(_check(
                "MARKET_PREFETCH",
                "CORE",
                False,
                f"market prefetch failed: {type(exc).__name__}: {str(exc)[:260]}",
                assets,
            ))

        fresh_groups_by_asset: dict[str, tuple[str, ...]] = {}
        if market is None:
            checks.extend((
                _check(
                    "SENSOR_FRESHNESS",
                    "NEW_RISK",
                    False,
                    "sensor freshness unavailable because market prefetch failed",
                    assets,
                ),
                _check(
                    "COVARIANCE_HISTORY",
                    "NEW_RISK",
                    False,
                    "covariance history unavailable because market prefetch failed",
                    assets,
                ),
            ))
        else:
            stale_assets: list[str] = []
            for asset in assets:
                fresh_groups: set[str] = set()
                for row in market.asset_inputs[asset].observations:
                    if not row.available:
                        continue
                    budget = HORIZON_MAX_AGE_SECONDS.get(row.horizon)
                    if budget is None:
                        continue
                    age = observed_at - float(row.observed_at)
                    if -1e-9 <= age <= budget:
                        fresh_groups.add(row.independent_group)
                fresh_groups_by_asset[asset] = tuple(sorted(fresh_groups))
                if not fresh_groups:
                    stale_assets.append(asset)
            checks.append(_check(
                "SENSOR_FRESHNESS",
                "NEW_RISK",
                not stale_assets,
                (
                    "each asset has at least one available horizon-fresh sensor group"
                    if not stale_assets
                    else "no horizon-fresh available sensor group for "
                    + ",".join(stale_assets)
                ),
                stale_assets or assets,
            ))

            short_assets = [
                asset
                for asset in assets
                if len(market.return_series_by_asset[asset].values)
                < config.covariance.min_observations
            ]
            checks.append(_check(
                "COVARIANCE_HISTORY",
                "NEW_RISK",
                not short_assets,
                (
                    f"each asset has >= {config.covariance.min_observations} aligned PIT returns"
                    if not short_assets
                    else (
                        "insufficient aligned PIT returns for "
                        + ",".join(short_assets)
                    )
                ),
                short_assets or assets,
            ))

        contexts: dict[str, AssetLaggedEdgeContext] | None = None
        if market is not None:
            groups_by_asset = {
                asset: (
                    fresh_groups_by_asset.get(asset)
                    or tuple(sorted({
                        row.independent_group
                        for row in market.asset_inputs[asset].observations
                        if row.available
                    }))
                )
                for asset in assets
            }
            try:
                contexts = self.edge_reader.load_contexts(
                    groups_by_asset=groups_by_asset,
                    decision_timestamp=observed_at,
                    cost_asset_id_by_asset=market.cost_asset_id_by_asset,
                    minimum_net_margin_bps=minimum_net_margin_bps,
                )
                if set(contexts) != set(assets):
                    raise CryptoShadowReadinessError(
                        "edge context asset coverage mismatch"
                    )
            except Exception as exc:
                checks.extend((
                    _check(
                        "RELIABILITY_FRESHNESS",
                        "NEW_RISK",
                        False,
                        f"edge context read failed: {type(exc).__name__}: {str(exc)[:240]}",
                        assets,
                    ),
                    _check(
                        "RELIABILITY_MATURITY",
                        "NEW_RISK",
                        False,
                        "reliability maturity unavailable because edge context read failed",
                        assets,
                    ),
                    _check(
                        "DYNAMIC_COST",
                        "NEW_RISK",
                        False,
                        "dynamic cost unavailable because edge context read failed",
                        assets,
                    ),
                ))
        else:
            checks.extend((
                _check(
                    "RELIABILITY_FRESHNESS",
                    "NEW_RISK",
                    False,
                    "reliability unavailable because market prefetch failed",
                    assets,
                ),
                _check(
                    "RELIABILITY_MATURITY",
                    "NEW_RISK",
                    False,
                    "reliability unavailable because market prefetch failed",
                    assets,
                ),
                _check(
                    "DYNAMIC_COST",
                    "NEW_RISK",
                    False,
                    "dynamic cost unavailable because market prefetch failed",
                    assets,
                ),
            ))

        if contexts is not None:
            stale_reliability_assets: list[str] = []
            immature_assets: list[str] = []
            missing_cost_assets: list[str] = []
            for asset in assets:
                context = contexts[asset]
                rows = tuple(context.reliability)
                latest_generated = max(
                    (row.snapshot_generated_at for row in rows),
                    default=None,
                )
                latest_window = max(
                    (row.snapshot_window_end for row in rows),
                    default=None,
                )
                if (
                    latest_generated is None
                    or latest_window is None
                    or observed_at - latest_generated
                    > self.reliability_max_age_seconds
                    or observed_at - latest_window
                    > self.reliability_max_age_seconds
                ):
                    stale_reliability_assets.append(asset)

                mature = {
                    row.group
                    for row in rows
                    if row.sample_count >= MIN_MATURE_SAMPLES
                }
                if len(mature) < MIN_MATURE_GROUPS:
                    immature_assets.append(asset)

                if (
                    context.round_trip_cost_bps is None
                    or context.cost_observed_at is None
                ):
                    missing_cost_assets.append(asset)

            checks.append(_check(
                "RELIABILITY_FRESHNESS",
                "NEW_RISK",
                not stale_reliability_assets,
                (
                    f"latest reliability window/generated_at <= "
                    f"{self.reliability_max_age_seconds:.0f}s old"
                    if not stale_reliability_assets
                    else "stale or missing reliability snapshot for "
                    + ",".join(stale_reliability_assets)
                ),
                stale_reliability_assets or assets,
            ))
            checks.append(_check(
                "RELIABILITY_MATURITY",
                "NEW_RISK",
                not immature_assets,
                (
                    f"each asset has >= {MIN_MATURE_GROUPS} groups with "
                    f">= {MIN_MATURE_SAMPLES} samples"
                    if not immature_assets
                    else "insufficient mature reliability groups for "
                    + ",".join(immature_assets)
                ),
                immature_assets or assets,
            ))
            checks.append(_check(
                "DYNAMIC_COST",
                "NEW_RISK",
                not missing_cost_assets,
                (
                    "fresh fillable decision-time round-trip cost exists for every asset"
                    if not missing_cost_assets
                    else "fresh fillable dynamic cost unavailable for "
                    + ",".join(missing_cost_assets)
                ),
                missing_cost_assets or assets,
            ))

        try:
            symbols = tuple(_crypto_symbol(asset) for asset in assets)
            execution = self.execution_provider.collect(symbols)
            if not isinstance(
                execution,
                BinanceSpotRecoveryEvidenceBundle,
            ):
                raise CryptoShadowReadinessError(
                    "execution provider returned invalid evidence type"
                )
            if {row.asset_id for row in execution.assets} != set(symbols):
                raise CryptoShadowReadinessError(
                    "execution evidence asset coverage mismatch"
                )
            checks.append(_check(
                "PUBLIC_EXECUTION_EVIDENCE",
                "CORE",
                True,
                "fresh public Binance depth/exchange-info evidence available",
                assets,
            ))
        except Exception as exc:
            checks.append(_check(
                "PUBLIC_EXECUTION_EVIDENCE",
                "CORE",
                False,
                f"public execution evidence failed: {type(exc).__name__}: {str(exc)[:240]}",
                assets,
            ))

        safe = all(
            row.status == "PASS"
            for row in checks
            if row.scope == "CORE"
        )
        new_risk_ready = safe and all(
            row.status == "PASS"
            for row in checks
            if row.scope == "NEW_RISK"
        )
        status = (
            "READY_FOR_EDGE_BOUND_SHADOW"
            if new_risk_ready
            else "SAFE_FAIL_CLOSED_ONLY"
            if safe
            else "NOT_READY"
        )
        return CryptoShadowReadinessReport(
            runtime_id=runtime_id,
            observed_at=observed_at,
            status=status,
            safe_to_invoke_shadow_worker=safe,
            new_risk_ready=new_risk_ready,
            checks=tuple(checks),
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
                rpc = getattr(
                    getattr(self.persisted_probe, "runtime_store", None),
                    "_rpc",
                    None,
                )
                close = getattr(rpc, "close", None)
                if callable(close):
                    close()
        finally:
            self._closed = True
        return True

    def __enter__(self) -> "CryptoShadowReadinessGate":
        if self._closed:
            raise CryptoShadowReadinessError("Phase115 gate is closed")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
