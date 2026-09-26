from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from .phase91_supabase_rpc_transport import (
    SupabaseRecoveryRpcConfigurationError,
    _is_secure_project_url,
    _read_scoped_project_url,
    _read_scoped_secret_key_from_env,
    _sanitize_error_payload,
    _validate_server_key,
)
from .phase105_lagged_prospective_edge import LaggedReliabilityEvidence
from .phase106_decision_bound_lagged_edge import AssetLaggedEdgeContext

PHASE108_SCHEMA_VERSION = "brian.phase108-supabase-lagged-edge-reader.v1"
READINESS_COST_COMPILER_VERSION = "brian.readiness-cost-sampler.v1"

_ALLOWED_TABLES = frozenset({
    "brian_sensor_reliability_shadow_snapshots",
    "brian_dynamic_cost_quotes",
})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_READ_MAX_ATTEMPTS = 3
_READ_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_READ_RETRY_BACKOFF_SECONDS = 0.25


class SupabaseLaggedEdgeReaderError(RuntimeError):
    pass


class SupabaseLaggedEdgeReaderResponseError(SupabaseLaggedEdgeReaderError):
    pass


def _iso_utc(timestamp: float) -> str:
    value = float(timestamp)
    if not math.isfinite(value):
        raise ValueError("timestamp must be finite")
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _timestamp(value: object, label: str) -> float:
    text = str(value or "").strip()
    if not text:
        raise SupabaseLaggedEdgeReaderResponseError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SupabaseLaggedEdgeReaderResponseError(
            f"{label} is not ISO-8601"
        ) from exc
    if parsed.tzinfo is None:
        raise SupabaseLaggedEdgeReaderResponseError(
            f"{label} must include timezone"
        )
    result = parsed.timestamp()
    if not math.isfinite(result):
        raise SupabaseLaggedEdgeReaderResponseError(
            f"{label} is not finite"
        )
    return result


def _number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SupabaseLaggedEdgeReaderResponseError(
            f"{label} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise SupabaseLaggedEdgeReaderResponseError(
            f"{label} must be finite"
        )
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise SupabaseLaggedEdgeReaderResponseError(
            f"{label} must be integer"
        )
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SupabaseLaggedEdgeReaderResponseError(
            f"{label} must be integer"
        ) from exc
    return result


def _identifier(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(text):
        raise ValueError(f"{label} has invalid identifier characters")
    return text


@dataclass(frozen=True, slots=True)
class SupabaseLaggedEdgeReaderConfig:
    project_url: str
    key_source: str
    cost_project_url: str | None = None
    cost_key_source: str | None = None
    timeout_seconds: float = 10.0
    cost_max_age_seconds: float = 300.0
    outcome_horizon_seconds: int = 900
    schema_version: str = PHASE108_SCHEMA_VERSION
    read_only: bool = True
    shadow_only: bool = True
    live_execution: bool = False

    def __post_init__(self) -> None:
        if not _is_secure_project_url(self.project_url):
            raise ValueError(
                "project_url must use https (http allowed only for localhost)"
            )
        if not self.key_source.strip():
            raise ValueError("key_source is required")
        if self.cost_project_url is not None:
            if not _is_secure_project_url(self.cost_project_url):
                raise ValueError(
                    "cost_project_url must use https (http allowed only for localhost)"
                )
            if not str(self.cost_key_source or "").strip():
                raise ValueError(
                    "cost_key_source is required with cost_project_url"
                )
        elif self.cost_key_source is not None:
            raise ValueError(
                "cost_key_source requires cost_project_url"
            )
        if (
            not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        if (
            not math.isfinite(self.cost_max_age_seconds)
            or self.cost_max_age_seconds <= 0
        ):
            raise ValueError("cost_max_age_seconds must be positive")
        if self.outcome_horizon_seconds not in (300, 900, 3600):
            raise ValueError(
                "outcome_horizon_seconds must be 300, 900 or 3600"
            )
        if not self.read_only or not self.shadow_only or self.live_execution:
            raise ValueError("Phase108 must remain read-only shadow-only")


class SupabaseLaggedEdgeReader:
    """Read only PIT reliability snapshots + decision-time cost evidence.

    The selection mirrors the proven ALPHA edge challenger:
    first choose one latest reliability window that existed by decision time,
    then load only rows from that exact window. Cost evidence is the latest
    fillable dynamic quote at/before decision time and is rejected when stale.
    No insert/update/delete/RPC/order surface exists in this transport.
    """

    def __init__(
        self,
        *,
        config: SupabaseLaggedEdgeReaderConfig,
        api_key: str,
        cost_api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise SupabaseRecoveryRpcConfigurationError(
                "Supabase API key is required"
            )
        _validate_server_key(api_key, config.key_source)
        resolved_cost_key = api_key if cost_api_key is None else cost_api_key
        resolved_cost_source = (
            config.key_source
            if config.cost_key_source is None
            else config.cost_key_source
        )
        _validate_server_key(resolved_cost_key, resolved_cost_source)
        self.config = config
        self._api_key = api_key
        self._cost_api_key = resolved_cost_key
        self._cost_project_url = (
            config.project_url
            if config.cost_project_url is None
            else config.cost_project_url
        )
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=config.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> "SupabaseLaggedEdgeReader":
        source = os.environ if env is None else env
        project_url, _project_url_source = _read_scoped_project_url(
            source,
            "BRIAN_EDGE",
        )
        key, key_source = _read_scoped_secret_key_from_env(
            source,
            "BRIAN_EDGE",
        )
        _validate_server_key(key, key_source)

        cost_scope_names = (
            "BRIAN_COST_SUPABASE_URL",
            "BRIAN_COST_SUPABASE_SECRET_KEY",
            "BRIAN_COST_SUPABASE_SECRET_KEYS",
            "BRIAN_COST_SUPABASE_SERVICE_ROLE_KEY",
        )
        cost_scope_requested = any(
            str(source.get(name, "")).strip()
            for name in cost_scope_names
        )
        if cost_scope_requested:
            cost_project_url = str(
                source.get("BRIAN_COST_SUPABASE_URL", "")
            ).strip().rstrip("/")
            if not cost_project_url:
                raise SupabaseRecoveryRpcConfigurationError(
                    "BRIAN_COST_SUPABASE_URL is required when cost scope is configured"
                )
            scoped_key_present = any(
                str(source.get(name, "")).strip()
                for name in (
                    "BRIAN_COST_SUPABASE_SECRET_KEY",
                    "BRIAN_COST_SUPABASE_SECRET_KEYS",
                    "BRIAN_COST_SUPABASE_SERVICE_ROLE_KEY",
                )
            )
            if not scoped_key_present:
                raise SupabaseRecoveryRpcConfigurationError(
                    "scoped BRIAN_COST Supabase server key is required"
                )
            cost_key, cost_key_source = _read_scoped_secret_key_from_env(
                source,
                "BRIAN_COST",
            )
            _validate_server_key(cost_key, cost_key_source)
        else:
            cost_project_url = None
            cost_key = key
            cost_key_source = None

        def _float(name: str, default: float) -> float:
            raw = source.get(name)
            if raw is None or not str(raw).strip():
                return default
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise SupabaseRecoveryRpcConfigurationError(
                    f"{name} must be numeric"
                ) from exc
            if not math.isfinite(value) or value <= 0:
                raise SupabaseRecoveryRpcConfigurationError(
                    f"{name} must be positive"
                )
            return value

        raw_horizon = source.get(
            "BRIAN_EDGE_OUTCOME_HORIZON_SECONDS",
            "900",
        )
        try:
            horizon = int(raw_horizon)
        except (TypeError, ValueError) as exc:
            raise SupabaseRecoveryRpcConfigurationError(
                "BRIAN_EDGE_OUTCOME_HORIZON_SECONDS must be integer"
            ) from exc

        return cls(
            config=SupabaseLaggedEdgeReaderConfig(
                project_url=project_url,
                key_source=key_source,
                cost_project_url=cost_project_url,
                cost_key_source=cost_key_source,
                timeout_seconds=_float(
                    "BRIAN_EDGE_READER_TIMEOUT_SECONDS",
                    10.0,
                ),
                cost_max_age_seconds=_float(
                    "BRIAN_EDGE_COST_MAX_AGE_SECONDS",
                    300.0,
                ),
                outcome_horizon_seconds=horizon,
            ),
            api_key=key,
            cost_api_key=cost_key,
            client=client,
        )

    def _get(
        self,
        table: str,
        *,
        params: Mapping[str, str],
    ) -> list[Mapping[str, Any]]:
        if table not in _ALLOWED_TABLES:
            raise SupabaseLaggedEdgeReaderError(
                f"table is not allowed by Phase108: {table}"
            )
        use_cost_source = table == "brian_dynamic_cost_quotes"
        project_url = (
            self._cost_project_url
            if use_cost_source
            else self.config.project_url
        )
        api_key = self._cost_api_key if use_cost_source else self._api_key
        url = f"{project_url}/rest/v1/{table}"
        headers = {
            "apikey": api_key,
            "accept": "application/json",
            "user-agent": "brian-phase108-lagged-edge-reader/1",
        }
        response: httpx.Response | None = None
        last_transport_error: BaseException | None = None
        for attempt in range(1, _READ_MAX_ATTEMPTS + 1):
            try:
                response = self._client.get(
                    url,
                    headers=headers,
                    params=dict(params),
                )
                last_transport_error = None
            except httpx.TimeoutException as exc:
                last_transport_error = exc
                if attempt >= _READ_MAX_ATTEMPTS:
                    raise SupabaseLaggedEdgeReaderError(
                        f"Supabase read timeout for {table} "
                        f"after {_READ_MAX_ATTEMPTS} attempts"
                    ) from exc
            except httpx.TransportError as exc:
                last_transport_error = exc
                if attempt >= _READ_MAX_ATTEMPTS:
                    raise SupabaseLaggedEdgeReaderError(
                        f"Supabase read transport failure for {table} "
                        f"after {_READ_MAX_ATTEMPTS} attempts"
                    ) from exc
            else:
                if 200 <= response.status_code < 300:
                    break
                if (
                    response.status_code not in _READ_RETRYABLE_STATUS
                    or attempt >= _READ_MAX_ATTEMPTS
                ):
                    detail = _sanitize_error_payload(response)
                    raise SupabaseLaggedEdgeReaderResponseError(
                        f"Supabase read {table} returned HTTP "
                        f"{response.status_code}: {detail}"
                    )
            if attempt < _READ_MAX_ATTEMPTS:
                time.sleep(_READ_RETRY_BACKOFF_SECONDS * attempt)

        if response is None:
            raise SupabaseLaggedEdgeReaderError(
                f"Supabase read transport failure for {table}"
            ) from last_transport_error
        try:
            payload = response.json()
        except ValueError as exc:
            raise SupabaseLaggedEdgeReaderResponseError(
                f"Supabase read {table} returned invalid JSON"
            ) from exc
        if not isinstance(payload, list):
            raise SupabaseLaggedEdgeReaderResponseError(
                f"Supabase read {table} must return a JSON array"
            )
        rows: list[Mapping[str, Any]] = []
        for value in payload:
            if not isinstance(value, Mapping):
                raise SupabaseLaggedEdgeReaderResponseError(
                    f"Supabase read {table} returned non-object row"
                )
            rows.append(value)
        return rows

    def _latest_reliability_window(
        self,
        *,
        decision_timestamp: float,
    ) -> tuple[str, str] | None:
        decision_iso = _iso_utc(decision_timestamp)
        rows = self._get(
            "brian_sensor_reliability_shadow_snapshots",
            params={
                "select": "window_end,generated_at",
                "outcome_horizon_seconds": (
                    f"eq.{self.config.outcome_horizon_seconds}"
                ),
                "window_end": f"lte.{decision_iso}",
                "generated_at": f"lte.{decision_iso}",
                "order": "window_end.desc,generated_at.desc",
                "limit": "1",
            },
        )
        if not rows:
            return None
        row = rows[0]
        window_end = str(row.get("window_end") or "").strip()
        generated_at = str(row.get("generated_at") or "").strip()
        # Parse now so malformed/future values cannot become query selectors.
        if _timestamp(window_end, "window_end") > decision_timestamp:
            raise SupabaseLaggedEdgeReaderResponseError(
                "reliability window is after decision time"
            )
        if _timestamp(generated_at, "generated_at") > decision_timestamp:
            raise SupabaseLaggedEdgeReaderResponseError(
                "reliability generation is after decision time"
            )
        return window_end, generated_at

    def _load_reliability(
        self,
        *,
        groups: Sequence[str],
        decision_timestamp: float,
    ) -> tuple[LaggedReliabilityEvidence, ...]:
        clean_groups = tuple(sorted({
            _identifier(group, "independent_group")
            for group in groups
        }))
        if not clean_groups:
            return ()
        window = self._latest_reliability_window(
            decision_timestamp=decision_timestamp,
        )
        if window is None:
            return ()
        window_end, generated_at = window
        rows = self._get(
            "brian_sensor_reliability_shadow_snapshots",
            params={
                "select": (
                    "independent_group,sample_count,"
                    "bayesian_hit_rate_beta10_10,avg_signed_bps,"
                    "avg_cost_adjusted_signed_bps,outcome_horizon_seconds,"
                    "window_end,generated_at,evidence_class,"
                    "shadow_only,live_execution"
                ),
                "outcome_horizon_seconds": (
                    f"eq.{self.config.outcome_horizon_seconds}"
                ),
                "window_end": f"eq.{window_end}",
                "generated_at": f"eq.{generated_at}",
                "independent_group": (
                    "in.(" + ",".join(clean_groups) + ")"
                ),
                "order": "independent_group.asc",
                "limit": str(max(100, len(clean_groups) * 4)),
            },
        )
        result: list[LaggedReliabilityEvidence] = []
        seen_groups: set[str] = set()
        expected_window_end = _timestamp(window_end, "window_end")
        expected_generated_at = _timestamp(generated_at, "generated_at")
        for row in rows:
            if row.get("shadow_only") is not True:
                raise SupabaseLaggedEdgeReaderResponseError(
                    "reliability row is not shadow_only"
                )
            if row.get("live_execution") is not False:
                raise SupabaseLaggedEdgeReaderResponseError(
                    "reliability row crossed live boundary"
                )
            if str(row.get("evidence_class") or "") != "PROSPECTIVE_DEVELOPMENT_SHADOW":
                raise SupabaseLaggedEdgeReaderResponseError(
                    "reliability row has wrong evidence class"
                )
            row_window_end = _timestamp(row.get("window_end"), "window_end")
            row_generated_at = _timestamp(row.get("generated_at"), "generated_at")
            if (
                abs(row_window_end - expected_window_end) > 1e-6
                or abs(row_generated_at - expected_generated_at) > 1e-6
                or row_window_end > decision_timestamp
                or row_generated_at > decision_timestamp
            ):
                raise SupabaseLaggedEdgeReaderResponseError(
                    "reliability row escaped selected PIT window"
                )
            horizon = _integer(
                row.get("outcome_horizon_seconds"),
                "outcome_horizon_seconds",
            )
            if horizon != self.config.outcome_horizon_seconds:
                raise SupabaseLaggedEdgeReaderResponseError(
                    "reliability row has wrong outcome horizon"
                )
            group = _identifier(
                row.get("independent_group"),
                "independent_group",
            )
            if group in seen_groups:
                raise SupabaseLaggedEdgeReaderResponseError(
                    f"ambiguous reliability rows for independent group {group}"
                )
            seen_groups.add(group)
            hit = row.get("bayesian_hit_rate_beta10_10")
            avg = row.get("avg_signed_bps")
            after = row.get("avg_cost_adjusted_signed_bps")
            if hit is None or avg is None or after is None:
                # An immature/incomplete group contributes no fabricated prior.
                continue
            result.append(LaggedReliabilityEvidence(
                group=group,
                sample_count=_integer(
                    row.get("sample_count"),
                    "sample_count",
                ),
                bayesian_hit_rate=_number(
                    hit,
                    "bayesian_hit_rate_beta10_10",
                ),
                avg_signed_bps=_number(avg, "avg_signed_bps"),
                avg_cost_adjusted_signed_bps=_number(
                    after,
                    "avg_cost_adjusted_signed_bps",
                ),
                outcome_horizon_seconds=horizon,
                snapshot_window_end=row_window_end,
                snapshot_generated_at=row_generated_at,
                evidence_class=str(
                    row.get("evidence_class")
                    or "PROSPECTIVE_DEVELOPMENT_SHADOW"
                ),
                shadow_only=True,
                live_execution=False,
            ))
        return tuple(result)

    def _latest_costs(
        self,
        *,
        cost_asset_ids: Sequence[str],
        decision_timestamp: float,
    ) -> dict[str, tuple[float, float]]:
        assets = tuple(sorted({
            _identifier(asset, "cost_asset_id")
            for asset in cost_asset_ids
        }))
        if not assets:
            return {}
        decision_iso = _iso_utc(decision_timestamp)
        rows = self._get(
            "brian_dynamic_cost_quotes",
            params={
                "select": (
                    "asset_id,observed_at,estimated_round_trip_cost_bps,"
                    "fillable,quality,shadow_only,live_execution"
                ),
                "asset_id": "in.(" + ",".join(assets) + ")",
                "compiler_version": f"eq.{READINESS_COST_COMPILER_VERSION}",
                "observed_at": f"lte.{decision_iso}",
                "fillable": "eq.true",
                "quality": "neq.UNAVAILABLE",
                "order": "observed_at.desc",
                "limit": str(max(100, len(assets) * 20)),
            },
        )
        latest: dict[str, tuple[float, float]] = {}
        for row in rows:
            asset = _identifier(row.get("asset_id"), "asset_id")
            if asset in latest or asset not in assets:
                continue
            if row.get("shadow_only") is not True:
                raise SupabaseLaggedEdgeReaderResponseError(
                    "cost row is not shadow_only"
                )
            if row.get("live_execution") is not False:
                raise SupabaseLaggedEdgeReaderResponseError(
                    "cost row crossed live boundary"
                )
            if row.get("fillable") is not True:
                continue
            if str(row.get("quality") or "") == "UNAVAILABLE":
                raise SupabaseLaggedEdgeReaderResponseError(
                    "cost row quality is unavailable"
                )
            observed = _timestamp(row.get("observed_at"), "cost observed_at")
            if observed > decision_timestamp:
                raise SupabaseLaggedEdgeReaderResponseError(
                    "cost row is after decision time"
                )
            if decision_timestamp - observed > self.config.cost_max_age_seconds:
                continue
            cost = _number(
                row.get("estimated_round_trip_cost_bps"),
                "estimated_round_trip_cost_bps",
            )
            if cost < 0:
                raise SupabaseLaggedEdgeReaderResponseError(
                    "estimated_round_trip_cost_bps cannot be negative"
                )
            latest[asset] = (cost, observed)
        return latest

    def load_contexts(
        self,
        *,
        groups_by_asset: Mapping[str, Sequence[str]],
        decision_timestamp: float,
        cost_asset_id_by_asset: Mapping[str, str] | None = None,
        minimum_net_margin_bps: float = 2.0,
    ) -> dict[str, AssetLaggedEdgeContext]:
        timestamp = float(decision_timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("decision_timestamp must be finite")
        if (
            not math.isfinite(float(minimum_net_margin_bps))
            or minimum_net_margin_bps < 0
        ):
            raise ValueError(
                "minimum_net_margin_bps must be finite and non-negative"
            )

        normalized_groups: dict[str, tuple[str, ...]] = {}
        for raw_asset, groups in groups_by_asset.items():
            asset = _identifier(raw_asset, "asset_id")
            normalized_groups[asset] = tuple(sorted({
                _identifier(group, "independent_group")
                for group in groups
            }))
        union_groups = tuple(sorted({
            group
            for groups in normalized_groups.values()
            for group in groups
        }))
        reliability = self._load_reliability(
            groups=union_groups,
            decision_timestamp=timestamp,
        )
        reliability_by_group = {
            row.group: row
            for row in reliability
        }

        cost_ids: dict[str, str] = {}
        explicit = cost_asset_id_by_asset or {}
        for asset in normalized_groups:
            cost_ids[asset] = _identifier(
                explicit.get(asset, asset),
                "cost_asset_id",
            )
        latest_costs = self._latest_costs(
            cost_asset_ids=tuple(cost_ids.values()),
            decision_timestamp=timestamp,
        )

        result: dict[str, AssetLaggedEdgeContext] = {}
        for asset, groups in normalized_groups.items():
            cost = latest_costs.get(cost_ids[asset])
            result[asset] = AssetLaggedEdgeContext(
                reliability=tuple(
                    reliability_by_group[group]
                    for group in groups
                    if group in reliability_by_group
                ),
                round_trip_cost_bps=None if cost is None else cost[0],
                cost_observed_at=None if cost is None else cost[1],
                minimum_net_margin_bps=float(minimum_net_margin_bps),
            )
        return result

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SupabaseLaggedEdgeReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
