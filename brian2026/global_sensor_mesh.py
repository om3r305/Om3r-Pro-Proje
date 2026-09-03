from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence
import hashlib
import json
import math

SensorFamily = Literal[
    "price_structure",
    "orderbook",
    "derivatives",
    "onchain",
    "news",
    "social_psychology",
    "cross_asset",
    "macro",
]
MarketDomain = Literal[
    "crypto",
    "fx",
    "commodity",
    "equity",
    "etf",
    "rates",
    "macro",
]
Horizon = Literal["MICRO_1_5M", "FAST_5_30M", "INTRADAY_30M_6H", "SWING_6H_7D", "MACRO_1D_PLUS"]
EvidenceClass = Literal["PROSPECTIVE_DEVELOPMENT_SHADOW"]

SENSOR_MESH_SCHEMA_VERSION = "brian.global-sensor-mesh.v1"
PROSPECTIVE_EVIDENCE_CLASS: EvidenceClass = "PROSPECTIVE_DEVELOPMENT_SHADOW"
DEFAULT_VIRTUAL_TICKETS_USD: tuple[float, ...] = (2.0, 3.0, 5.0, 10.0, 20.0)


def _hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SensorTemplate:
    template_id: str
    family: SensorFamily
    market_domain: MarketDomain
    horizon: Horizon
    independent_group: str
    signal_kind: str
    virtual_ticket_usd: float = 5.0
    shadow_only: bool = True
    live_execution: bool = False
    schema_version: str = SENSOR_MESH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.template_id.strip() or not self.independent_group.strip() or not self.signal_kind.strip():
            raise ValueError("sensor template identity must be explicit")
        if self.virtual_ticket_usd not in DEFAULT_VIRTUAL_TICKETS_USD:
            raise ValueError("virtual ticket must use a preregistered micro-book size")
        if not self.shadow_only or self.live_execution:
            raise ValueError("sensor templates are shadow-only and cannot execute live orders")


@dataclass(frozen=True, slots=True)
class LogicalEye:
    eye_id: str
    template_id: str
    asset_id: str
    family: SensorFamily
    market_domain: MarketDomain
    horizon: Horizon
    independent_group: str
    signal_kind: str
    virtual_ticket_usd: float
    shadow_only: bool = True
    live_execution: bool = False
    schema_version: str = SENSOR_MESH_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class SensorObservation:
    eye_id: str
    asset_id: str
    observed_at: float
    direction: int
    strength: float
    confidence: float
    reliability: float
    available: bool
    independent_group: str
    source_ids: tuple[str, ...]
    horizon: Horizon
    reason: str
    evidence_class: EvidenceClass = PROSPECTIVE_EVIDENCE_CLASS
    shadow_only: bool = True
    live_execution: bool = False
    schema_version: str = SENSOR_MESH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0 or 1")
        for name, value in (("strength", self.strength), ("confidence", self.confidence), ("reliability", self.reliability)):
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be finite in [0,1]")
        if not math.isfinite(float(self.observed_at)):
            raise ValueError("observed_at must be finite")
        if not self.available and self.direction != 0:
            raise ValueError("unavailable sensors cannot invent a directional signal")
        if self.available and not self.source_ids:
            raise ValueError("available sensor observations require provenance source ids")
        if not self.shadow_only or self.live_execution:
            raise ValueError("sensor observations are shadow-only")

    @property
    def observation_id(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True, slots=True)
class VirtualMicroBookReceipt:
    eye_id: str
    asset_id: str
    starting_equity: float
    ending_equity: float
    net_pnl: float
    max_drawdown_pct: float
    turnover_notional: float
    trading_cost: float
    active_decisions: int
    wins: int
    losses: int
    horizon: Horizon
    evidence_class: EvidenceClass = PROSPECTIVE_EVIDENCE_CLASS
    shadow_only: bool = True
    live_execution: bool = False
    schema_version: str = SENSOR_MESH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.starting_equity not in DEFAULT_VIRTUAL_TICKETS_USD:
            raise ValueError("micro-book starting equity must be preregistered")
        if min(self.ending_equity, self.turnover_notional, self.trading_cost) < 0:
            raise ValueError("micro-book accounting values cannot be negative")
        if not 0 <= self.max_drawdown_pct <= 100:
            raise ValueError("max drawdown must be in [0,100]")
        if min(self.active_decisions, self.wins, self.losses) < 0 or self.wins + self.losses > self.active_decisions:
            raise ValueError("invalid micro-book decision counts")
        if abs((self.ending_equity - self.starting_equity) - self.net_pnl) > 1e-8:
            raise ValueError("net_pnl must reconcile starting and ending equity")
        if not self.shadow_only or self.live_execution:
            raise ValueError("micro-books are virtual only")


def expand_logical_eyes(
    templates: Sequence[SensorTemplate],
    asset_ids: Sequence[str],
    *,
    market_domain: MarketDomain | None = None,
) -> tuple[LogicalEye, ...]:
    clean_assets = tuple(sorted({str(asset).strip() for asset in asset_ids if str(asset).strip()}))
    if not clean_assets:
        raise ValueError("logical eye expansion requires assets")
    eyes: list[LogicalEye] = []
    for template in sorted(templates, key=lambda row: row.template_id):
        if market_domain is not None and template.market_domain != market_domain:
            continue
        for asset_id in clean_assets:
            eye_id = _hash({"template_id": template.template_id, "asset_id": asset_id})
            eyes.append(LogicalEye(
                eye_id=eye_id,
                template_id=template.template_id,
                asset_id=asset_id,
                family=template.family,
                market_domain=template.market_domain,
                horizon=template.horizon,
                independent_group=template.independent_group,
                signal_kind=template.signal_kind,
                virtual_ticket_usd=template.virtual_ticket_usd,
            ))
    return tuple(eyes)


def default_global_sensor_templates() -> tuple[SensorTemplate, ...]:
    """Provider-neutral logical specialists.

    A template may exist before its provider is live. Missing evidence must emit
    available=False/direction=0 rather than a fabricated signal.
    """
    rows = (
        SensorTemplate("structure-fast", "price_structure", "crypto", "FAST_5_30M", "price_structure", "structure_break_retest", 5.0),
        SensorTemplate("momentum-fast", "price_structure", "crypto", "FAST_5_30M", "price_momentum", "relative_momentum", 5.0),
        SensorTemplate("mean-reversion-fast", "price_structure", "crypto", "FAST_5_30M", "price_mean_reversion", "exhaustion_reversion", 3.0),
        SensorTemplate("orderbook-fast", "orderbook", "crypto", "FAST_5_30M", "orderbook", "spread_imbalance_liquidity", 5.0),
        SensorTemplate("derivatives-fast", "derivatives", "crypto", "FAST_5_30M", "derivatives", "funding_oi_liquidation", 5.0),
        SensorTemplate("onchain-fast", "onchain", "crypto", "FAST_5_30M", "onchain", "economic_whale_flow", 5.0),
        SensorTemplate("news-fast", "news", "crypto", "FAST_5_30M", "news", "verified_event_impulse", 5.0),
        SensorTemplate("social-fast", "social_psychology", "crypto", "FAST_5_30M", "social", "panic_greed_propagation", 3.0),
        SensorTemplate("cross-asset-fast", "cross_asset", "crypto", "FAST_5_30M", "cross_asset", "risk_on_off_confirmation", 5.0),
        SensorTemplate("macro-context", "macro", "crypto", "INTRADAY_30M_6H", "macro", "rates_fx_liquidity_context", 2.0),
    )
    return rows


def mesh_manifest(templates: Sequence[SensorTemplate], assets_by_domain: Mapping[str, Sequence[str]]) -> dict[str, object]:
    logical_eye_count = 0
    for domain, assets in assets_by_domain.items():
        logical_eye_count += sum(1 for template in templates if template.market_domain == domain) * len(set(assets))
    return {
        "schema_version": SENSOR_MESH_SCHEMA_VERSION,
        "templates": [asdict(row) for row in templates],
        "asset_counts": {key: len(set(values)) for key, values in sorted(assets_by_domain.items())},
        "logical_eye_count": logical_eye_count,
        "virtual_micro_books": True,
        "live_execution": False,
        "shadow_only": True,
        "missing_evidence_is_unavailable": True,
    }
