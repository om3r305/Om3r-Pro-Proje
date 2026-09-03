from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Literal, Mapping, Sequence
import json
import math


SourceKind = Literal[
    "official", "exchange", "news", "social", "onchain", "market", "derivatives"
]
TrustClass = Literal[
    "verified_official", "verified_provider", "inferred_provider", "user_generated", "unknown"
]
EventKind = Literal[
    "listing", "delisting", "regulatory", "hack", "partnership", "token_unlock",
    "social_surge", "whale_transfer", "smart_money_flow", "liquidity_shift",
    "volume_anomaly", "funding_dislocation", "new_token", "other"
]
WhaleFlow = Literal[
    "exchange_deposit", "exchange_withdrawal", "dex_buy", "dex_sell", "internal_transfer", "unknown"
]

TRUST_WEIGHT: Mapping[TrustClass, float] = {
    "verified_official": 1.00,
    "verified_provider": 0.90,
    "inferred_provider": 0.65,
    "user_generated": 0.35,
    "unknown": 0.20,
}

SOURCE_WEIGHT: Mapping[SourceKind, float] = {
    "official": 1.00,
    "exchange": 0.95,
    "onchain": 0.90,
    "market": 0.85,
    "derivatives": 0.82,
    "news": 0.75,
    "social": 0.55,
}


def _finite(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _bounded(value: float, name: str, low: float = 0.0, high: float = 1.0) -> float:
    out = _finite(value, name)
    if not low <= out <= high:
        raise ValueError(f"{name} must be in [{low},{high}]")
    return out


def _canonical_hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                             allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class IntelEvent:
    asset: str
    event_kind: EventKind
    source_kind: SourceKind
    source_id: str
    published_at: float
    observed_at: float
    claim: str
    direction: float
    magnitude: float
    trust_class: TrustClass
    entity_confidence: float = 1.0
    content_fingerprint: str = ""
    corroboration_key: str = ""
    provenance_uri: str | None = None
    historical_timestamp_verified: bool = False
    schema_version: str = "brian.intel-event.v1"

    def __post_init__(self) -> None:
        asset = self.asset.strip().upper()
        if not asset or not self.source_id.strip() or not self.claim.strip():
            raise ValueError("asset, source_id and claim are required")
        published = _finite(self.published_at, "published_at")
        observed = _finite(self.observed_at, "observed_at")
        if observed < published:
            raise ValueError("an event cannot be observed before it was published")
        _bounded(self.direction, "direction", -1.0, 1.0)
        _bounded(self.magnitude, "magnitude")
        _bounded(self.entity_confidence, "entity_confidence")
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "published_at", published)
        object.__setattr__(self, "observed_at", observed)
        if not self.content_fingerprint:
            object.__setattr__(self, "content_fingerprint", _canonical_hash({
                "claim": " ".join(self.claim.lower().split()),
                "asset": asset,
                "kind": self.event_kind,
            }))
        if not self.corroboration_key:
            object.__setattr__(self, "corroboration_key", f"{asset}:{self.event_kind}")

    @property
    def event_id(self) -> str:
        return _canonical_hash({
            "schema_version": self.schema_version,
            "asset": self.asset,
            "event_kind": self.event_kind,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "published_at": self.published_at,
            "observed_at": self.observed_at,
            "claim": self.claim,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "trust_class": self.trust_class,
            "entity_confidence": self.entity_confidence,
            "content_fingerprint": self.content_fingerprint,
            "corroboration_key": self.corroboration_key,
            "provenance_uri": self.provenance_uri,
            "historical_timestamp_verified": self.historical_timestamp_verified,
        })

    def manifest(self) -> dict:
        return {**asdict(self), "event_id": self.event_id}


@dataclass(frozen=True, slots=True)
class SocialBurst:
    asset: str
    observed_at: float
    mention_count: int
    unique_authors: int
    unique_text_ratio: float
    median_account_age_days: float | None
    verified_or_official_fraction: float
    bot_likelihood: float
    cross_platform_count: int
    schema_version: str = "brian.social-burst.v1"

    def __post_init__(self) -> None:
        if self.mention_count < 0 or self.unique_authors < 0 or self.cross_platform_count < 0:
            raise ValueError("social counts cannot be negative")
        if self.unique_authors > self.mention_count:
            raise ValueError("unique authors cannot exceed mentions")
        _bounded(self.unique_text_ratio, "unique_text_ratio")
        _bounded(self.verified_or_official_fraction, "verified_or_official_fraction")
        _bounded(self.bot_likelihood, "bot_likelihood")
        if self.median_account_age_days is not None and self.median_account_age_days < 0:
            raise ValueError("account age cannot be negative")


def social_authenticity(burst: SocialBurst) -> float:
    if burst.mention_count <= 0:
        return 0.0
    author_diversity = min(1.0, burst.unique_authors / max(1, burst.mention_count) * 4.0)
    text_diversity = burst.unique_text_ratio
    official = burst.verified_or_official_fraction
    cross_platform = min(1.0, burst.cross_platform_count / 3.0)
    age = 0.5
    if burst.median_account_age_days is not None:
        age = min(1.0, burst.median_account_age_days / 365.0)
    raw = (
        0.24 * author_diversity + 0.24 * text_diversity + 0.18 * official +
        0.14 * cross_platform + 0.20 * age
    )
    return max(0.0, min(1.0, raw * (1.0 - 0.80 * burst.bot_likelihood)))


@dataclass(frozen=True, slots=True)
class WhaleObservation:
    asset: str
    observed_at: float
    entity_id: str
    label_source: str
    label_trust: TrustClass
    label_confidence: float
    flow: WhaleFlow
    usd_value: float
    counterparty_entity: str | None = None
    tx_hash: str | None = None
    historical_timestamp_verified: bool = False
    schema_version: str = "brian.whale-observation.v1"

    def __post_init__(self) -> None:
        if not self.asset.strip() or not self.entity_id.strip() or not self.label_source.strip():
            raise ValueError("asset/entity/label source are required")
        _bounded(self.label_confidence, "label_confidence")
        if _finite(self.usd_value, "usd_value") < 0:
            raise ValueError("usd_value cannot be negative")


@dataclass(frozen=True, slots=True)
class WhaleAssessment:
    economic_direction: float
    confidence: float
    materiality: float
    suspicious_or_unresolved: bool
    reasons: tuple[str, ...]


def assess_whale(observation: WhaleObservation, *, material_usd: float = 100_000.0) -> WhaleAssessment:
    if material_usd <= 0:
        raise ValueError("material_usd must be positive")
    trust = TRUST_WEIGHT[observation.label_trust]
    confidence = observation.label_confidence * trust
    materiality = min(1.0, observation.usd_value / material_usd)
    direction_map: Mapping[WhaleFlow, float] = {
        "exchange_deposit": -0.65,
        "exchange_withdrawal": 0.55,
        "dex_buy": 1.0,
        "dex_sell": -1.0,
        "internal_transfer": 0.0,
        "unknown": 0.0,
    }
    direction = direction_map[observation.flow]
    reasons: list[str] = []
    unresolved = False
    if observation.flow == "internal_transfer":
        reasons.append("internal transfer has no assumed buy/sell meaning")
    if observation.flow == "unknown":
        unresolved = True
        reasons.append("flow semantics unresolved")
    if observation.label_trust in {"user_generated", "unknown"}:
        unresolved = True
        reasons.append("wallet/entity attribution is not provider-verified")
    if confidence < 0.55:
        unresolved = True
        reasons.append("entity confidence below whale-action threshold")
    return WhaleAssessment(direction * confidence * materiality, confidence, materiality,
                           unresolved, tuple(reasons))


@dataclass(frozen=True, slots=True)
class TruthAssessment:
    truth_score: float
    manipulation_risk: float
    independent_sources: int
    source_kinds: int
    official_confirmation: bool
    duplicate_ratio: float
    reasons: tuple[str, ...]


def assess_truth(events: Sequence[IntelEvent]) -> TruthAssessment:
    if not events:
        return TruthAssessment(0.0, 1.0, 0, 0, False, 0.0, ("no evidence",))
    asset = events[0].asset
    key = events[0].corroboration_key
    if any(row.asset != asset or row.corroboration_key != key for row in events):
        raise ValueError("truth assessment requires one asset/corroboration claim cluster")

    source_ids = {row.source_id for row in events}
    source_kinds = {row.source_kind for row in events}
    fingerprints = {row.content_fingerprint for row in events}
    duplicate_ratio = 1.0 - len(fingerprints) / len(events)
    official = any(
        row.source_kind in {"official", "exchange"} and row.trust_class == "verified_official"
        for row in events
    )
    weighted = [
        TRUST_WEIGHT[row.trust_class] * SOURCE_WEIGHT[row.source_kind] * row.entity_confidence
        for row in events
    ]
    mean_quality = sum(weighted) / len(weighted)
    source_bonus = min(0.22, 0.07 * max(0, len(source_ids) - 1))
    diversity_bonus = min(0.15, 0.05 * max(0, len(source_kinds) - 1))
    official_bonus = 0.18 if official else 0.0
    duplicate_penalty = 0.35 * duplicate_ratio
    social_only_penalty = 0.20 if source_kinds == {"social"} else 0.0
    truth = max(0.0, min(1.0, mean_quality + source_bonus + diversity_bonus + official_bonus
                         - duplicate_penalty - social_only_penalty))
    low_trust_fraction = sum(row.trust_class in {"user_generated", "unknown"} for row in events) / len(events)
    manipulation = max(0.0, min(1.0,
        0.55 * duplicate_ratio + 0.30 * low_trust_fraction +
        (0.25 if source_kinds == {"social"} else 0.0) - (0.20 if official else 0.0)
    ))
    reasons: list[str] = []
    if official:
        reasons.append("official/exchange confirmation present")
    if len(source_ids) >= 2:
        reasons.append("independent corroboration present")
    if duplicate_ratio > 0.50:
        reasons.append("high duplicate-content ratio")
    if source_kinds == {"social"}:
        reasons.append("social-only claim cluster")
    if low_trust_fraction > 0:
        reasons.append("cluster contains low-trust attribution")
    return TruthAssessment(truth, manipulation, len(source_ids), len(source_kinds), official,
                           duplicate_ratio, tuple(reasons))


@dataclass(frozen=True, slots=True)
class AssetIntelligence:
    asset: str
    observed_at: float
    truth_score: float
    manipulation_risk: float
    event_direction: float
    event_strength: float
    social_authenticity: float
    whale_direction: float
    market_confirmation: float
    opportunity_priority: float
    veto_reasons: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    shadow_only: bool = True
    schema_version: str = "brian.asset-intelligence.v1"


def fuse_asset_intelligence(
    events: Sequence[IntelEvent],
    *,
    social: SocialBurst | None = None,
    whales: Sequence[WhaleObservation] = (),
    market_confirmation: float = 0.0,
) -> AssetIntelligence:
    if not events:
        raise ValueError("asset intelligence requires at least one event")
    asset = events[0].asset
    if any(row.asset != asset for row in events):
        raise ValueError("all events must refer to one asset")
    market = _bounded(market_confirmation, "market_confirmation", -1.0, 1.0)
    grouped: dict[str, list[IntelEvent]] = {}
    for row in events:
        grouped.setdefault(row.corroboration_key, []).append(row)
    truths = [assess_truth(group) for group in grouped.values()]
    truth_score = sum(x.truth_score for x in truths) / len(truths)
    manipulation = max(x.manipulation_risk for x in truths)

    weights = [TRUST_WEIGHT[e.trust_class] * SOURCE_WEIGHT[e.source_kind] * e.entity_confidence for e in events]
    total_weight = sum(weights)
    direction = sum(e.direction * e.magnitude * w for e, w in zip(events, weights)) / total_weight if total_weight else 0.0
    strength = sum(e.magnitude * w for e, w in zip(events, weights)) / total_weight if total_weight else 0.0

    social_score = social_authenticity(social) if social is not None else 0.5
    whale_rows = [assess_whale(row) for row in whales]
    whale_direction = sum(x.economic_direction for x in whale_rows) / len(whale_rows) if whale_rows else 0.0
    unresolved_whales = sum(x.suspicious_or_unresolved for x in whale_rows)

    veto: list[str] = []
    if truth_score < 0.35:
        veto.append("claim truth score too low")
    if manipulation >= 0.70:
        veto.append("manipulation risk too high")
    if social is not None and social_score < 0.25 and all(e.source_kind == "social" for e in events):
        veto.append("social burst appears inorganic")
    if whales and unresolved_whales == len(whale_rows) and all(e.event_kind == "whale_transfer" for e in events):
        veto.append("whale attribution/flow unresolved")

    directional_alignment = max(0.0, min(1.0, 0.5 + 0.5 * direction * market)) if market else 0.5
    whale_alignment = max(0.0, min(1.0, 0.5 + 0.5 * direction * whale_direction)) if whale_rows else 0.5
    priority = strength * truth_score * (1.0 - manipulation) * (
        0.30 + 0.20 * social_score + 0.25 * directional_alignment + 0.25 * whale_alignment
    )
    if veto:
        priority *= 0.10
    return AssetIntelligence(
        asset=asset,
        observed_at=max(row.observed_at for row in events),
        truth_score=truth_score,
        manipulation_risk=manipulation,
        event_direction=max(-1.0, min(1.0, direction)),
        event_strength=max(0.0, min(1.0, strength)),
        social_authenticity=social_score,
        whale_direction=max(-1.0, min(1.0, whale_direction)),
        market_confirmation=market,
        opportunity_priority=max(0.0, min(1.0, priority)),
        veto_reasons=tuple(veto),
        evidence_event_ids=tuple(sorted(row.event_id for row in events)),
    )


def rank_opportunities(rows: Sequence[AssetIntelligence], *, top_n: int = 20) -> tuple[AssetIntelligence, ...]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    return tuple(sorted(rows, key=lambda row: (-row.opportunity_priority, row.asset))[:top_n])


def assert_historical_replay_safe(events: Sequence[IntelEvent], whales: Sequence[WhaleObservation] = ()) -> None:
    """Reject hindsight backfills whose timestamp/label was not known at the historical time."""
    unsafe_events = [row.event_id for row in events if not row.historical_timestamp_verified]
    unsafe_whales = [row.tx_hash or row.entity_id for row in whales if not row.historical_timestamp_verified]
    if unsafe_events or unsafe_whales:
        raise ValueError(
            "historical intelligence replay requires point-in-time verified timestamps/labels; "
            "use prospective shadow capture instead of hindsight backfill"
        )
