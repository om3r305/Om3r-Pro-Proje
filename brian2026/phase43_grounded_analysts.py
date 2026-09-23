from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from statistics import fmean
from typing import Callable, Mapping, Sequence
import json
import math

from .expert_reasoner import ExpertDecision, reason_market
from .global_sensor_mesh import SensorObservation

PHASE43_SCHEMA_VERSION = "brian.phase43-grounded-analysts.v1"

HORIZON_MAX_AGE_SECONDS: Mapping[str, float] = {
    "MICRO_1_5M": 5 * 60.0,
    "FAST_5_30M": 30 * 60.0,
    "INTRADAY_30M_6H": 6 * 60 * 60.0,
    "SWING_6H_7D": 7 * 24 * 60 * 60.0,
    "MACRO_1D_PLUS": 7 * 24 * 60 * 60.0,
}


def _hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(text.encode("utf-8")).hexdigest()


def _finite(snapshot: Mapping[str, object], key: str) -> float | None:
    value = snapshot.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _state(snapshot: Mapping[str, object], key: str) -> int | None:
    value = _finite(snapshot, key)
    if value is None:
        return None
    if value > 0.5:
        return 1
    if value < -0.5:
        return -1
    return 0


@dataclass(frozen=True, slots=True)
class GroundedEvidenceBlock:
    evidence_id: str
    observation_id: str
    asset_id: str
    source_kind: str
    independent_group: str
    observed_at: float
    horizon: str
    direction: int
    strength: float
    confidence: float
    reliability: float
    available: bool
    fresh: bool
    directional_allowed: bool
    source_ids: tuple[str, ...]
    reason: str

    @property
    def quality(self) -> float:
        if not self.available or not self.fresh or not self.directional_allowed:
            return 0.0
        return self.strength * self.confidence * self.reliability

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GroundedEvidencePacket:
    asset_id: str
    decision_timestamp: float
    blocks: tuple[GroundedEvidenceBlock, ...]
    unavailable_source_kinds: tuple[str, ...]
    stale_evidence_ids: tuple[str, ...]
    schema_version: str = PHASE43_SCHEMA_VERSION
    external_tools_allowed_after_prefetch: bool = False
    shadow_only: bool = True
    live_execution: bool = False

    @property
    def usable_evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            block.evidence_id
            for block in self.blocks
            if block.available and block.fresh and block.directional_allowed
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "asset_id": self.asset_id,
            "decision_timestamp": self.decision_timestamp,
            "blocks": [row.to_dict() for row in self.blocks],
            "unavailable_source_kinds": self.unavailable_source_kinds,
            "stale_evidence_ids": self.stale_evidence_ids,
            "external_tools_allowed_after_prefetch": self.external_tools_allowed_after_prefetch,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
        }


def prefetch_structured_evidence(
    observations: Sequence[SensorObservation],
    *,
    decision_timestamp: float,
    source_kind_by_eye: Mapping[str, str] | None = None,
) -> GroundedEvidencePacket:
    """Freeze all analyst inputs before reasoning.

    This mirrors TradingAgents' corrected sentiment workflow: data is collected
    first and injected as structured blocks. Once the packet exists, analysts do
    not get an open-ended external-tool surface that could invite fabricated
    evidence. Future observations fail closed instead of being silently trimmed.
    """
    if not math.isfinite(float(decision_timestamp)):
        raise ValueError("decision_timestamp must be finite")
    rows = tuple(observations)
    if not rows:
        raise ValueError("grounded analyst prefetch requires observations")
    assets = {row.asset_id for row in rows}
    if len(assets) != 1:
        raise ValueError("one grounded packet may contain exactly one asset")
    source_map = source_kind_by_eye or {}

    blocks: list[GroundedEvidenceBlock] = []
    unavailable: set[str] = set()
    stale: list[str] = []
    for row in rows:
        if row.observed_at > decision_timestamp:
            raise ValueError(
                f"future evidence forbidden: {row.observation_id} observed_at={row.observed_at} "
                f"> decision_timestamp={decision_timestamp}"
            )
        source_kind = str(source_map.get(row.eye_id, row.independent_group)).strip()
        if not source_kind:
            raise ValueError("source_kind must be explicit")
        max_age = HORIZON_MAX_AGE_SECONDS.get(row.horizon)
        if max_age is None:
            raise ValueError(f"unsupported evidence horizon: {row.horizon}")
        fresh = (decision_timestamp - row.observed_at) <= max_age
        directional_allowed = row.independent_group != "news_gdelt" and source_kind != "news_gdelt"
        evidence_id = _hash({
            "observation_id": row.observation_id,
            "source_kind": source_kind,
            "decision_timestamp": float(decision_timestamp),
            "schema_version": PHASE43_SCHEMA_VERSION,
        })
        block = GroundedEvidenceBlock(
            evidence_id=evidence_id,
            observation_id=row.observation_id,
            asset_id=row.asset_id,
            source_kind=source_kind,
            independent_group=row.independent_group,
            observed_at=float(row.observed_at),
            horizon=row.horizon,
            direction=row.direction,
            strength=float(row.strength),
            confidence=float(row.confidence),
            reliability=float(row.reliability),
            available=bool(row.available),
            fresh=fresh,
            directional_allowed=directional_allowed,
            source_ids=tuple(row.source_ids),
            reason=row.reason,
        )
        blocks.append(block)
        if not row.available:
            unavailable.add(source_kind)
        if not fresh:
            stale.append(evidence_id)

    return GroundedEvidencePacket(
        asset_id=next(iter(assets)),
        decision_timestamp=float(decision_timestamp),
        blocks=tuple(sorted(blocks, key=lambda item: (item.source_kind, item.independent_group, item.evidence_id))),
        unavailable_source_kinds=tuple(sorted(unavailable)),
        stale_evidence_ids=tuple(sorted(stale)),
    )


@dataclass(frozen=True, slots=True)
class GroundedAnalystClaim:
    analyst: str
    direction: int
    confidence: float
    summary: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.analyst.strip() or not self.summary.strip():
            raise ValueError("analyst and summary are required")
        if self.direction not in (-1, 0, 1):
            raise ValueError("claim direction must be -1, 0 or 1")
        if not math.isfinite(float(self.confidence)) or not 0 <= float(self.confidence) <= 1:
            raise ValueError("claim confidence must be in [0,1]")
        if self.direction != 0 and not self.evidence_ids:
            raise ValueError("directional claims require evidence ids")


@dataclass(frozen=True, slots=True)
class ValidatedAnalystClaim:
    analyst: str
    direction: int
    requested_confidence: float
    grounded_confidence: float
    summary: str
    evidence_ids: tuple[str, ...]
    support_evidence_ids: tuple[str, ...]
    conflict_evidence_ids: tuple[str, ...]
    independent_support_groups: tuple[str, ...]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_grounded_claim(
    packet: GroundedEvidencePacket,
    claim: GroundedAnalystClaim,
) -> ValidatedAnalystClaim:
    by_id = {row.evidence_id: row for row in packet.blocks}
    unknown = tuple(value for value in claim.evidence_ids if value not in by_id)
    if unknown:
        raise ValueError(f"claim references unknown evidence ids: {unknown}")

    referenced = tuple(by_id[value] for value in claim.evidence_ids)
    unusable = tuple(
        row.evidence_id
        for row in referenced
        if not row.available or not row.fresh or not row.directional_allowed
    )
    if unusable and claim.direction != 0:
        raise ValueError(f"directional claim references unavailable/stale/non-directional evidence: {unusable}")

    if claim.direction == 0:
        return ValidatedAnalystClaim(
            analyst=claim.analyst,
            direction=0,
            requested_confidence=claim.confidence,
            grounded_confidence=0.0 if not referenced else min(claim.confidence, fmean(row.quality for row in referenced)),
            summary=claim.summary,
            evidence_ids=claim.evidence_ids,
            support_evidence_ids=(),
            conflict_evidence_ids=(),
            independent_support_groups=(),
            status="GROUNDED_NEUTRAL",
        )

    # Multiple observations from the same independent group count once, matching
    # Brian ALPHA's existing independence semantics.
    by_group: dict[str, GroundedEvidenceBlock] = {}
    for row in referenced:
        prior = by_group.get(row.independent_group)
        if prior is None or row.quality > prior.quality or (
            row.quality == prior.quality and row.observed_at > prior.observed_at
        ):
            by_group[row.independent_group] = row

    selected = tuple(by_group.values())
    support = tuple(row for row in selected if row.direction == claim.direction)
    conflicts = tuple(row for row in selected if row.direction == -claim.direction)
    if not support:
        raise ValueError("directional claim has no supporting grounded evidence")

    support_quality = fmean(row.quality for row in support)
    support_ratio = len(support) / max(1, len(selected))
    grounded_confidence = min(float(claim.confidence), support_quality * (0.5 + 0.5 * support_ratio))
    return ValidatedAnalystClaim(
        analyst=claim.analyst,
        direction=claim.direction,
        requested_confidence=float(claim.confidence),
        grounded_confidence=max(0.0, min(1.0, grounded_confidence)),
        summary=claim.summary,
        evidence_ids=tuple(claim.evidence_ids),
        support_evidence_ids=tuple(sorted(row.evidence_id for row in support)),
        conflict_evidence_ids=tuple(sorted(row.evidence_id for row in conflicts)),
        independent_support_groups=tuple(sorted(row.independent_group for row in support)),
        status="GROUNDED_DIRECTIONAL",
    )


def compile_prefetched_analyst_claims(packet: GroundedEvidencePacket) -> tuple[ValidatedAnalystClaim, ...]:
    """Create typed source-level analyst outputs from the frozen evidence packet.

    A later LLM narrative layer may summarize these rows, but it is not allowed
    to invent new evidence ids or modify the deterministic direction/confidence.
    """
    grouped: dict[str, list[GroundedEvidenceBlock]] = {}
    for row in packet.blocks:
        grouped.setdefault(row.source_kind, []).append(row)

    claims: list[ValidatedAnalystClaim] = []
    for source_kind, rows in sorted(grouped.items()):
        usable = [
            row for row in rows
            if row.available and row.fresh and row.directional_allowed and row.direction != 0
        ]
        if not usable:
            continue
        by_group: dict[str, GroundedEvidenceBlock] = {}
        for row in usable:
            prior = by_group.get(row.independent_group)
            if prior is None or row.quality > prior.quality:
                by_group[row.independent_group] = row
        selected = tuple(by_group.values())
        signed = [row.direction * row.quality for row in selected]
        aggregate = sum(signed) / len(signed)
        direction = 1 if aggregate > 0 else -1 if aggregate < 0 else 0
        if direction == 0:
            continue
        evidence_ids = tuple(sorted(row.evidence_id for row in selected))
        raw = GroundedAnalystClaim(
            analyst=f"{source_kind}_analyst",
            direction=direction,
            confidence=min(1.0, abs(aggregate)),
            summary=(
                f"{source_kind}: deterministic pre-fetched evidence aggregate; "
                f"{len(selected)} independent group(s), no external tool access after prefetch"
            ),
            evidence_ids=evidence_ids,
        )
        claims.append(validate_grounded_claim(packet, raw))
    return tuple(claims)


@dataclass(frozen=True, slots=True)
class RegimeAwareRoute:
    regime: str
    selected_experts: tuple[str, ...]
    selected_features: tuple[str, ...]
    rationale: str
    schema_version: str = PHASE43_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.selected_experts:
            raise ValueError("route requires experts")
        if len(self.selected_features) > 8:
            raise ValueError("route may select at most 8 complementary features")
        if len(set(self.selected_features)) != len(self.selected_features):
            raise ValueError("route features must be non-redundant")


def route_specialists(snapshot: Mapping[str, object]) -> RegimeAwareRoute:
    """Select complementary expert/feature families for the current regime.

    This is Brian's deterministic adaptation of TradingAgents' 'up to eight
    complementary indicators' behavior. Selection changes what is consulted;
    it never changes feature values or manufactures missing inputs.
    """
    s5 = _state(snapshot, "structure_state")
    s15 = _state(snapshot, "structure_15m")
    s1h = _state(snapshot, "structure_1h")
    expansion = _finite(snapshot, "range_expansion")

    if expansion is not None and expansion >= 3.0:
        return RegimeAwareRoute(
            "VOLATILITY_SHOCK",
            ("structure_expert", "trend_expert", "momentum_expert", "volume_expert"),
            (
                "range_expansion", "structure_state", "structure_15m", "structure_1h",
                "relative_volume", "volume_zscore", "acceleration", "return_1",
            ),
            "Extreme range expansion prioritizes structure, velocity and volume; mean reversion is withheld during shock.",
        )
    if s5 == s15 == s1h == 1:
        return RegimeAwareRoute(
            "ALIGNED_UPTREND",
            ("structure_expert", "trend_expert", "momentum_expert", "volume_expert"),
            (
                "structure_state", "structure_15m", "structure_1h", "ema_slope",
                "relative_volume", "acceleration", "rsi", "support_distance_atr",
            ),
            "Aligned trend emphasizes complementary trend, momentum and participation evidence.",
        )
    if s5 == s15 == s1h == -1:
        return RegimeAwareRoute(
            "ALIGNED_DOWNTREND",
            ("structure_expert", "trend_expert", "momentum_expert", "volume_expert"),
            (
                "structure_state", "structure_15m", "structure_1h", "ema_slope",
                "relative_volume", "acceleration", "rsi", "resistance_distance_atr",
            ),
            "Aligned downtrend emphasizes complementary trend, momentum and participation evidence.",
        )
    if s15 is not None and s1h is not None and s15 * s1h == -1:
        return RegimeAwareRoute(
            "HTF_CONFLICT",
            ("structure_expert", "trend_expert", "mean_reversion_expert", "volume_expert"),
            (
                "structure_state", "structure_15m", "structure_1h", "range_expansion",
                "zscore", "bb_position", "support_distance_atr", "resistance_distance_atr",
            ),
            "Higher-timeframe conflict reduces momentum chasing and adds location/mean-reversion context.",
        )
    if s15 == 0 and s1h == 0:
        return RegimeAwareRoute(
            "RANGE",
            ("structure_expert", "mean_reversion_expert", "momentum_expert", "volume_expert"),
            (
                "zscore", "bb_position", "support_distance_atr", "resistance_distance_atr",
                "rsi", "bullish_rsi_divergence", "bearish_rsi_divergence", "relative_volume",
            ),
            "Range regime prioritizes location, rejection, divergence and participation instead of redundant trend indicators.",
        )
    return RegimeAwareRoute(
        "MIXED_TRANSITION",
        (
            "structure_expert", "trend_expert", "momentum_expert",
            "volume_expert", "mean_reversion_expert",
        ),
        (
            "structure_state", "structure_15m", "structure_1h", "ema_slope",
            "rsi", "relative_volume", "zscore", "bb_position",
        ),
        "Transition regime keeps broad but capped coverage until structure becomes decisive.",
    )


@dataclass(frozen=True, slots=True)
class Phase43GroundedResult:
    packet: GroundedEvidencePacket
    route: RegimeAwareRoute
    analyst_claims: tuple[ValidatedAnalystClaim, ...]
    expert_decision: ExpertDecision
    analyst_direction: int
    analyst_confidence: float
    analyst_support_evidence_ids: tuple[str, ...]
    schema_version: str = PHASE43_SCHEMA_VERSION
    shadow_only: bool = True
    live_execution: bool = False
    automatic_promotion: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "packet": self.packet.to_dict(),
            "route": asdict(self.route),
            "analyst_claims": [row.to_dict() for row in self.analyst_claims],
            "expert_decision": self.expert_decision.manifest(),
            "analyst_direction": self.analyst_direction,
            "analyst_confidence": self.analyst_confidence,
            "analyst_support_evidence_ids": self.analyst_support_evidence_ids,
            "shadow_only": self.shadow_only,
            "live_execution": self.live_execution,
            "automatic_promotion": self.automatic_promotion,
        }


GroundedReasoner = Callable[..., ExpertDecision]


def run_grounded_phase43(
    snapshot: Mapping[str, object],
    observations: Sequence[SensorObservation],
    *,
    timestamp: float,
    source_kind_by_eye: Mapping[str, str] | None = None,
    reasoner: GroundedReasoner = reason_market,
) -> Phase43GroundedResult:
    packet = prefetch_structured_evidence(
        observations,
        decision_timestamp=timestamp,
        source_kind_by_eye=source_kind_by_eye,
    )
    route = route_specialists(snapshot)
    claims = compile_prefetched_analyst_claims(packet)
    if not callable(reasoner):
        raise TypeError("reasoner must be callable")
    decision = reasoner(
        snapshot,
        timestamp=timestamp,
        selected_experts=route.selected_experts,
    )
    if not decision.shadow_only:
        raise ValueError("grounded expert decision must remain shadow-only")
    if not math.isclose(float(decision.timestamp), float(timestamp), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("grounded expert decision timestamp drift")

    directional = tuple(row for row in claims if row.direction != 0 and row.grounded_confidence > 0)
    if directional:
        signed = sum(row.direction * row.grounded_confidence for row in directional)
        denominator = sum(row.grounded_confidence for row in directional)
        score = signed / denominator if denominator > 0 else 0.0
        analyst_direction = 1 if score > 0 else -1 if score < 0 else 0
        analyst_confidence = min(1.0, abs(score) * min(1.0, denominator / max(1, len(directional))))
    else:
        analyst_direction = 0
        analyst_confidence = 0.0

    support_ids = tuple(sorted({
        evidence_id
        for row in directional
        if row.direction == analyst_direction
        for evidence_id in row.support_evidence_ids
    }))
    return Phase43GroundedResult(
        packet=packet,
        route=route,
        analyst_claims=claims,
        expert_decision=decision,
        analyst_direction=analyst_direction,
        analyst_confidence=analyst_confidence,
        analyst_support_evidence_ids=support_ids,
    )
