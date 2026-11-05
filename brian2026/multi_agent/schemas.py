"""Structured records of the multi-agent layer.

Rules that hold for every record:
* numbers that were not measured stay ``None`` - nothing is filled in to look complete;
* model output never supplies the expected edge or the cost: those come from the measured evidence
  ledger and the cost model, and model text is kept separate from evidence and counter-evidence;
* ``execution_allowed`` is always ``False``: a decision is an assessment, never an order.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

SCHEMA_VERSION = "brian.multi-agent-decision.v1"


class Decision(str, Enum):
    BUY_CANDIDATE = "BUY_CANDIDATE"
    SELL_CANDIDATE = "SELL_CANDIDATE"
    WAIT = "WAIT"
    REJECT = "REJECT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NO_TRADE = "NO_TRADE"


class Stance(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"
    REJECT = "reject"


ASSET_CLASSES = frozenset({"equity", "crypto", "etf", "index", "commodity", "fx"})


def parse_utc(value: str | None) -> datetime | None:
    """ISO-8601 to aware UTC; naive or unparseable timestamps are rejected (None)."""
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True, slots=True)
class Source:
    """One piece of cited evidence. ``published_at`` is when it became public, not when we read it."""
    url: str
    source_type: str  # OFFICIAL_PRIMARY | INDEPENDENT_PROFESSIONAL | SOCIAL | MARKET_DATA | UNKNOWN
    published_at: str | None
    accessed_at: str | None
    claim: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Source":
        return cls(
            url=str(raw.get("url") or ""),
            source_type=str(raw.get("source_type") or "UNKNOWN").upper(),
            published_at=raw.get("published_at") if isinstance(raw.get("published_at"), str) else None,
            accessed_at=raw.get("accessed_at") if isinstance(raw.get("accessed_at"), str) else None,
            claim=str(raw.get("claim") or "")[:500],
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """A market-scanner candidate with the point-in-time facts known at ``observed_at``."""
    candidate_id: str
    asset: str
    asset_class: str
    observed_at: str
    reference_price: float | None
    price_observed_at: str | None
    move_bps: float | None
    direction: int  # +1 up move, -1 down move
    signal_lane: str = "MULTIASSET_EVENT_REACTION"
    news_class: str = "NONE"
    volume_ratio: float | None = None
    spread_bps: float | None = None
    sources: tuple[Source, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MeasuredEdge:
    """Measured after-cost evidence for the candidate's segment (from the treasury evidence ledger)."""
    segment: str
    samples: int
    gross_bps: float | None
    standard_error_bps: float | None
    cost_bps: float | None
    computed_at: str | None

    @property
    def shrunk_gross_bps(self) -> float | None:
        if self.gross_bps is None or self.samples <= 0:
            return None
        return self.gross_bps * self.samples / (self.samples + 100)

    @property
    def net_bps(self) -> float | None:
        shrunk = self.shrunk_gross_bps
        if shrunk is None or self.standard_error_bps is None or self.cost_bps is None:
            return None
        return shrunk - self.standard_error_bps - self.cost_bps


@dataclass(frozen=True, slots=True)
class AgentAssessment:
    agent: str
    provider: str | None
    model: str | None
    status: str  # OK | SKIPPED | FAILED | MOCK
    stance: str = Stance.UNCERTAIN.value
    reason: str = ""
    evidence: tuple[str, ...] = ()
    counter_evidence: tuple[str, ...] = ()
    sources: tuple[Source, ...] = ()
    round: int = 1
    error_class: str | None = None


@dataclass(frozen=True, slots=True)
class SkepticFinding:
    code: str
    detail: str
    blocking: bool
    evidence: str


@dataclass(frozen=True, slots=True)
class Verification:
    status: str  # VERIFIED | INSUFFICIENT_DATA | FAILED
    checks: Mapping[str, str]
    problems: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    status: str  # PASS | BLOCKED
    reasons: tuple[str, ...]
    expected_gross_edge_bps: float | None
    estimated_total_cost_bps: float | None
    expected_net_edge_bps: float | None
    limits: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class UsageRecord:
    provider: str
    model: str
    agent: str
    status: str
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float
    reported_cost_usd: float | None
    latency_ms: int | None
    cached: bool = False
    mock: bool = False
    error_class: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    candidate_id: str
    asset: str
    asset_class: str
    decision: str
    horizon_minutes: int
    decision_time_utc: str
    candidate_observed_at: str
    assessments: tuple[AgentAssessment, ...]
    skeptic_findings: tuple[SkepticFinding, ...]
    verification: Verification
    risk: RiskVerdict
    decision_reasons: tuple[str, ...]
    usage: tuple[UsageRecord, ...]
    provider_cost_usd: float
    debate_rounds: int
    mock: bool
    execution_allowed: bool = False
    shadow_only: bool = True
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.execution_allowed is not False or self.shadow_only is not True:
            raise ValueError("a multi-agent decision can never allow execution")
        if self.decision not in Decision._value2member_map_:
            raise ValueError(f"unknown decision {self.decision!r}")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["expected_gross_edge_bps"] = self.risk.expected_gross_edge_bps
        out["estimated_total_cost_bps"] = self.risk.estimated_total_cost_bps
        out["expected_net_edge_bps"] = self.risk.expected_net_edge_bps
        return out


# JSON schema every provider must answer with (Claude structured outputs / Grok response_format).
ASSESSMENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stance": {"type": "string", "enum": [s.value for s in Stance]},
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "counter_evidence": {"type": "array", "items": {"type": "string"}},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "source_type": {"type": "string"},
                    "published_at": {"type": "string"},
                    "claim": {"type": "string"},
                },
                "required": ["url", "source_type", "published_at", "claim"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["stance", "reason", "evidence", "counter_evidence", "sources"],
    "additionalProperties": False,
}


def validate_assessment_payload(payload: Any) -> list[str]:
    """Minimal structural validation of a provider answer; returns problems (empty = valid)."""
    problems: list[str] = []
    if not isinstance(payload, Mapping):
        return ["payload is not an object"]
    if payload.get("stance") not in Stance._value2member_map_:
        problems.append("stance missing or not allowed")
    if not isinstance(payload.get("reason"), str):
        problems.append("reason must be a string")
    for key in ("evidence", "counter_evidence"):
        value = payload.get(key)
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            problems.append(f"{key} must be a list of strings")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not all(isinstance(x, Mapping) for x in sources):
        problems.append("sources must be a list of objects")
    extra = set(payload) - set(ASSESSMENT_JSON_SCHEMA["properties"])
    if extra:
        problems.append(f"unexpected fields: {sorted(extra)}")
    return problems
