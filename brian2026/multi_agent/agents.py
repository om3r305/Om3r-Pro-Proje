"""The six roles. Four are deterministic code over point-in-time data; two call a model.

* market scanner  - deterministic filter over BRIAN's existing scanner output (no model call);
* news intelligence - Grok (+ X Search with a date cut), cited sources only;
* strategist      - Claude, first without seeing Grok, then once against Grok's cited answer;
* skeptic         - deterministic, evidence-backed objections (no "disagree for the sake of it");
* data verifier   - deterministic timestamp / source / freshness / future-data checks;
* risk manager    - deterministic, frozen limits; the models cannot change them.

Numbers that decide anything (edge, cost, samples) come from the measured evidence ledger, never from
model text. Model answers only contribute a stance, reasons and cited sources.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

from .providers.base import CallCounter, GuardedProvider, ProviderRequest, ProviderResponse
from .schemas import (
    ASSESSMENT_JSON_SCHEMA,
    ASSET_CLASSES,
    AgentAssessment,
    Candidate,
    MeasuredEdge,
    RiskVerdict,
    SkepticFinding,
    Source,
    Stance,
    Verification,
    parse_utc,
    utc_iso,
)


@dataclass(frozen=True, slots=True)
class ScannerRules:
    min_abs_move_bps: float = 100.0
    min_volume_ratio: float = 2.0
    max_candidate_age_seconds: int = 15 * 60


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Frozen: changing a limit is a code change with review, never a model decision."""
    min_net_edge_bps: float = 3.0
    min_samples: int = 100
    max_spread_bps: float = 25.0
    max_price_age_seconds: int = 15 * 60
    max_signal_age_seconds: int = 15 * 60
    max_opens_per_day: int = 12
    max_daily_loss_pct: float = 0.02
    max_position_pct: float = 0.10


RISK_LIMITS = RiskLimits()


# --- 1. market scanner -----------------------------------------------------------------------------
def scan(candidates: Iterable[Candidate], decision_time: datetime,
         rules: ScannerRules = ScannerRules()) -> tuple[list[Candidate], list[tuple[Candidate, str]]]:
    """Keep only meaningful, fresh moves; everything else is rejected with a reason (no model call)."""
    selected: list[Candidate] = []
    rejected: list[tuple[Candidate, str]] = []
    for c in candidates:
        observed = parse_utc(c.observed_at)
        if observed is None:
            rejected.append((c, "INVALID_TIMESTAMP"))
        elif observed > decision_time:
            rejected.append((c, "FUTURE_CANDIDATE"))
        elif (decision_time - observed).total_seconds() > rules.max_candidate_age_seconds:
            rejected.append((c, "STALE_CANDIDATE"))
        elif c.direction not in (1, -1):
            rejected.append((c, "NO_DIRECTION"))
        elif not ((c.move_bps is not None and abs(c.move_bps) >= rules.min_abs_move_bps)
                  or (c.volume_ratio is not None and c.volume_ratio >= rules.min_volume_ratio)):
            rejected.append((c, "MOVE_NOT_UNUSUAL"))
        else:
            selected.append(c)
    return selected, rejected


# --- prompts shared by the model roles --------------------------------------------------------------
SYSTEM_STRATEGIST = (
    "You are the strategist of a SHADOW-ONLY market research system; nothing you say places an order. "
    "Assess whether the observed move has an economic mechanism that makes continuation over the stated "
    "horizon more likely than not after costs. Use only the facts given and facts you can cite. Facts "
    "published after the decision time do not exist. Separate evidence from counter-evidence. If the facts "
    "are insufficient, answer 'uncertain'. Do not invent prices, edges or costs."
)
SYSTEM_INTELLIGENCE = (
    "You are the news-intelligence analyst of a SHADOW-ONLY research system. Find what caused the observed "
    "move: company or official announcements, filings, earnings, credible reports, unusual social attention. "
    "For every claim give the source URL, source type (OFFICIAL_PRIMARY, INDEPENDENT_PROFESSIONAL, SOCIAL, "
    "UNKNOWN) and publication time in ISO-8601 UTC. Ignore anything published after the decision time. Say "
    "when a claim is only a rumour or only on social media, whether the move started before the news, and "
    "whether the news concerns this exact company or instrument."
)


def _facts(candidate: Candidate, decision_time: datetime, edge: MeasuredEdge | None) -> dict[str, Any]:
    return {
        "decision_time_utc": utc_iso(decision_time),
        "asset": candidate.asset,
        "asset_class": candidate.asset_class,
        "observed_at": candidate.observed_at,
        "move_bps": candidate.move_bps,
        "direction": "up" if candidate.direction == 1 else "down",
        "volume_ratio": candidate.volume_ratio,
        "spread_bps": candidate.spread_bps,
        "horizon_minutes": 60,
        "measured_segment_evidence": None if edge is None else {
            "segment": edge.segment, "samples": edge.samples, "gross_bps": edge.gross_bps,
            "standard_error_bps": edge.standard_error_bps, "round_trip_cost_bps": edge.cost_bps,
        },
        "known_sources": [_source_dict(s) for s in candidate.sources],
    }


def _source_dict(source: Source) -> dict[str, Any]:
    return {"url": source.url, "source_type": source.source_type, "published_at": source.published_at,
            "claim": source.claim}


def _summary(assessment: AgentAssessment) -> dict[str, Any]:
    return {"stance": assessment.stance, "reason": assessment.reason, "evidence": list(assessment.evidence),
            "counter_evidence": list(assessment.counter_evidence),
            "sources": [_source_dict(s) for s in assessment.sources]}


def _assessment(agent: str, response: ProviderResponse, round_no: int, accessed_at: str) -> AgentAssessment:
    if response.status != "OK" or response.payload is None:
        return AgentAssessment(agent, response.provider, response.model, response.status,
                               reason="no model assessment", round=round_no, error_class=response.error_class)
    p = response.payload
    sources = tuple(
        Source.from_mapping({**s, "accessed_at": accessed_at}) for s in p.get("sources", []) if isinstance(s, Mapping)
    )[:20]
    return AgentAssessment(
        agent, response.provider, response.model, "MOCK" if response.mock else "OK",
        stance=str(p.get("stance")), reason=str(p.get("reason", ""))[:1_000],
        evidence=tuple(str(x)[:300] for x in p.get("evidence", []))[:10],
        counter_evidence=tuple(str(x)[:300] for x in p.get("counter_evidence", []))[:10],
        sources=sources, round=round_no,
    )


# --- 2./3. model roles -----------------------------------------------------------------------------
def run_model_role(agent: str, provider: GuardedProvider, system: str, facts: Mapping[str, Any],
                   decision_time: datetime, calls: CallCounter, allow_paid: bool, *, round_no: int = 1,
                   other: AgentAssessment | None = None, x_search: bool = False) -> tuple[AgentAssessment, ProviderResponse]:
    prompt = "Facts known at the decision time (JSON):\n" + json.dumps(facts, sort_keys=True, default=str)
    if other is not None:
        prompt += ("\n\nIndependent assessment from another analyst (treat as a claim to check, not as truth; "
                   "its sources may be wrong):\n" + json.dumps(_summary(other), sort_keys=True)
                   + "\n\nReturn your revised assessment. Change your stance only for evidence, not for agreement.")
    request = ProviderRequest(
        agent=agent, system=system, prompt=prompt, json_schema=ASSESSMENT_JSON_SCHEMA,
        x_search=x_search, x_search_to_date=(decision_time + timedelta(days=1)).strftime("%Y-%m-%d") if x_search else None,
    )
    response = provider.call(request, calls=calls, allow_paid=allow_paid)
    return _assessment(agent, response, round_no, utc_iso(decision_time)), response


# --- 5. data verifier --------------------------------------------------------------------------------
def verify(candidate: Candidate, decision_time: datetime, edge: MeasuredEdge | None,
           assessments: Sequence[AgentAssessment], limits: RiskLimits = RISK_LIMITS) -> Verification:
    checks: dict[str, str] = {}
    failures: list[str] = []
    gaps: list[str] = []

    observed = parse_utc(candidate.observed_at)
    price_at = parse_utc(candidate.price_observed_at)
    if candidate.asset_class not in ASSET_CLASSES:
        failures.append("UNKNOWN_ASSET_CLASS")
    if observed is None or observed > decision_time:
        failures.append("CANDIDATE_TIME_INVALID_OR_FUTURE")
    if candidate.reference_price is None or candidate.reference_price <= 0 or price_at is None:
        gaps.append("PRICE_MISSING")
    elif price_at > decision_time:
        failures.append("PRICE_FROM_THE_FUTURE")
    elif (decision_time - price_at).total_seconds() > limits.max_price_age_seconds:
        gaps.append("PRICE_STALE")
    checks["price"] = "OK" if not {"PRICE_MISSING", "PRICE_FROM_THE_FUTURE", "PRICE_STALE"} & set(failures + gaps) else "PROBLEM"

    if edge is None or edge.net_bps is None:
        gaps.append("NO_MEASURED_EDGE")
    else:
        computed = parse_utc(edge.computed_at)
        if computed is None or computed > decision_time:
            failures.append("EDGE_COMPUTED_AFTER_DECISION")
    checks["measured_edge"] = "OK" if edge is not None and edge.net_bps is not None else "MISSING"

    all_sources = list(candidate.sources) + [s for a in assessments for s in a.sources]
    seen: set[str] = set()
    future, undated, insecure, duplicates = 0, 0, 0, 0
    for s in all_sources:
        published = parse_utc(s.published_at)
        if not s.url.startswith("https://"):
            insecure += 1
        if published is None:
            undated += 1
        elif published > decision_time:
            future += 1
        if s.url in seen:
            duplicates += 1
        seen.add(s.url)
    if future:
        failures.append(f"SOURCES_PUBLISHED_AFTER_DECISION:{future}")
    checks["sources"] = (f"{len(seen)} unique, {duplicates} duplicate, {undated} undated, "
                         f"{insecure} non-https, {future} after decision")
    usable = [s for s in all_sources if parse_utc(s.published_at) and parse_utc(s.published_at) <= decision_time
              and s.url.startswith("https://")]
    if usable and all(s.source_type == "SOCIAL" for s in usable):
        gaps.append("SOCIAL_ONLY_SOURCES")

    if failures:
        return Verification("FAILED", checks, tuple(failures + gaps))
    if gaps:
        return Verification("INSUFFICIENT_DATA", checks, tuple(gaps))
    return Verification("VERIFIED", checks, ())


# --- 4. skeptic ------------------------------------------------------------------------------------
def skeptic(candidate: Candidate, decision_time: datetime, edge: MeasuredEdge | None,
            assessments: Sequence[AgentAssessment], limits: RiskLimits = RISK_LIMITS) -> list[SkepticFinding]:
    findings: list[SkepticFinding] = []
    observed = parse_utc(candidate.observed_at)
    if observed is not None and (decision_time - observed).total_seconds() > limits.max_signal_age_seconds:
        age = int((decision_time - observed).total_seconds())
        findings.append(SkepticFinding("LATE_ENTRY", "the move is older than the entry window", True, f"age={age}s"))
    if edge is not None and edge.net_bps is not None and edge.net_bps <= 0:
        findings.append(SkepticFinding(
            "COST_EXCEEDS_MEASURED_EDGE", "this kind of signal lost money after costs in measured outcomes", True,
            f"segment={edge.segment} n={edge.samples} net={edge.net_bps:.1f}bps cost={edge.cost_bps}bps"))
    if edge is not None and edge.samples < limits.min_samples:
        findings.append(SkepticFinding("THIN_SAMPLE", "too few measured outcomes to trust the edge", True,
                                       f"n={edge.samples} < {limits.min_samples}"))
    if candidate.spread_bps is not None and candidate.spread_bps > limits.max_spread_bps:
        findings.append(SkepticFinding("WIDE_SPREAD", "spread alone eats the expected move", True,
                                       f"spread={candidate.spread_bps}bps"))
    move_started = parse_utc(str(candidate.metadata.get("move_started_at") or "")) if candidate.metadata else None
    dated = [parse_utc(s.published_at) for a in assessments for s in a.sources]
    dated = [d for d in dated if d is not None and d <= decision_time]
    if move_started is not None and dated and min(dated) > move_started:
        findings.append(SkepticFinding("MOVE_PRECEDED_NEWS", "price moved before the earliest cited source", False,
                                       f"move_started={utc_iso(move_started)} first_source={utc_iso(min(dated))}"))
    stances = {a.stance for a in assessments if a.status in ("OK", "MOCK") and a.round == max(x.round for x in assessments)}
    if Stance.BULLISH.value in stances and Stance.BEARISH.value in stances:
        findings.append(SkepticFinding("ANALYSTS_DISAGREE", "the final assessments point in opposite directions",
                                       False, ",".join(sorted(stances))))
    return findings


# --- 6. risk manager -------------------------------------------------------------------------------
def risk_check(candidate: Candidate, edge: MeasuredEdge | None, *, opens_today: int | None = None,
               day_return_pct: float | None = None, limits: RiskLimits = RISK_LIMITS) -> RiskVerdict:
    reasons: list[str] = []
    gross = edge.shrunk_gross_bps if edge else None
    cost = edge.cost_bps if edge else None
    if cost is not None and candidate.spread_bps is not None:
        # Measured costs assume a normal spread; charge the excess of a wider live spread.
        cost = cost + max(0.0, candidate.spread_bps - cost / 2)
    net = None if gross is None or cost is None or edge is None or edge.standard_error_bps is None \
        else gross - edge.standard_error_bps - cost
    if edge is None or net is None:
        reasons.append("EXPECTED_EDGE_NOT_MEASURED")
    else:
        if edge.samples < limits.min_samples:
            reasons.append(f"SAMPLES_BELOW_{limits.min_samples}")
        if net < limits.min_net_edge_bps:
            reasons.append(f"NET_EDGE_BELOW_{limits.min_net_edge_bps:g}BPS")
    if candidate.spread_bps is None:
        reasons.append("SPREAD_UNKNOWN")
    elif candidate.spread_bps > limits.max_spread_bps:
        reasons.append("SPREAD_TOO_WIDE")
    if opens_today is None or opens_today >= limits.max_opens_per_day:
        reasons.append("DAILY_TRADE_LIMIT_OR_UNKNOWN")
    if day_return_pct is not None and day_return_pct <= -limits.max_daily_loss_pct:
        reasons.append("DAILY_LOSS_LIMIT")
    return RiskVerdict(
        status="BLOCKED" if reasons else "PASS", reasons=tuple(reasons),
        expected_gross_edge_bps=gross, estimated_total_cost_bps=cost, expected_net_edge_bps=net,
        limits={k: getattr(limits, k) for k in limits.__dataclass_fields__},
    )


facts_for = _facts
