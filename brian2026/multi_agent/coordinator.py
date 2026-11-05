"""Runs one candidate through the six roles and returns an immutable decision record.

Order: operator STOP / duplicate check -> independent Claude and Grok (round 1) -> at most one cross-review
round -> data verifier -> skeptic -> independent risk manager -> decision. Every failure path ends in a
non-trading decision; model agreement is never treated as proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from . import agents
from .config import MultiAgentConfig
from .providers.base import CallCounter, GuardedProvider, ProviderResponse
from .schemas import (
    AgentAssessment,
    Candidate,
    Decision,
    DecisionRecord,
    MeasuredEdge,
    RiskVerdict,
    SkepticFinding,
    Stance,
    UsageRecord,
    Verification,
    utc_iso,
)


def decide(assessments: list[AgentAssessment], verification: Verification, findings: list[SkepticFinding],
           risk: RiskVerdict) -> tuple[Decision, list[str]]:
    """Pure decision rule. The order of checks is the order of precedence."""
    final_round = max((a.round for a in assessments), default=0)
    final = [a for a in assessments if a.round == final_round]
    usable = [a for a in final if a.status in ("OK", "MOCK")]
    if verification.status == "FAILED":
        return Decision.NO_TRADE, [f"verification failed: {', '.join(verification.problems)}"]
    if any(a.status == "FAILED" for a in final):
        return Decision.NO_TRADE, ["a provider failed; no automatic decision without both assessments"]
    if verification.status == "INSUFFICIENT_DATA":
        return Decision.INSUFFICIENT_DATA, [f"missing data: {', '.join(verification.problems)}"]
    if risk.status != "PASS":
        return Decision.NO_TRADE, [f"risk blocked: {', '.join(risk.reasons)}"]
    if len(usable) < 2:
        return Decision.INSUFFICIENT_DATA, ["both the strategist and the news assessment are required"]
    blocking = [f for f in findings if f.blocking]
    if blocking:
        return Decision.WAIT, [f"skeptic: {f.code} ({f.evidence})" for f in blocking]
    stances = {a.stance for a in usable}
    if Stance.REJECT.value in stances:
        return Decision.REJECT, ["an analyst rejected the candidate"]
    if len(stances) != 1:
        return Decision.WAIT, [f"analysts disagree: {sorted(stances)}"]
    stance = stances.pop()
    if stance == Stance.BULLISH.value:
        return Decision.BUY_CANDIDATE, ["both analysts bullish, data verified, measured edge clears cost"]
    if stance == Stance.BEARISH.value:
        return Decision.SELL_CANDIDATE, ["both analysts bearish, data verified, measured edge clears cost"]
    return Decision.WAIT, [f"analysts {stance}"]


@dataclass
class Coordinator:
    config: MultiAgentConfig
    strategist: GuardedProvider
    intelligence: GuardedProvider
    _seen: set[str] = field(default_factory=set)

    def run(self, candidate: Candidate, decision_time: datetime, edge: MeasuredEdge | None, *,
            opens_today: int | None = None, day_return_pct: float | None = None) -> DecisionRecord:
        if decision_time.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        risk = agents.risk_check(candidate, edge, opens_today=opens_today, day_return_pct=day_return_pct)
        if self.config.operator_stop:
            return self._record(candidate, decision_time, Decision.NO_TRADE, ["operator STOP is active"],
                                [], Verification("INSUFFICIENT_DATA", {}, ("OPERATOR_STOP",)), [], risk, [], 0)
        if candidate.candidate_id in self._seen:
            return self._record(candidate, decision_time, Decision.REJECT, ["duplicate candidate; no new calls"],
                                [], Verification("INSUFFICIENT_DATA", {}, ("DUPLICATE",)), [], risk, [], 0)
        self._seen.add(candidate.candidate_id)

        calls = CallCounter(self.config.max_provider_calls_per_candidate)
        allow_paid = self.config.paid_calls_allowed
        facts = agents.facts_for(candidate, decision_time, edge)
        responses: list[tuple[str, ProviderResponse]] = []
        assessments: list[AgentAssessment] = []

        rounds = 0
        # Cheapest path first: when the measured edge cannot clear cost, no model can change that, so no
        # model is called (no spend). Models only weigh in on candidates the deterministic gates allow.
        if risk.status == "PASS" and self.config.max_debate_rounds >= 1:
            rounds = 1
            claude_1, r = agents.run_model_role("strategist", self.strategist, agents.SYSTEM_STRATEGIST, facts,
                                                decision_time, calls, allow_paid)
            responses.append(("strategist", r))
            grok_1, r = agents.run_model_role("news_intelligence", self.intelligence, agents.SYSTEM_INTELLIGENCE,
                                              facts, decision_time, calls, allow_paid, x_search=True)
            responses.append(("news_intelligence", r))
            assessments += [claude_1, grok_1]
            both_ok = all(a.status in ("OK", "MOCK") for a in (claude_1, grok_1))
            if self.config.max_debate_rounds >= 2 and both_ok:
                rounds = 2
                claude_2, r = agents.run_model_role("strategist", self.strategist, agents.SYSTEM_STRATEGIST, facts,
                                                    decision_time, calls, allow_paid, round_no=2, other=grok_1)
                responses.append(("strategist", r))
                grok_2, r = agents.run_model_role("news_intelligence", self.intelligence, agents.SYSTEM_INTELLIGENCE,
                                                  facts, decision_time, calls, allow_paid, round_no=2, other=claude_1)
                responses.append(("news_intelligence", r))
                assessments += [claude_2, grok_2]

        verification = agents.verify(candidate, decision_time, edge, assessments)
        findings = agents.skeptic(candidate, decision_time, edge, assessments)
        decision, reasons = decide(assessments, verification, findings, risk)
        return self._record(candidate, decision_time, decision, reasons, assessments, verification, findings, risk,
                            responses, rounds)

    @staticmethod
    def _record(candidate: Candidate, decision_time: datetime, decision: Decision, reasons: Iterable[str],
                assessments: list[AgentAssessment], verification: Verification, findings: list[SkepticFinding],
                risk: RiskVerdict, responses: list[tuple[str, ProviderResponse]] | int, rounds: int) -> DecisionRecord:
        usage = tuple(
            UsageRecord(r.provider, r.model, agent, r.status, r.input_tokens, r.output_tokens, r.estimated_cost_usd,
                        r.reported_cost_usd, r.latency_ms, r.cached, r.mock, r.error_class)
            for agent, r in (responses if isinstance(responses, list) else [])
        )
        return DecisionRecord(
            candidate_id=candidate.candidate_id, asset=candidate.asset, asset_class=candidate.asset_class,
            decision=decision.value, horizon_minutes=60, decision_time_utc=utc_iso(decision_time),
            candidate_observed_at=candidate.observed_at, assessments=tuple(assessments),
            skeptic_findings=tuple(findings), verification=verification, risk=risk,
            decision_reasons=tuple(reasons), usage=usage,
            provider_cost_usd=round(sum(u.estimated_cost_usd for u in usage), 6), debate_rounds=rounds,
            mock=any(u.mock for u in usage),
        )
