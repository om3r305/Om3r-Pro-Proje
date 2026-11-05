"""Compare decision groups (A existing BRIAN, B Claude-led, C Claude+Grok) on the same candidates.

An outcome counts only if it was learned after the decision (``known_at`` > decision time) - an earlier
``known_at`` means the outcome could have leaked into the decision and is reported, not used.
Missing outcomes stay missing; they are never counted as wins or losses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .schemas import Decision, finite_or_none, parse_utc

ACTION = {Decision.BUY_CANDIDATE.value: 1, Decision.SELL_CANDIDATE.value: -1}


@dataclass(frozen=True, slots=True)
class GroupReport:
    group: str
    decisions: int
    actionable: int
    no_trade: int
    outcomes_used: int
    outcomes_missing: int
    outcomes_leaky: int
    true_positives: int
    false_positives: int
    missed_opportunities: int
    gross_bps_total: float
    net_bps_total: float
    net_bps_mean: float | None
    max_drawdown_bps: float
    provider_cost_usd: float


def compare_groups(groups: Mapping[str, Iterable[Mapping[str, Any]]],
                   outcomes: Mapping[str, Mapping[str, Any]]) -> list[GroupReport]:
    """``outcomes[candidate_id]`` = {direction, gross_bps (in the candidate's move direction), cost_bps, known_at}."""
    reports = []
    for name, decisions in groups.items():
        rows = sorted(decisions, key=lambda d: str(d.get("decision_time_utc")))
        actionable = no_trade = used = missing = leaky = tp = fp = missed = 0
        gross_total = net_total = equity = peak = drawdown = cost_usd = 0.0
        for d in rows:
            cost_usd += finite_or_none(d.get("provider_cost_usd")) or 0.0
            side = ACTION.get(str(d.get("decision")))
            if side is None:
                no_trade += 1
            else:
                actionable += 1
            o = outcomes.get(str(d.get("candidate_id")))
            decided = parse_utc(str(d.get("decision_time_utc")))
            known = parse_utc(str(o.get("known_at"))) if o else None
            gross = finite_or_none(o.get("gross_bps")) if o else None
            cost = finite_or_none(o.get("cost_bps")) if o else None
            move_dir = o.get("direction") if o else None
            if o is None or gross is None or cost is None or decided is None or known is None or move_dir not in (1, -1):
                missing += 1
                continue
            if known <= decided:
                leaky += 1
                continue
            used += 1
            if side is None:
                if gross - cost > 0:
                    missed += 1  # following the move would have paid after costs
                continue
            signed = gross if side == move_dir else -gross
            net = signed - cost
            gross_total += signed
            net_total += net
            tp += net > 0
            fp += net <= 0
            equity += net
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        trades = tp + fp
        reports.append(GroupReport(
            name, len(rows), actionable, no_trade, used, missing, leaky, tp, fp, missed,
            round(gross_total, 4), round(net_total, 4), round(net_total / trades, 4) if trades else None,
            round(drawdown, 4), round(cost_usd, 6),
        ))
    return reports
