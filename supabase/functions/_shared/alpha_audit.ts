export type AlphaAuditDecision = {
  observedAt: string;
  action: string;
  direction: number;
  referencePrice: string | number | null;
  estimatedRoundTripCostBps: string | number | null;
};

export type AlphaAuditPricePoint = {
  observed_at: string;
  observed_mid_price: string | number;
  estimated_round_trip_cost_bps: string | number | null;
};

function finite(v: unknown, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export function resolveAlphaAuditHorizon(
  decision: AlphaAuditDecision,
  horizonSeconds: number,
  points: AlphaAuditPricePoint[],
) {
  const reference = finite(decision.referencePrice, NaN);
  if (!(reference > 0)) return null;

  const observedMs = Date.parse(decision.observedAt);
  if (!Number.isFinite(observedMs)) return null;
  const targetMs = observedMs + horizonSeconds * 1000;
  const eligible = points
    .filter((p) => {
      const t = Date.parse(p.observed_at);
      return Number.isFinite(t) && t >= observedMs && t <= targetMs + 120_000;
    })
    .sort((a, b) => Date.parse(a.observed_at) - Date.parse(b.observed_at));
  if (!eligible.length) return null;

  const afterTarget = eligible.filter((p) => Date.parse(p.observed_at) >= targetMs);
  const resolvedPoint = afterTarget[0] ?? eligible.at(-1)!;
  if (Math.abs(Date.parse(resolvedPoint.observed_at) - targetMs) > 120_000) return null;
  const resolved = finite(resolvedPoint.observed_mid_price);
  if (!(resolved > 0)) return null;

  const prices = eligible.map((p) => finite(p.observed_mid_price)).filter((p) => p > 0);
  if (!prices.length) return null;
  const maxPx = Math.max(...prices), minPx = Math.min(...prices);
  const gross = resolved / reference - 1;
  const mfe = maxPx / reference - 1;
  const mae = minPx / reference - 1;
  const directionAdjusted = decision.direction === 0 ? 0 : decision.direction * gross;
  const fallbackCost = eligible.map((p) => finite(p.estimated_round_trip_cost_bps, NaN)).find((x) => Number.isFinite(x) && x >= 0);
  const decisionCost = finite(decision.estimatedRoundTripCostBps, NaN);
  const costBps = Number.isFinite(decisionCost) && decisionCost >= 0 ? decisionCost : (fallbackCost ?? 0);
  const longOpportunity = mfe * 10_000 > costBps;
  const shortOpportunity = -mae * 10_000 > costBps;

  let classification: string;
  let explanation: string;
  if (decision.action === "WAIT" || decision.action === "VETO") {
    if (longOpportunity && shortOpportunity) classification = "WAIT_VOLATILE_TWO_SIDED";
    else if (longOpportunity) classification = "WAIT_MISSED_LONG";
    else if (shortOpportunity) classification = "WAIT_MISSED_SHORT";
    else classification = "WAIT_JUSTIFIED_BY_COST";
    explanation = `ex-post excursion from the immutable decision-time reference compared with contemporaneous round-trip cost ${costBps.toFixed(4)} bps; this receipt audits the decision and does not retune thresholds`;
  } else {
    const netDirBps = directionAdjusted * 10_000 - costBps;
    classification = netDirBps > 0 ? "ACTION_FAVORABLE_AFTER_COST" : "ACTION_UNFAVORABLE_AFTER_COST";
    explanation = `direction-adjusted terminal return from immutable decision-time reference minus contemporaneous round-trip cost = ${netDirBps.toFixed(4)} bps`;
  }

  return {
    reference,
    resolved,
    gross,
    directionAdjusted,
    mfe,
    mae,
    costBps,
    longOpportunity,
    shortOpportunity,
    classification,
    explanation,
    resolvedAt: resolvedPoint.observed_at,
  };
}
