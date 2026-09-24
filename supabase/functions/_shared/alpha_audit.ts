export type AlphaAuditAction = "OPEN_LONG" | "OPEN_SHORT" | "WAIT" | "VETO";

export type AlphaAuditDecision = {
  observedAt: string;
  action: AlphaAuditAction;
  direction: -1 | 0 | 1;
  referencePrice: string | number | null;
  estimatedRoundTripCostBps: string | number | null;
};

export type AlphaAuditPricePoint = {
  observed_at: string;
  observed_mid_price: string | number;
  estimated_round_trip_cost_bps: string | number | null;
};

function finite(v: unknown, fallback = 0) {
  if (v == null || (typeof v === "string" && !v.trim())) return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Resolve the objective market outcome at a preregistered horizon without
 * requiring transaction-cost evidence.
 *
 * This is intentionally narrower than resolveAlphaAuditHorizon(): it may
 * support gross prospective measurement/reliability learning, but it does not
 * classify a missed opportunity or claim that an action beat costs.
 */
export function resolveAlphaAuditGrossHorizon(
  decision: AlphaAuditDecision,
  horizonSeconds: number,
  points: AlphaAuditPricePoint[],
) {
  const reference = finite(decision.referencePrice, NaN);
  if (!(reference > 0)) return null;

  const observedMs = Date.parse(decision.observedAt);
  if (!Number.isFinite(observedMs)) return null;
  const targetMs = observedMs + horizonSeconds * 1000;

  // Terminal resolution may use the first point at/after the requested horizon,
  // with at most +120s tolerance for sparse cadence. A pre-horizon point is
  // never relabelled as the horizon.
  const resolutionEligible = points
    .filter((p) => {
      const t = Date.parse(p.observed_at);
      return Number.isFinite(t) && t >= observedMs && t <= targetMs + 120_000;
    })
    .sort((a, b) => Date.parse(a.observed_at) - Date.parse(b.observed_at));
  if (!resolutionEligible.length) return null;

  const resolvedPoint = resolutionEligible.find(
    (p) => Date.parse(p.observed_at) >= targetMs,
  );
  if (!resolvedPoint) return null;
  if (Date.parse(resolvedPoint.observed_at) - targetMs > 120_000) return null;
  const resolved = finite(resolvedPoint.observed_mid_price);
  if (!(resolved > 0)) return null;

  // Excursion evidence is strictly in-horizon. The +120s terminal tolerance
  // can resolve terminal price only and cannot contaminate MFE/MAE.
  const excursionPoints = resolutionEligible.filter(
    (p) => Date.parse(p.observed_at) <= targetMs,
  );
  const prices = excursionPoints
    .map((p) => finite(p.observed_mid_price))
    .filter((p) => p > 0);
  if (!prices.length) return null;

  const maxPx = Math.max(...prices);
  const minPx = Math.min(...prices);
  const gross = resolved / reference - 1;
  const upExcursion = maxPx / reference - 1;
  const downExcursion = minPx / reference - 1;
  const directionAdjusted =
    decision.direction === 0 ? 0 : decision.direction * gross;

  // OPEN_SHORT gets direction-aware favorable/adverse semantics. WAIT/VETO
  // deliberately retain raw up/down excursions because a later cost-aware
  // missed-opportunity audit considers both directions.
  const mfe = decision.action === "OPEN_SHORT" ? -downExcursion : upExcursion;
  const mae = decision.action === "OPEN_SHORT" ? -upExcursion : downExcursion;

  return {
    reference,
    resolved,
    gross,
    directionAdjusted,
    mfe,
    mae,
    upExcursion,
    downExcursion,
    resolvedAt: resolvedPoint.observed_at,
  };
}

export function resolveAlphaAuditHorizon(
  decision: AlphaAuditDecision,
  horizonSeconds: number,
  points: AlphaAuditPricePoint[],
) {
  const grossResolution = resolveAlphaAuditGrossHorizon(
    decision,
    horizonSeconds,
    points,
  );
  if (!grossResolution) return null;

  const observedMs = Date.parse(decision.observedAt);
  const targetMs = observedMs + horizonSeconds * 1000;

  // Cost evidence is stricter than terminal-price tolerance: only evidence
  // captured at/before the exact horizon can support a cost-aware receipt.
  const fallbackCost = points
    .filter((p) => {
      const t = Date.parse(p.observed_at);
      return Number.isFinite(t) && t >= observedMs && t <= targetMs;
    })
    .sort((a, b) => Date.parse(a.observed_at) - Date.parse(b.observed_at))
    .map((p) => finite(p.estimated_round_trip_cost_bps, NaN))
    .find((x) => Number.isFinite(x) && x >= 0);

  const decisionCost = finite(decision.estimatedRoundTripCostBps, NaN);
  if (
    !(Number.isFinite(decisionCost) && decisionCost >= 0) &&
    fallbackCost === undefined
  ) {
    // Unknown cost never becomes 0 bps. Gross outcome measurement may be
    // resolved separately, but a cost-aware receipt remains fail-closed.
    return null;
  }
  const costBps = Number.isFinite(decisionCost) && decisionCost >= 0
    ? decisionCost
    : fallbackCost!;

  const longOpportunity = grossResolution.upExcursion * 10_000 > costBps;
  const shortOpportunity = -grossResolution.downExcursion * 10_000 > costBps;

  let classification: string;
  let explanation: string;
  if (decision.action === "WAIT" || decision.action === "VETO") {
    if (longOpportunity && shortOpportunity) {
      classification = "WAIT_VOLATILE_TWO_SIDED";
    } else if (longOpportunity) {
      classification = "WAIT_MISSED_LONG";
    } else if (shortOpportunity) {
      classification = "WAIT_MISSED_SHORT";
    } else {
      classification = "WAIT_JUSTIFIED_BY_COST";
    }
    explanation =
      `ex-post in-horizon excursion from the immutable decision-time reference compared with contemporaneous round-trip cost ${costBps.toFixed(4)} bps; this receipt audits the decision and does not retune thresholds`;
  } else {
    const netDirBps =
      grossResolution.directionAdjusted * 10_000 - costBps;
    classification = netDirBps > 0
      ? "ACTION_FAVORABLE_AFTER_COST"
      : "ACTION_UNFAVORABLE_AFTER_COST";
    explanation =
      `direction-adjusted terminal return from immutable decision-time reference minus contemporaneous round-trip cost = ${netDirBps.toFixed(4)} bps`;
  }

  return {
    ...grossResolution,
    costBps,
    longOpportunity,
    shortOpportunity,
    classification,
    explanation,
  };
}
