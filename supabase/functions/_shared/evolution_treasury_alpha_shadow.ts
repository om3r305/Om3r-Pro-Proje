import type { TreasuryOpportunity, TreasuryPosition } from "./evolution_treasury.ts";

export const CANONICAL_ALPHA_SHADOW_POSITION_PREFIX = "treasury-alpha-shadow:";
export const CANONICAL_ALPHA_SHADOW_MAX_TICKET_USD = 20;
export const CANONICAL_ALPHA_SHADOW_OPEN_MAX_AGE_SECONDS = 5 * 60;
export const CANONICAL_ALPHA_SHADOW_MAINTENANCE_MAX_AGE_SECONDS = 25 * 60;

export type CanonicalAlphaShadowAction = "OPEN_LONG" | "OPEN_SHORT" | "WAIT" | "VETO" | string;

export interface CanonicalAlphaDecisionRow {
  decisionId: string;
  observedAt: string;
  assetId: string;
  referencePrice: number | null;
  action: CanonicalAlphaShadowAction;
  direction: number;
  evidenceScore: number | null;
  independentGroupCount: number;
  requestedVirtualNotionalUsd: number | null;
  estimatedRoundTripCostBps: number | null;
  vetoReason: string | null;
  evidenceClass: string;
  shadowOnly: boolean;
  liveExecution: boolean;
  costQuality: string | null;
  l2RuntimeStatus: string | null;
}

export interface CanonicalAlphaShadowOpportunity extends TreasuryOpportunity {
  shadowLane: "CANONICAL_ALPHA_SHADOW";
  alphaAction: CanonicalAlphaShadowAction;
  requestedCapitalUsd: number;
  signalScore: number;
  actionable: boolean;
}

function finite(value: unknown): number | null {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function time(value: string): number | null {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function isCanonicalAlphaShadowPosition(position: TreasuryPosition): boolean {
  return String(position.positionId || "").startsWith(CANONICAL_ALPHA_SHADOW_POSITION_PREFIX);
}

function canonicalOpenDirection(action: string): -1 | 1 | null {
  if (action === "OPEN_LONG") return 1;
  if (action === "OPEN_SHORT") return -1;
  return null;
}

function rowCanOpen(row: CanonicalAlphaDecisionRow, ageSeconds: number): boolean {
  const expectedDirection = canonicalOpenDirection(String(row.action || ""));
  const requested = finite(row.requestedVirtualNotionalUsd);
  const cost = finite(row.estimatedRoundTripCostBps);
  return Boolean(
    expectedDirection != null && row.direction === expectedDirection &&
    row.shadowOnly === true && row.liveExecution === false &&
    row.evidenceClass === "PROSPECTIVE_DEVELOPMENT_SHADOW" &&
    !row.vetoReason &&
    requested != null && requested > 0 && requested <= CANONICAL_ALPHA_SHADOW_MAX_TICKET_USD &&
    cost != null && cost >= 0 &&
    row.costQuality === "L2_OBSERVED" && row.l2RuntimeStatus === "OBSERVED" &&
    ageSeconds >= 0 && ageSeconds <= CANONICAL_ALPHA_SHADOW_OPEN_MAX_AGE_SECONDS
  );
}

/**
 * Converts only the newest canonical ALPHA decision per asset into a Treasury SHADOW-lane
 * candidate. OPEN signals may create tiny virtual positions using ALPHA's own requested
 * ticket ($3/$5/$10/$20 today). WAIT/VETO/invalid/stale rows are emitted only for an
 * already-open fallback position so the Treasury can flatten it fail-closed.
 *
 * This helper never promotes EXPECTED_EDGE, never mutates canonical ALPHA, and never
 * enables live execution.
 */
export function buildCanonicalAlphaShadowOpportunities(
  rows: CanonicalAlphaDecisionRow[],
  positions: TreasuryPosition[],
  nowIso: string,
): CanonicalAlphaShadowOpportunity[] {
  const nowMs = time(nowIso);
  if (nowMs == null) return [];
  const existingByAsset = new Map(
    positions.filter(isCanonicalAlphaShadowPosition).map((position) => [position.assetId, position]),
  );
  const newestByAsset = new Map<string, CanonicalAlphaDecisionRow>();
  for (const row of rows) {
    const at = time(row.observedAt);
    if (!row.assetId || at == null || at > nowMs + 5_000) continue;
    const ageSeconds = (nowMs - at) / 1000;
    if (ageSeconds > CANONICAL_ALPHA_SHADOW_MAINTENANCE_MAX_AGE_SECONDS) continue;
    const previous = newestByAsset.get(row.assetId);
    const previousAt = previous ? time(previous.observedAt) ?? -1 : -1;
    if (!previous || at > previousAt) newestByAsset.set(row.assetId, row);
  }

  const out: CanonicalAlphaShadowOpportunity[] = [];
  for (const [assetId, row] of newestByAsset) {
    const at = time(row.observedAt)!;
    const ageSeconds = Math.max(0, (nowMs - at) / 1000);
    const existing = existingByAsset.get(assetId);
    const referencePrice = finite(row.referencePrice);
    if (referencePrice == null || referencePrice <= 0) continue;
    const requested = finite(row.requestedVirtualNotionalUsd) ?? 0;
    const rowCost = finite(row.estimatedRoundTripCostBps);
    const roundTripCostBps = rowCost != null && rowCost >= 0 ? rowCost : existing?.roundTripCostBps ?? null;
    if (roundTripCostBps == null || roundTripCostBps < 0) continue;

    const actionable = rowCanOpen(row, ageSeconds);
    const openDirection = canonicalOpenDirection(String(row.action || ""));
    if (!existing && !actionable) continue;
    const direction = (openDirection ?? existing?.direction ?? (row.direction === -1 ? -1 : row.direction === 1 ? 1 : null));
    if (direction == null) continue;

    out.push({
      assetId,
      direction,
      observedAt: row.observedAt,
      referencePrice,
      // Canonical ALPHA's evidence score is deliberately NOT an expected-return estimate.
      expectedNetEdgeBps: 0,
      roundTripCostBps,
      reliabilityConfidence: 0.5,
      matureGroupCount: Math.max(0, Math.trunc(row.independentGroupCount || 0)),
      pitClear: actionable,
      recommendation: actionable ? "ALLOW_CANONICAL_ALPHA_SHADOW" : "CANONICAL_ALPHA_SHADOW_REVOKED",
      sourceDecisionId: row.decisionId,
      shadowLane: "CANONICAL_ALPHA_SHADOW",
      alphaAction: row.action,
      requestedCapitalUsd: actionable ? requested : 0,
      signalScore: Math.max(0, finite(row.evidenceScore) ?? 0),
      actionable,
    });
  }
  return out;
}
