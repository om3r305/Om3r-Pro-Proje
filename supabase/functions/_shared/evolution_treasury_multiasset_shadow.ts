import type { TreasuryPosition } from "./evolution_treasury.ts";

export const MULTIASSET_SHADOW_POSITION_PREFIX = "treasury-multiasset-shadow:";
export const MULTIASSET_SHADOW_MAX_TICKET_USD = 20;
export const MULTIASSET_SHADOW_OPEN_MAX_AGE_SECONDS = 15 * 60;
export const MULTIASSET_SHADOW_MAINTENANCE_MAX_AGE_SECONDS = 35 * 60;
export const MULTIASSET_SHADOW_PROVIDER_QUALITY = "PUBLIC_UNOFFICIAL_SHADOW_ONLY";

export interface MultiassetAlphaDecisionRow {
  decisionId: string;
  observedAt: string;
  assetId: string;
  assetClass: string;
  providerTime: string;
  referencePrice: number | null;
  action: string;
  direction: number;
  evidenceScore: number | null;
  independentGroupCount: number;
  supportGroups: string[];
  requestedVirtualNotionalUsd: number | null;
  estimatedRoundTripCostBps: number | null;
  vetoReason: string | null;
  sessionState: string;
  dataLatencySeconds: number | null;
  shadowOnly: boolean;
  liveExecution: boolean;
  metadata: Record<string, unknown>;
}

export interface MultiassetShadowOpportunity {
  sourceDecisionId: string;
  observedAt: string;
  assetId: string;
  assetClass: string;
  referencePrice: number;
  direction: -1 | 1;
  signalScore: number;
  matureGroupCount: number;
  requestedCapitalUsd: number;
  roundTripCostBps: number;
  actionable: boolean;
  pitClear: boolean;
  recommendation: "ALLOW_MULTIASSET_SHADOW" | "MULTIASSET_SHADOW_WAIT";
  vetoReason: string | null;
  sessionState: string;
  dataLatencySeconds: number;
  providerQuality: string;
  linkedEventIds: string[];
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

function metadataString(row: MultiassetAlphaDecisionRow, key: string): string {
  const value = row.metadata?.[key];
  return value == null ? "" : String(value);
}

function metadataStrings(row: MultiassetAlphaDecisionRow, key: string): string[] {
  const value = row.metadata?.[key];
  return Array.isArray(value) ? value.map(String).filter(Boolean) : [];
}

export function isMultiassetShadowPosition(position: TreasuryPosition): boolean {
  return String(position.positionId || "").startsWith(MULTIASSET_SHADOW_POSITION_PREFIX);
}

export function buildMultiassetShadowOpportunities(
  rows: MultiassetAlphaDecisionRow[],
  _positions: TreasuryPosition[],
  nowIso: string,
): MultiassetShadowOpportunity[] {
  const nowMs = time(nowIso);
  if (nowMs == null) throw new Error("multiasset shadow: invalid cycle timestamp");

  const latest = new Map<string, MultiassetAlphaDecisionRow>();
  for (const row of rows) {
    const rowMs = time(row.observedAt);
    if (rowMs == null || rowMs > nowMs + 5_000) continue;
    const previous = latest.get(row.assetId);
    const previousMs = previous ? time(previous.observedAt) ?? -1 : -1;
    if (!previous || rowMs > previousMs) latest.set(row.assetId, row);
  }

  const out: MultiassetShadowOpportunity[] = [];
  for (const row of latest.values()) {
    const observedMs = time(row.observedAt);
    const providerMs = time(row.providerTime);
    const referencePrice = finite(row.referencePrice);
    const score = finite(row.evidenceScore);
    const requested = finite(row.requestedVirtualNotionalUsd);
    const cost = finite(row.estimatedRoundTripCostBps);
    const latency = finite(row.dataLatencySeconds);
    const direction = Number(row.direction);
    if (
      observedMs == null || providerMs == null || referencePrice == null || referencePrice <= 0 ||
      score == null || requested == null || cost == null || cost < 0 || latency == null || latency < 0 ||
      (direction !== 1 && direction !== -1)
    ) continue;

    const ageSeconds = Math.max(0, (nowMs - observedMs) / 1000);
    if (ageSeconds > MULTIASSET_SHADOW_MAINTENANCE_MAX_AGE_SECONDS) continue;

    const providerQuality = metadataString(row, "provider_quality");
    const directionSource = metadataString(row, "direction_source");
    const executionGrade = row.metadata?.execution_grade === true;
    const lane = metadataString(row, "shadow_lane");
    const linkedEventIds = metadataStrings(row, "linked_event_ids");
    const actionMatchesDirection =
      (row.action === "OPEN_LONG" && direction === 1) ||
      (row.action === "OPEN_SHORT" && direction === -1);

    const actionable =
      ageSeconds <= MULTIASSET_SHADOW_OPEN_MAX_AGE_SECONDS &&
      row.shadowOnly === true &&
      row.liveExecution === false &&
      row.vetoReason == null &&
      actionMatchesDirection &&
      row.independentGroupCount >= 2 &&
      row.supportGroups.includes("event_link") &&
      row.supportGroups.includes("market_reaction") &&
      score >= 0.52 &&
      requested > 0 &&
      requested <= MULTIASSET_SHADOW_MAX_TICKET_USD &&
      String(row.sessionState).toUpperCase() === "REGULAR" &&
      latency <= 15 * 60 &&
      providerQuality === MULTIASSET_SHADOW_PROVIDER_QUALITY &&
      directionSource === "OBSERVED_MARKET_REACTION_NOT_HEADLINE_GUESS" &&
      executionGrade === false &&
      lane === "MULTIASSET_EVENT_REACTION";

    out.push({
      sourceDecisionId: row.decisionId,
      observedAt: row.observedAt,
      assetId: row.assetId,
      assetClass: row.assetClass,
      referencePrice,
      direction: direction as -1 | 1,
      signalScore: Math.max(0, Math.min(1, score)),
      matureGroupCount: Math.max(0, Math.trunc(row.independentGroupCount)),
      requestedCapitalUsd: Math.min(MULTIASSET_SHADOW_MAX_TICKET_USD, Math.max(0, requested)),
      roundTripCostBps: cost,
      actionable,
      pitClear: actionable,
      recommendation: actionable ? "ALLOW_MULTIASSET_SHADOW" : "MULTIASSET_SHADOW_WAIT",
      vetoReason: row.vetoReason,
      sessionState: String(row.sessionState),
      dataLatencySeconds: latency,
      providerQuality,
      linkedEventIds,
    });
  }
  return out;
}
