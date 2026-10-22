// Unified SHADOW treasury lane (operator request 2026-10-22): every signal family is turned into the
// same TreasuryOpportunity and sized by ONE planner (evolution_treasury.ts) on ONE number — the
// expected net edge after costs — estimated from point-in-time realised outcomes of that signal
// family (brian_treasury_signal_outcomes). Consequences, by design:
//   * cost gate: nothing opens unless shrunk expected gross minus uncertainty minus cost >= minNetEdgeBps;
//   * evidence-based sizing: the planner scales capital with edge, reliability and maturity, up to the
//     whole cashbox for exceptional, well-sampled evidence; weak evidence keeps size small or zero;
//   * anti-churn: a unified position is held at least minHoldSeconds (the horizon the edge was measured
//     on) unless the risk stop fires or a validated opposite signal appears.
// Nothing here trades: SHADOW only.

import type { TreasuryOpportunity, TreasuryPosition } from "./evolution_treasury.ts";
import type { MultiassetAlphaDecisionRow } from "./evolution_treasury_multiasset_shadow.ts";

export const UNIFIED_TREASURY_VERSION = "brian.treasury-unified.v1";
export const UNIFIED_POSITION_PREFIX = "treasury-unified:";
export const UNIFIED_RULES = Object.freeze({
  lane: "MULTIASSET_EVENT_REACTION",
  horizonMinutes: 60,
  minSamples: 100,
  shrinkSamples: 100, // gross is shrunk by n / (n + shrinkSamples) toward zero
  uncertaintyZ: 1, // minus one standard error of the segment mean
  minNetEdgeBps: 3,
  minHoldSeconds: 60 * 60,
  maxSignalAgeSeconds: 15 * 60,
  maxMarkAgeSeconds: 35 * 60,
});

export interface SegmentEdge {
  lane: string;
  assetClass: string;
  direction: -1 | 1;
  n: number;
  grossBps: number;
  seBps: number;
  hitRate: number;
}

export interface SegmentVerdict {
  netEdgeBps: number | null;
  shrunkGrossBps: number | null;
  penaltyBps: number | null;
  reliability: number;
  reason: "OK" | "NO_SEGMENT" | "TOO_FEW_SAMPLES" | "INVALID_SEGMENT";
}

const finite = (v: unknown) => typeof v === "number" && Number.isFinite(v);
const time = (v: string) => {
  const t = Date.parse(v);
  return Number.isFinite(t) ? t : null;
};
export const segmentKey = (lane: string, assetClass: string, direction: number) => `${lane}|${assetClass}|${direction}`;

export function segmentVerdict(segment: SegmentEdge | undefined, costBps: number): SegmentVerdict {
  if (!segment) return { netEdgeBps: null, shrunkGrossBps: null, penaltyBps: null, reliability: 0.5, reason: "NO_SEGMENT" };
  if (![segment.n, segment.grossBps, segment.seBps, segment.hitRate, costBps].every(finite) || segment.n < 0 || costBps < 0) {
    return { netEdgeBps: null, shrunkGrossBps: null, penaltyBps: null, reliability: 0.5, reason: "INVALID_SEGMENT" };
  }
  const w = segment.n / (segment.n + UNIFIED_RULES.shrinkSamples);
  const reliability = 0.5 + (Math.min(1, Math.max(0, segment.hitRate)) - 0.5) * w;
  if (segment.n < UNIFIED_RULES.minSamples) {
    return { netEdgeBps: null, shrunkGrossBps: null, penaltyBps: null, reliability, reason: "TOO_FEW_SAMPLES" };
  }
  const shrunkGrossBps = segment.grossBps * w;
  const penaltyBps = UNIFIED_RULES.uncertaintyZ * segment.seBps;
  return { netEdgeBps: shrunkGrossBps - penaltyBps - costBps, shrunkGrossBps, penaltyBps, reliability, reason: "OK" };
}

export function isUnifiedPosition(p: TreasuryPosition): boolean {
  return String(p.positionId || "").startsWith(UNIFIED_POSITION_PREFIX);
}

function signalIsEntryGrade(row: MultiassetAlphaDecisionRow, nowMs: number): boolean {
  const at = time(row.observedAt);
  const dir = Number(row.direction);
  const actionMatches = (row.action === "OPEN_LONG" && dir === 1) || (row.action === "OPEN_SHORT" && dir === -1);
  return at != null && (nowMs - at) / 1000 <= UNIFIED_RULES.maxSignalAgeSeconds && at <= nowMs + 5_000 &&
    actionMatches && row.vetoReason == null && row.shadowOnly === true && row.liveExecution === false &&
    String(row.sessionState).toUpperCase() === "REGULAR";
}

export interface UnifiedBuild {
  opportunities: TreasuryOpportunity[];
  verdicts: Array<{ assetId: string; assetClass: string; direction: number; action: string; entryGrade: boolean } & SegmentVerdict>;
}

/**
 * One TreasuryOpportunity per asset from its newest multiasset decision:
 *  - entry-grade signal + positive segment net edge  -> ALLOW_EDGE with that net edge;
 *  - anything else -> a mark-only, non-allowed opportunity (so open positions are always valued at a
 *    current price, never silently at their entry price), oriented to the open position if there is one.
 */
export function buildUnifiedOpportunities(
  rows: MultiassetAlphaDecisionRow[],
  segments: SegmentEdge[],
  positions: TreasuryPosition[],
  nowIso: string,
): UnifiedBuild {
  const nowMs = time(nowIso);
  if (nowMs == null) throw new Error("unified treasury: invalid cycle timestamp");
  const seg = new Map(segments.map((s) => [segmentKey(s.lane, s.assetClass, s.direction), s]));
  const open = new Map(positions.filter(isUnifiedPosition).map((p) => [p.assetId, p]));
  const latest = new Map<string, MultiassetAlphaDecisionRow>();
  for (const r of rows) {
    const at = time(r.observedAt);
    if (at == null || at > nowMs + 5_000 || (nowMs - at) / 1000 > UNIFIED_RULES.maxMarkAgeSeconds) continue;
    const prev = latest.get(r.assetId);
    if (!prev || at > (time(prev.observedAt) ?? -1)) latest.set(r.assetId, r);
  }
  const opportunities: TreasuryOpportunity[] = [];
  const verdicts: UnifiedBuild["verdicts"] = [];
  for (const r of latest.values()) {
    const price = r.referencePrice;
    const cost = r.estimatedRoundTripCostBps;
    if (price == null || !(price > 0) || cost == null || !(cost >= 0)) continue;
    const position = open.get(r.assetId);
    const rawDir = Number(r.direction);
    const entryGrade = signalIsEntryGrade(r, nowMs);
    const direction = (entryGrade ? rawDir : position?.direction ?? (rawDir === 1 || rawDir === -1 ? rawDir : 0)) as -1 | 1 | 0;
    if (direction !== 1 && direction !== -1) continue;
    const v = segmentVerdict(seg.get(segmentKey(UNIFIED_RULES.lane, r.assetClass, direction)), cost);
    const allowed = entryGrade && v.netEdgeBps != null && v.netEdgeBps >= UNIFIED_RULES.minNetEdgeBps;
    verdicts.push({ assetId: r.assetId, assetClass: r.assetClass, direction, action: r.action, entryGrade, ...v });
    if (!allowed && !position) continue; // nothing to enter, nothing to value
    opportunities.push({
      assetId: r.assetId,
      direction,
      observedAt: r.observedAt,
      referencePrice: price,
      expectedNetEdgeBps: allowed ? v.netEdgeBps! : Math.min(0, v.netEdgeBps ?? 0),
      roundTripCostBps: cost,
      reliabilityConfidence: v.reliability,
      matureGroupCount: Math.max(0, Math.trunc(r.independentGroupCount)),
      pitClear: allowed,
      recommendation: allowed ? "ALLOW_EDGE" : `UNIFIED_WAIT_${v.reason === "OK" ? "EDGE_BELOW_COST" : v.reason}`,
      sourceDecisionId: r.decisionId,
    });
  }
  return { opportunities, verdicts };
}

export function parseSegmentEdges(rows: unknown): SegmentEdge[] {
  return (Array.isArray(rows) ? rows : []).flatMap((x) => {
    const r = x as Record<string, unknown>;
    const direction = Number(r.direction);
    const out = {
      lane: String(r.lane ?? ""), assetClass: String(r.asset_class ?? ""), direction: direction as -1 | 1,
      n: Number(r.n), grossBps: Number(r.gross_bps), seBps: Number(r.se_bps), hitRate: Number(r.hit_rate),
    };
    return (direction === 1 || direction === -1) && out.lane && out.assetClass &&
        [out.n, out.grossBps, out.seBps, out.hitRate].every(Number.isFinite)
      ? [out]
      : [];
  });
}
