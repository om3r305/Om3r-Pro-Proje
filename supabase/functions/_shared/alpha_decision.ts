// Brian ALPHA v2 evidence-fusion compiler.
// Pure/import-safe. No I/O, no Supabase client, no exchange/order surface.
//
// IMPORTANT: consensus constants intentionally reuse Phase 4.0 Intrabar Eye's preregistered
// semantics (min 2 independent support groups, score >= 0.18). They are NOT tuned from the
// Sep-4 observed market move. Phase 3.7 remains a frozen/control input outside this module.

import type { DynamicCostQuote } from "./dynamic_cost.ts";

export const ALPHA_COMPILER_VERSION = "brian.alpha-decision-v2.1";
export const MIN_SUPPORT_GROUPS = 2;
export const MIN_CONSENSUS_SCORE = 0.18;

export type AlphaAction = "OPEN_LONG" | "OPEN_SHORT" | "WAIT" | "VETO";

export interface AlphaEvidenceRow {
  observationId: string;
  sourceKind: string;
  independentGroup: string;
  direction: number;
  strength: number;
  confidence: number;
  reliability: number;
  observedAt: string;
  horizon: string;
  fresh: boolean;
  reason: string;
}

export interface IntrabarVetoContext {
  eventId: string;
  direction: number;
  status: "WATCH" | "ACTIONABLE_SHADOW" | "VETOED_LATE_CHASE";
  lateChase: boolean;
  observedAt: string;
  fresh: boolean;
  reason: string;
}

export interface AlphaDecisionResult {
  compilerVersion: string;
  action: AlphaAction;
  direction: -1 | 0 | 1;
  evidenceScore: number;
  independentGroupCount: number;
  supportGroups: string[];
  conflictGroups: string[];
  sourceObservationIds: string[];
  ignoredObservationIds: string[];
  lateChaseVetoEventId: string | null;
  costQuality: string | null;
  estimatedRoundTripCostBps: number | null;
  fillable: boolean | null;
  reason: string;
  vetoReason: string | null;
}

function clip(value: number, low = 0, high = 1): number {
  return Math.max(low, Math.min(high, value));
}

function sign(value: number): -1 | 0 | 1 {
  return value > 0 ? 1 : value < 0 ? -1 : 0;
}

function finiteUnit(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error(`${label} must be in [0,1]`);
  return value;
}

function validateDirection(value: number): -1 | 0 | 1 {
  if (value !== -1 && value !== 0 && value !== 1) throw new Error("evidence direction must be -1, 0, or 1");
  return value;
}

/**
 * Dedupe correlated evidence by independent_group. The strongest quality row wins in each group;
 * source count is never treated as independence. Neutral/stale/unavailable rows cannot vote.
 */
export function dedupeIndependentEvidence(rows: AlphaEvidenceRow[]): {
  evidence: AlphaEvidenceRow[];
  ignoredObservationIds: string[];
} {
  const byGroup = new Map<string, AlphaEvidenceRow>();
  const ignored = new Set<string>();

  for (const row of rows) {
    if (!row.observationId || !row.independentGroup) throw new Error("evidence lineage/group is required");
    validateDirection(row.direction);
    finiteUnit(row.strength, "strength");
    finiteUnit(row.confidence, "confidence");
    finiteUnit(row.reliability, "reliability");
    if (!Number.isFinite(Date.parse(row.observedAt))) throw new Error("evidence observedAt must be a valid timestamp");
    if (!row.fresh || row.direction === 0) {
      ignored.add(row.observationId);
      continue;
    }

    const quality = row.strength * row.confidence * row.reliability;
    const prior = byGroup.get(row.independentGroup);
    const priorQuality = prior ? prior.strength * prior.confidence * prior.reliability : Number.NEGATIVE_INFINITY;
    if (!prior || quality > priorQuality || (quality === priorQuality && row.observedAt > prior.observedAt)) {
      if (prior) ignored.add(prior.observationId);
      byGroup.set(row.independentGroup, row);
    } else {
      ignored.add(row.observationId);
    }
  }

  return {
    evidence: [...byGroup.values()].sort((a, b) => a.independentGroup.localeCompare(b.independentGroup)),
    ignoredObservationIds: [...ignored].sort(),
  };
}

/**
 * Evidence fusion uses the same score form as Phase 4.0 Intrabar consensus:
 * mean signed quality × support ratio × breadth. No Sep-4 hindsight tuning.
 */
export function compileAlphaDecision(params: {
  evidenceRows: AlphaEvidenceRow[];
  costQuote?: DynamicCostQuote | null;
  intrabarContext?: IntrabarVetoContext | null;
}): AlphaDecisionResult {
  const deduped = dedupeIndependentEvidence(params.evidenceRows);
  const evidence = deduped.evidence;

  if (!evidence.length) {
    return {
      compilerVersion: ALPHA_COMPILER_VERSION,
      action: "WAIT",
      direction: 0,
      evidenceScore: 0,
      independentGroupCount: 0,
      supportGroups: [],
      conflictGroups: [],
      sourceObservationIds: [],
      ignoredObservationIds: deduped.ignoredObservationIds,
      lateChaseVetoEventId: null,
      costQuality: params.costQuote?.quality ?? null,
      estimatedRoundTripCostBps: params.costQuote?.estimatedRoundTripCostBps ?? null,
      fillable: params.costQuote?.fillable ?? null,
      reason: "no fresh directional independent evidence",
      vetoReason: null,
    };
  }

  const signed = evidence.map((row) => row.direction * row.strength * row.confidence * row.reliability);
  const aggregate = signed.reduce((a, b) => a + b, 0) / signed.length;
  const direction = sign(aggregate);
  const support = evidence.filter((row) => row.direction === direction);
  const conflicts = evidence.filter((row) => row.direction !== direction);
  const supportRatio = support.length / evidence.length;
  const breadth = 0.4 + 0.6 * Math.min(1, evidence.length / 3);
  const score = clip(Math.abs(aggregate) * (0.5 + 0.5 * supportRatio) * breadth);
  const eligible = direction !== 0 && support.length >= MIN_SUPPORT_GROUPS && score >= MIN_CONSENSUS_SCORE;

  const base = {
    compilerVersion: ALPHA_COMPILER_VERSION,
    direction,
    evidenceScore: score,
    independentGroupCount: evidence.length,
    supportGroups: support.map((row) => row.independentGroup).sort(),
    conflictGroups: conflicts.map((row) => row.independentGroup).sort(),
    sourceObservationIds: evidence.map((row) => row.observationId).sort(),
    ignoredObservationIds: deduped.ignoredObservationIds,
    costQuality: params.costQuote?.quality ?? null,
    estimatedRoundTripCostBps: params.costQuote?.estimatedRoundTripCostBps ?? null,
    fillable: params.costQuote?.fillable ?? null,
  };

  if (!eligible) {
    return {
      ...base,
      action: "WAIT",
      lateChaseVetoEventId: null,
      reason: `consensus below preregistered gate: support=${support.length}, score=${score.toFixed(6)}`,
      vetoReason: null,
    };
  }

  const late = params.intrabarContext;
  if (
    late?.fresh && late.status === "VETOED_LATE_CHASE" && late.lateChase &&
    late.direction === direction
  ) {
    return {
      ...base,
      action: "VETO",
      lateChaseVetoEventId: late.eventId,
      reason: "independent evidence aligned but current intrabar state is a late-chase veto",
      vetoReason: "LATE_CHASE",
    };
  }

  if (!params.costQuote) {
    return {
      ...base,
      action: "VETO",
      lateChaseVetoEventId: null,
      reason: "evidence is actionable but execution-cost evidence is unavailable",
      vetoReason: "COST_UNAVAILABLE",
    };
  }

  if (!params.costQuote.fillable) {
    return {
      ...base,
      action: "VETO",
      lateChaseVetoEventId: null,
      reason: "evidence is actionable but visible liquidity cannot fill the requested shadow notional",
      vetoReason: "INSUFFICIENT_VISIBLE_DEPTH",
    };
  }

  return {
    ...base,
    action: direction === 1 ? "OPEN_LONG" : "OPEN_SHORT",
    lateChaseVetoEventId: null,
    reason: params.costQuote.quality === "L2_OBSERVED"
      ? "fresh independent evidence passed preregistered consensus and synchronized L2 cost is fillable"
      : "fresh independent evidence passed preregistered consensus; degraded top-of-book cost used",
    vetoReason: null,
  };
}
