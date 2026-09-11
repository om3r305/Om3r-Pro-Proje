import { canonicalIndependentGroup } from "./alpha_decision.ts";
import type { LaggedReliabilityEvidence } from "./evolution_alpha_intelligence.ts";

export interface DecisionSourceObservation {
  observationId: string;
  independentGroup: string;
  sensorFamily: string;
  sensorHorizon: string;
  direction: -1 | 1;
  observedAt: string;
}

export interface ReliabilitySnapshotCandidate {
  independentGroup: string;
  sensorFamily: string;
  sensorHorizon: string;
  sampleCount: number;
  bayesianHitRate: number;
  avgSignedBps: number;
  avgCostAdjustedSignedBps: number;
  outcomeHorizonSeconds: number;
  snapshotWindowEnd: string;
  snapshotGeneratedAt: string;
}

function tuple(group: string, family: string, horizon: string): string {
  return `${group}\u0000${family}\u0000${horizon}`;
}

/**
 * Bind lagged reliability to the exact sensor observations that actually supported
 * the canonical decision. This prevents two subtle look-alike errors:
 *
 * 1. ALPHA canonicalizes the five micro tape groups to `intrabar_tape`, while the
 *    reliability learner stores the original raw sensor group. Querying snapshots
 *    by `support_groups` therefore drops intrabar evidence entirely.
 * 2. A raw independent group can have multiple sensor families/horizons. Picking
 *    the largest sample row can attach history from a different sensor than the
 *    one that voted in the decision.
 *
 * `avgSignedBps` is already sensor-direction aligned by the reliability learner
 * (`sensor_direction * gross_return * 10000`), so it must NOT be multiplied by
 * the current decision direction again.
 */
export function bindLaggedReliabilityToDecision(params: {
  direction: -1 | 1;
  supportGroups: string[];
  sourceObservations: DecisionSourceObservation[];
  snapshotCandidates: ReliabilitySnapshotCandidate[];
}): LaggedReliabilityEvidence[] {
  const support = new Set(params.supportGroups.map(String));
  const snapshotsByTuple = new Map<string, ReliabilitySnapshotCandidate[]>();
  for (const row of params.snapshotCandidates) {
    const key = tuple(row.independentGroup, row.sensorFamily, row.sensorHorizon);
    const rows = snapshotsByTuple.get(key) ?? [];
    rows.push(row);
    snapshotsByTuple.set(key, rows);
  }

  const selected = new Map<string, LaggedReliabilityEvidence>();
  for (const source of params.sourceObservations) {
    if (source.direction !== params.direction) continue;
    const canonicalGroup = canonicalIndependentGroup(source.independentGroup);
    if (!support.has(canonicalGroup)) continue;
    const rows = snapshotsByTuple.get(tuple(source.independentGroup, source.sensorFamily, source.sensorHorizon)) ?? [];
    for (const row of rows) {
      const candidate: LaggedReliabilityEvidence = {
        group: canonicalGroup,
        sampleCount: row.sampleCount,
        bayesianHitRate: row.bayesianHitRate,
        avgSignedBps: row.avgSignedBps,
        avgCostAdjustedSignedBps: row.avgCostAdjustedSignedBps,
        outcomeHorizonSeconds: row.outcomeHorizonSeconds,
        snapshotWindowEnd: row.snapshotWindowEnd,
        snapshotGeneratedAt: row.snapshotGeneratedAt,
      };
      const prior = selected.get(canonicalGroup);
      if (!prior || candidate.sampleCount > prior.sampleCount) selected.set(canonicalGroup, candidate);
    }
  }

  return [...selected.values()].sort((a, b) => a.group.localeCompare(b.group));
}
