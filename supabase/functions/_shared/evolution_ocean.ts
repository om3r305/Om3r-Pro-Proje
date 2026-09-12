export const BRIAN_OCEAN_VERSION = "brian.ocean-shadow.v1";
export type OceanCommandKind = "START" | "STOP";

export interface OceanCommand {
  commandId: string;
  runId: string;
  command: OceanCommandKind;
  requestedAt: string;
  durationHours: 24 | 48 | null;
  reason: string | null;
}

export interface OceanRunState {
  runId: string;
  startedAt: string;
  plannedEndAt: string;
  stoppedAt: string | null;
  effectiveEndAt: string;
  durationHours: 24 | 48;
  status: "ACTIVE" | "ENDED";
  startCommandId: string;
  stopCommandId: string | null;
}

export interface OceanTreasuryPoint {
  observedAt: string;
  equityUsd: number;
  cashUsd: number;
  deploymentUsd: number;
  realizedPnlUsd: number;
  cumulativeCostsUsd: number;
  openPositions: number;
}

export interface OceanReportInput {
  run: OceanRunState;
  treasuryStart: OceanTreasuryPoint | null;
  treasuryEnd: OceanTreasuryPoint | null;
  treasuryActions: number;
  replacements: number;
  newSources: number;
  newHypotheses: number;
  newCodeCandidates: number;
  experimentResults: number;
  promotionCandidates: number;
  rejectedPromotions: number;
  driftEvents: number;
  capabilityEvents: number;
  missedOpportunities: number;
  alphaOutcomeSamples: number;
  alphaFavorableAfterCost: number;
  collectorRuns: number;
  collectorFailures: number;
  degradedRuns: number;
}

export interface OceanReportSummary {
  version: typeof BRIAN_OCEAN_VERSION;
  runId: string;
  startedAt: string;
  endedAt: string;
  durationHours: number;
  treasury: {
    beginningEquityUsd: number | null;
    endingEquityUsd: number | null;
    netPnlUsd: number | null;
    endingCashUsd: number | null;
    endingDeploymentUsd: number | null;
    realizedPnlUsd: number | null;
    cumulativeCostsUsd: number | null;
    endingOpenPositions: number | null;
    actions: number;
    replacements: number;
  };
  evolution: {
    newSources: number;
    newHypotheses: number;
    newCodeCandidates: number;
    experimentResults: number;
    promotionCandidates: number;
    rejectedPromotions: number;
    driftEvents: number;
    capabilityEvents: number;
  };
  alpha: {
    outcomeSamples: number;
    favorableAfterCost: number;
    favorableAfterCostRate: number | null;
    missedOpportunities: number;
  };
  health: {
    collectorRuns: number;
    collectorFailures: number;
    degradedRuns: number;
    healthyRunRate: number | null;
  };
  shadowOnly: true;
  liveExecution: false;
}

function time(value: string): number | null {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function validDuration(value: number | null): value is 24 | 48 {
  return value === 24 || value === 48;
}

export function deriveOceanRuns(commands: OceanCommand[], nowIso: string): OceanRunState[] {
  const now = time(nowIso);
  if (now == null) throw new Error("invalid nowIso");
  const byRun = new Map<string, OceanCommand[]>();
  for (const command of commands) {
    if (!command.runId || !command.commandId || time(command.requestedAt) == null) continue;
    const rows = byRun.get(command.runId) ?? [];
    rows.push(command);
    byRun.set(command.runId, rows);
  }
  const runs: OceanRunState[] = [];
  for (const [runId, rows] of byRun) {
    rows.sort((a, b) => Number(time(a.requestedAt)) - Number(time(b.requestedAt)));
    const start = rows.find((row) => row.command === "START" && validDuration(row.durationHours));
    if (!start || !validDuration(start.durationHours)) continue;
    const startedMs = time(start.requestedAt)!;
    const plannedEndMs = startedMs + start.durationHours * 3600_000;
    const stop = rows.find((row) => row.command === "STOP" && Number(time(row.requestedAt)) >= startedMs);
    const stopMs = stop ? time(stop.requestedAt) : null;
    const effectiveEndMs = stopMs == null ? plannedEndMs : Math.min(plannedEndMs, stopMs);
    runs.push({
      runId,
      startedAt: new Date(startedMs).toISOString(),
      plannedEndAt: new Date(plannedEndMs).toISOString(),
      stoppedAt: stopMs == null ? null : new Date(stopMs).toISOString(),
      effectiveEndAt: new Date(effectiveEndMs).toISOString(),
      durationHours: start.durationHours,
      status: now < effectiveEndMs ? "ACTIVE" : "ENDED",
      startCommandId: start.commandId,
      stopCommandId: stop?.commandId ?? null,
    });
  }
  return runs.sort((a, b) => Date.parse(b.startedAt) - Date.parse(a.startedAt));
}

export function activeOceanRun(commands: OceanCommand[], nowIso: string): OceanRunState | null {
  return deriveOceanRuns(commands, nowIso).find((run) => run.status === "ACTIVE") ?? null;
}

export function buildOceanReport(input: OceanReportInput): OceanReportSummary {
  const start = input.treasuryStart;
  const end = input.treasuryEnd;
  const beginning = start?.equityUsd ?? null;
  const ending = end?.equityUsd ?? null;
  const netPnl = beginning != null && ending != null ? ending - beginning : null;
  const outcomeRate = input.alphaOutcomeSamples > 0 ? input.alphaFavorableAfterCost / input.alphaOutcomeSamples : null;
  const healthyRuns = Math.max(0, input.collectorRuns - input.collectorFailures - input.degradedRuns);
  const healthyRunRate = input.collectorRuns > 0 ? healthyRuns / input.collectorRuns : null;
  const elapsedHours = Math.max(0, (Date.parse(input.run.effectiveEndAt) - Date.parse(input.run.startedAt)) / 3600_000);
  return {
    version: BRIAN_OCEAN_VERSION,
    runId: input.run.runId,
    startedAt: input.run.startedAt,
    endedAt: input.run.effectiveEndAt,
    durationHours: elapsedHours,
    treasury: {
      beginningEquityUsd: beginning,
      endingEquityUsd: ending,
      netPnlUsd: netPnl,
      endingCashUsd: end?.cashUsd ?? null,
      endingDeploymentUsd: end?.deploymentUsd ?? null,
      realizedPnlUsd: end?.realizedPnlUsd ?? null,
      cumulativeCostsUsd: end?.cumulativeCostsUsd ?? null,
      endingOpenPositions: end?.openPositions ?? null,
      actions: Math.max(0, input.treasuryActions),
      replacements: Math.max(0, input.replacements),
    },
    evolution: {
      newSources: Math.max(0, input.newSources),
      newHypotheses: Math.max(0, input.newHypotheses),
      newCodeCandidates: Math.max(0, input.newCodeCandidates),
      experimentResults: Math.max(0, input.experimentResults),
      promotionCandidates: Math.max(0, input.promotionCandidates),
      rejectedPromotions: Math.max(0, input.rejectedPromotions),
      driftEvents: Math.max(0, input.driftEvents),
      capabilityEvents: Math.max(0, input.capabilityEvents),
    },
    alpha: {
      outcomeSamples: Math.max(0, input.alphaOutcomeSamples),
      favorableAfterCost: Math.max(0, input.alphaFavorableAfterCost),
      favorableAfterCostRate: outcomeRate,
      missedOpportunities: Math.max(0, input.missedOpportunities),
    },
    health: {
      collectorRuns: Math.max(0, input.collectorRuns),
      collectorFailures: Math.max(0, input.collectorFailures),
      degradedRuns: Math.max(0, input.degradedRuns),
      healthyRunRate,
    },
    shadowOnly: true,
    liveExecution: false,
  };
}
