import {
  EVOLUTION_EVIDENCE_CLASS,
  type CapabilityDomain,
  type CapabilityHealth,
  type CapabilitySnapshot,
  type EvolutionStage,
  type WorldSourceCandidate,
} from "./evolution_contract.ts";

export const EVOLUTION_CORE_VERSION = "brian.evolution-core.v1";

export interface CollectorRunLike {
  collector_id: string;
  started_at?: string | null;
  finished_at?: string | null;
  status?: string | null;
  observed_records?: number | null;
  stored_records?: number | null;
  degraded_sources?: string[] | null;
  error_class?: string | null;
  error_message?: string | null;
}

export interface IntelEventLike {
  event_id?: string | null;
  source_id?: string | null;
  provenance_uri?: string | null;
  source_kind?: string | null;
  trust_class?: string | null;
  first_observed_at?: string | null;
  published_at?: string | null;
  claim?: string | null;
  asset?: string | null;
}

export interface CapabilityDefinition {
  capabilityId: string;
  domain: CapabilityDomain;
  name: string;
  description: string;
  collectorIds: string[];
  staleAfterSeconds: number;
  dependencies: string[];
  limitations: string[];
  expectedStage: EvolutionStage;
}

export interface CapabilityGap {
  gapId: string;
  capabilityId: string;
  domain: CapabilityDomain;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  reason: string;
  suggestedAction: string;
  evidenceRefs: string[];
}

export interface SourceAssessment {
  authorityScore: number;
  freshnessScore: number;
  manipulationPenalty: number;
  corroborationPenalty: number;
  accessPenalty: number;
  trustScore: number;
  eligibleForResearch: boolean;
  eligibleForDecisionEvidence: boolean;
  reasons: string[];
}

export interface EvolutionDashboardModel {
  schemaVersion: string;
  observedAt: string;
  overall: "HEALTHY" | "DEGRADED" | "BUILDING";
  capabilityCounts: Record<CapabilityHealth, number>;
  stageCounts: Partial<Record<EvolutionStage, number>>;
  criticalGaps: CapabilityGap[];
  topGaps: CapabilityGap[];
  sourceCounts: {
    total: number;
    official: number;
    verifying: number;
    rejected: number;
    researchEligible: number;
    decisionEligible: number;
  };
  latestJournalAt: string | null;
  cloudIndependent: true;
  shadowOnly: true;
  liveExecution: false;
}

const MAIN_CAPABILITIES: CapabilityDefinition[] = [
  {
    capabilityId: "market.universe",
    domain: "MARKET_DATA",
    name: "Market Universe Radar",
    description: "Finds and maintains the tradable/observable market universe.",
    collectorIds: ["brian-universe-collector"],
    staleAfterSeconds: 600,
    dependencies: [],
    limitations: ["coverage is provider constrained", "discovery is not an execution signal"],
    expectedStage: "ACTIVE",
  },
  {
    capabilityId: "market.sensor-mesh",
    domain: "SENSOR",
    name: "Global Sensor Mesh",
    description: "Produces prospective structure, momentum, mean-reversion and market micro observations.",
    collectorIds: ["brian-sensor-mesh"],
    staleAfterSeconds: 600,
    dependencies: ["market.universe"],
    limitations: ["canonical reliability feedback is not yet closed-loop"],
    expectedStage: "ACTIVE",
  },
  {
    capabilityId: "market.intrabar",
    domain: "SENSOR",
    name: "Intrabar Reaction Eye",
    description: "Observes early 1m velocity, volume, breakout, reclaim and taker-flow reactions.",
    collectorIds: ["brian-intrabar-eye"],
    staleAfterSeconds: 240,
    dependencies: ["market.universe"],
    limitations: ["intrabar sub-sensors are correlated and must remain grouped"],
    expectedStage: "ACTIVE",
  },
  {
    capabilityId: "market.derivatives",
    domain: "MARKET_DATA",
    name: "Derivatives Eye",
    description: "Observes funding, open interest and derivatives taker context.",
    collectorIds: ["phase39-binance-usdm-derivatives", "brian-derivatives-eye"],
    staleAfterSeconds: 900,
    dependencies: ["market.universe"],
    limitations: ["venue-specific", "not a standalone directional truth source"],
    expectedStage: "ACTIVE",
  },
  {
    capabilityId: "market.l2",
    domain: "MARKET_DATA",
    name: "Real L2 Capture",
    description: "Captures order-book state for spread, depth, fillability and dynamic-cost evidence.",
    collectorIds: ["brian-l2-capture"],
    staleAfterSeconds: 300,
    dependencies: ["market.universe"],
    limitations: ["static depth does not fully model queue movement or adverse selection"],
    expectedStage: "ACTIVE",
  },
  {
    capabilityId: "world.news-discovery",
    domain: "NEWS_MACRO",
    name: "Global News Discovery",
    description: "Discovers market-relevant public news and event candidates.",
    collectorIds: ["phase39-gdelt-news", "brian-news-eye"],
    staleAfterSeconds: 1200,
    dependencies: [],
    limitations: ["headline discovery is not source truth", "corroboration required"],
    expectedStage: "ACTIVE",
  },
  {
    capabilityId: "world.official-macro",
    domain: "NEWS_MACRO",
    name: "Official Macro Eye",
    description: "Captures official macro releases and authority-first context.",
    collectorIds: ["brian-official-macro-eye"],
    staleAfterSeconds: 7200,
    dependencies: [],
    limitations: ["release cadence is event-driven", "context may be non-directional"],
    expectedStage: "ACTIVE",
  },
  {
    capabilityId: "world.fx",
    domain: "CROSS_ASSET",
    name: "FX Eye",
    description: "Captures official FX context for cross-asset reasoning.",
    collectorIds: ["phase39-ecb-fx", "brian-fx-eye"],
    staleAfterSeconds: 86400,
    dependencies: [],
    limitations: ["daily official cadence", "not a complete rates/yields model"],
    expectedStage: "ACTIVE",
  },
  {
    capabilityId: "alpha.compiler",
    domain: "ALPHA",
    name: "ALPHA Decision Compiler",
    description: "Fuses independent evidence groups into direction-only prospective ALPHA decisions.",
    collectorIds: ["brian-alpha-decision-compiler-v2"],
    staleAfterSeconds: 240,
    dependencies: ["market.sensor-mesh", "market.intrabar", "market.derivatives"],
    limitations: ["evidence score is not expected return", "portfolio sizing is not canonical yet"],
    expectedStage: "ACTIVE",
  },
  {
    capabilityId: "research.missed-opportunity",
    domain: "RESEARCH",
    name: "Missed Opportunity Auditor",
    description: "Resolves prospective decisions after fixed horizons and classifies after-cost outcomes.",
    collectorIds: ["brian-missed-opportunity-auditor-v2"],
    staleAfterSeconds: 1200,
    dependencies: ["alpha.compiler"],
    limitations: ["audits but does not mutate canonical policy"],
    expectedStage: "ACTIVE",
  },
  {
    capabilityId: "research.calibration",
    domain: "RESEARCH",
    name: "ALPHA Calibration Challenger",
    description: "Challenges canonical ALPHA using mature prospective reliability and cost-adjusted evidence.",
    collectorIds: ["brian-alpha-calibration-challenger-v1"],
    staleAfterSeconds: 7200,
    dependencies: ["alpha.compiler", "research.missed-opportunity"],
    limitations: ["canonical_mutation=false", "promotion remains gated"],
    expectedStage: "ACTIVE",
  },
  {
    capabilityId: "evolution.world-explorer",
    domain: "WORLD_SOURCE",
    name: "World Explorer",
    description: "Discovers and evaluates candidate public information sources without treating discovery as truth.",
    collectorIds: ["brian-evolution-orchestrator-v1"],
    staleAfterSeconds: 1800,
    dependencies: ["world.news-discovery"],
    limitations: ["Layer 1 candidate discovery only", "no direct ALPHA influence"],
    expectedStage: "EXPERIMENTAL",
  },
  {
    capabilityId: "evolution.capability-graph",
    domain: "OBSERVABILITY",
    name: "Capability Graph",
    description: "Lets Brian describe what it can do, what is stale and what is missing.",
    collectorIds: ["brian-evolution-orchestrator-v1"],
    staleAfterSeconds: 1800,
    dependencies: [],
    limitations: ["runtime inventory is evidence-based, not filesystem omniscience"],
    expectedStage: "EXPERIMENTAL",
  },
];

const FUTURE_REQUIRED: Array<Omit<CapabilityDefinition, "collectorIds" | "staleAfterSeconds">> = [
  {
    capabilityId: "world.entity-graph",
    domain: "ENTITY_GRAPH",
    name: "Global Entity & Supply-Chain Graph",
    description: "Connects companies, products, people, countries, commodities and assets.",
    dependencies: ["evolution.world-explorer"],
    limitations: ["Layer 2 target"],
    expectedStage: "DISCOVERED",
  },
  {
    capabilityId: "world.future-calendar",
    domain: "NEWS_MACRO",
    name: "Future Event Calendar",
    description: "Tracks scheduled macro, earnings, product, regulatory and token events before they happen.",
    dependencies: ["evolution.world-explorer"],
    limitations: ["Layer 2 target"],
    expectedStage: "DISCOVERED",
  },
  {
    capabilityId: "world.causal-scenario",
    domain: "CROSS_ASSET",
    name: "Causal & Scenario Reasoner",
    description: "Separates correlation from mechanism and builds cross-asset scenario paths.",
    dependencies: ["world.entity-graph"],
    limitations: ["Layer 2 target"],
    expectedStage: "DISCOVERED",
  },
  {
    capabilityId: "evolution.hypothesis-engine",
    domain: "RESEARCH",
    name: "Hypothesis Engine",
    description: "Turns measured weaknesses into falsifiable improvement proposals.",
    dependencies: ["evolution.capability-graph"],
    limitations: ["Layer 3 target"],
    expectedStage: "DISCOVERED",
  },
  {
    capabilityId: "evolution.self-coding",
    domain: "CODEGEN",
    name: "Self-Coding Sandbox",
    description: "Generates bounded code candidates with tests, lineage and no direct canonical apply.",
    dependencies: ["evolution.hypothesis-engine"],
    limitations: ["Layer 3 target", "protected paths are never autonomously mutable"],
    expectedStage: "DISCOVERED",
  },
  {
    capabilityId: "alpha.expected-edge",
    domain: "ALPHA",
    name: "Expected Net Edge",
    description: "Estimates gross move minus cost, uncertainty and decay before action.",
    dependencies: ["research.calibration"],
    limitations: ["Layer 4 target"],
    expectedStage: "DISCOVERED",
  },
  {
    capabilityId: "portfolio.treasury",
    domain: "PORTFOLIO",
    name: "Brian Treasury",
    description: "Allocates a unified shadow cash pool by opportunity cost and risk.",
    dependencies: ["alpha.expected-edge"],
    limitations: ["Layer 5 target", "shadow only"],
    expectedStage: "DISCOVERED",
  },
  {
    capabilityId: "portfolio.exit-brain",
    domain: "EXIT",
    name: "Exit & Capital Recycling Brain",
    description: "Manages invalidation, exits, replacements and cash recycling.",
    dependencies: ["portfolio.treasury"],
    limitations: ["Layer 5 target", "shadow only"],
    expectedStage: "DISCOVERED",
  },
];

function parseTime(value: string | null | undefined): number | null {
  if (!value) return null;
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms : null;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function newestRun(runs: CollectorRunLike[], collectorIds: string[]): CollectorRunLike | null {
  const idSet = new Set(collectorIds);
  return runs
    .filter((row) => idSet.has(String(row.collector_id)))
    .sort((a, b) => (parseTime(b.finished_at ?? b.started_at) ?? 0) - (parseTime(a.finished_at ?? a.started_at) ?? 0))[0] ?? null;
}

export function capabilityDefinitions(): CapabilityDefinition[] {
  return MAIN_CAPABILITIES.map((row) => ({ ...row, collectorIds: [...row.collectorIds], dependencies: [...row.dependencies], limitations: [...row.limitations] }));
}

export function deriveCapabilitySnapshots(
  runs: CollectorRunLike[],
  observedAt: string,
): CapabilitySnapshot[] {
  const nowMs = parseTime(observedAt) ?? Date.now();
  return MAIN_CAPABILITIES.map((definition) => {
    const run = newestRun(runs, definition.collectorIds);
    const finishedMs = parseTime(run?.finished_at ?? run?.started_at);
    const ageSeconds = finishedMs == null ? null : Math.max(0, Math.round((nowMs - finishedMs) / 1000));
    let health: CapabilityHealth = "MISSING";
    if (run) {
      if (String(run.status).toUpperCase() === "FAILED") health = "DEGRADED";
      else if (ageSeconds != null && ageSeconds > definition.staleAfterSeconds) health = "STALE";
      else if (String(run.status).toUpperCase() === "SUCCESS") health = "HEALTHY";
      else health = "DEGRADED";
    }
    const stage: EvolutionStage = definition.capabilityId.startsWith("evolution.")
      ? definition.expectedStage
      : "ACTIVE";
    const refs = run ? [`collector:${run.collector_id}:${run.finished_at ?? run.started_at ?? "unknown"}`] : [];
    return {
      capabilityId: definition.capabilityId,
      observedAt,
      domain: definition.domain,
      name: definition.name,
      version: null,
      stage,
      health,
      description: definition.description,
      sourceIds: run ? [String(run.collector_id)] : [],
      dependencies: [...definition.dependencies],
      limitations: [...definition.limitations],
      evidenceRefs: refs,
      metadata: {
        collector_ids: definition.collectorIds,
        last_run_at: run?.finished_at ?? run?.started_at ?? null,
        last_run_status: run?.status ?? null,
        age_seconds: ageSeconds,
        stale_after_seconds: definition.staleAfterSeconds,
        observed_records: run?.observed_records ?? null,
        stored_records: run?.stored_records ?? null,
        degraded_sources: run?.degraded_sources ?? [],
        error_class: run?.error_class ?? null,
      },
      evidenceClass: EVOLUTION_EVIDENCE_CLASS,
      shadowOnly: true,
      liveExecution: false,
    };
  });
}

function severityForHealth(health: CapabilityHealth): CapabilityGap["severity"] {
  if (health === "MISSING") return "CRITICAL";
  if (health === "STALE") return "HIGH";
  if (health === "DEGRADED") return "HIGH";
  if (health === "DISABLED") return "MEDIUM";
  return "LOW";
}

export function detectCapabilityGaps(snapshots: CapabilitySnapshot[]): CapabilityGap[] {
  const gaps: CapabilityGap[] = [];
  for (const snapshot of snapshots) {
    if (snapshot.health === "HEALTHY") continue;
    gaps.push({
      gapId: `runtime:${snapshot.capabilityId}:${snapshot.health}`,
      capabilityId: snapshot.capabilityId,
      domain: snapshot.domain,
      severity: severityForHealth(snapshot.health),
      reason: `${snapshot.name} is ${snapshot.health.toLowerCase()} at the latest evidence snapshot.`,
      suggestedAction: snapshot.health === "MISSING"
        ? "discover or restore a trustworthy data/compute path before relying on this capability"
        : "inspect freshness, provider health and recent collector failures before promotion",
      evidenceRefs: [...snapshot.evidenceRefs],
    });
  }
  const existing = new Set(snapshots.map((row) => row.capabilityId));
  for (const target of FUTURE_REQUIRED) {
    if (existing.has(target.capabilityId)) continue;
    gaps.push({
      gapId: `planned:${target.capabilityId}`,
      capabilityId: target.capabilityId,
      domain: target.domain,
      severity: target.capabilityId.startsWith("portfolio.") ? "HIGH" : "MEDIUM",
      reason: `${target.name} does not exist as a canonical capability yet.`,
      suggestedAction: `research and implement through Evolution lifecycle; current target stage is ${target.expectedStage}`,
      evidenceRefs: [],
    });
  }
  return gaps.sort((a, b) => {
    const rank = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 } as const;
    return rank[b.severity] - rank[a.severity] || a.capabilityId.localeCompare(b.capabilityId);
  });
}

export function inferAuthorityClass(provider: string, uri: string): WorldSourceCandidate["authorityClass"] {
  const value = `${provider} ${uri}`.toLowerCase();
  if (/(sec\.gov|federalreserve\.gov|ecb\.europa\.eu|bls\.gov|bea\.gov|eia\.gov|treasury\.gov|europa\.eu|gov\.uk|apple\.com|nvidia\.com)/.test(value)) {
    return "OFFICIAL_PRIMARY";
  }
  if (/(reuters\.com|bloomberg\.com|wsj\.com|ft\.com|apnews\.com)/.test(value)) return "INDEPENDENT_PROFESSIONAL";
  if (/(reddit\.com|x\.com|twitter\.com|t\.me|telegram)/.test(value)) return "COMMUNITY";
  return "UNKNOWN";
}

export function discoverWorldSources(events: IntelEventLike[], discoveredAt: string): WorldSourceCandidate[] {
  const seen = new Set<string>();
  const rows: WorldSourceCandidate[] = [];
  for (const event of events) {
    const uri = String(event.provenance_uri ?? "").trim();
    if (!uri) continue;
    let host = "";
    try {
      host = new URL(uri).hostname.toLowerCase().replace(/^www\./, "");
    } catch {
      continue;
    }
    if (!host || seen.has(host)) continue;
    seen.add(host);
    const provider = String(event.source_id ?? host).trim() || host;
    const authorityClass = inferAuthorityClass(provider, uri);
    const community = authorityClass === "COMMUNITY";
    rows.push({
      sourceId: `world:${host}`,
      discoveredAt,
      canonicalUri: `https://${host}/`,
      provider,
      sourceKind: String(event.source_kind ?? "DISCOVERED_WEB_SOURCE"),
      authorityClass,
      accessMode: "PUBLIC_NO_KEY",
      stage: "DISCOVERED",
      freshnessSeconds: null,
      corroborationRequired: authorityClass !== "OFFICIAL_PRIMARY",
      manipulationRisk: community ? 0.75 : authorityClass === "UNKNOWN" ? 0.5 : authorityClass === "INDEPENDENT_PROFESSIONAL" ? 0.2 : 0.08,
      rationale: `Discovered prospectively from event ${String(event.event_id ?? "unknown")} for ${String(event.asset ?? "unknown asset")}; discovery alone is not truth.`,
      metadata: {
        first_event_id: event.event_id ?? null,
        first_observed_at: event.first_observed_at ?? null,
        published_at: event.published_at ?? null,
        trust_class: event.trust_class ?? null,
        sample_claim: String(event.claim ?? "").slice(0, 280),
        discovered_from_event_stream: true,
      },
    });
  }
  return rows;
}

export function assessWorldSource(source: WorldSourceCandidate, nowIso: string): SourceAssessment {
  const authorityScore = source.authorityClass === "OFFICIAL_PRIMARY"
    ? 1
    : source.authorityClass === "INDEPENDENT_PROFESSIONAL"
    ? 0.82
    : source.authorityClass === "COMMUNITY"
    ? 0.35
    : 0.45;
  const freshnessScore = source.freshnessSeconds == null
    ? 0.5
    : clamp01(1 - source.freshnessSeconds / 86400);
  const manipulationPenalty = clamp01(source.manipulationRisk) * 0.4;
  const corroborationPenalty = source.corroborationRequired ? 0.12 : 0;
  const accessPenalty = source.accessMode === "UNAVAILABLE"
    ? 0.5
    : source.accessMode === "LICENSED_REQUIRED"
    ? 0.18
    : source.accessMode === "API_KEY_REQUIRED"
    ? 0.08
    : 0;
  const trustScore = clamp01(
    authorityScore * 0.62 + freshnessScore * 0.28 + 0.10 - manipulationPenalty - corroborationPenalty - accessPenalty,
  );
  const eligibleForResearch = source.accessMode !== "UNAVAILABLE" && trustScore >= 0.35;
  const eligibleForDecisionEvidence = source.stage === "ACTIVE" && trustScore >= 0.72 && !source.corroborationRequired;
  const reasons: string[] = [
    `authority=${source.authorityClass}:${authorityScore.toFixed(2)}`,
    `freshness=${source.freshnessSeconds == null ? "unknown" : `${source.freshnessSeconds}s`}`,
    `manipulation_risk=${source.manipulationRisk.toFixed(2)}`,
    `stage=${source.stage}`,
    `assessed_at=${nowIso}`,
  ];
  if (source.corroborationRequired) reasons.push("independent corroboration required before directional use");
  if (!eligibleForDecisionEvidence) reasons.push("not eligible to influence ALPHA decisions");
  return {
    authorityScore,
    freshnessScore,
    manipulationPenalty,
    corroborationPenalty,
    accessPenalty,
    trustScore,
    eligibleForResearch,
    eligibleForDecisionEvidence,
    reasons,
  };
}

export function buildEvolutionDashboardModel(
  snapshots: CapabilitySnapshot[],
  gaps: CapabilityGap[],
  sources: WorldSourceCandidate[],
  latestJournalAt: string | null,
  observedAt: string,
): EvolutionDashboardModel {
  const capabilityCounts: Record<CapabilityHealth, number> = {
    HEALTHY: 0,
    DEGRADED: 0,
    STALE: 0,
    MISSING: 0,
    DISABLED: 0,
  };
  const stageCounts: Partial<Record<EvolutionStage, number>> = {};
  for (const snapshot of snapshots) {
    capabilityCounts[snapshot.health] += 1;
    stageCounts[snapshot.stage] = (stageCounts[snapshot.stage] ?? 0) + 1;
  }
  let researchEligible = 0;
  let decisionEligible = 0;
  for (const source of sources) {
    const assessment = assessWorldSource(source, observedAt);
    if (assessment.eligibleForResearch) researchEligible += 1;
    if (assessment.eligibleForDecisionEvidence) decisionEligible += 1;
  }
  const criticalGaps = gaps.filter((gap) => gap.severity === "CRITICAL");
  const overall: EvolutionDashboardModel["overall"] = criticalGaps.length
    ? "DEGRADED"
    : capabilityCounts.MISSING || capabilityCounts.STALE || capabilityCounts.DEGRADED
    ? "BUILDING"
    : "HEALTHY";
  return {
    schemaVersion: EVOLUTION_CORE_VERSION,
    observedAt,
    overall,
    capabilityCounts,
    stageCounts,
    criticalGaps,
    topGaps: gaps.slice(0, 12),
    sourceCounts: {
      total: sources.length,
      official: sources.filter((row) => row.authorityClass === "OFFICIAL_PRIMARY").length,
      verifying: sources.filter((row) => row.stage === "VERIFYING" || row.stage === "DISCOVERED").length,
      rejected: sources.filter((row) => row.stage === "REJECTED").length,
      researchEligible,
      decisionEligible,
    },
    latestJournalAt,
    cloudIndependent: true,
    shadowOnly: true,
    liveExecution: false,
  };
}
