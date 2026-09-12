// Brian Evolution OS foundational contracts.
// Pure/import-safe: no I/O, no credentials, no network, no deployment surface.

export const EVOLUTION_SCHEMA_VERSION = "brian.evolution.v1";
export const EVOLUTION_EVIDENCE_CLASS = "PROSPECTIVE_EVOLUTION_SHADOW" as const;
export const EVOLUTION_SHADOW_ONLY = true as const;
export const EVOLUTION_LIVE_EXECUTION = false as const;

export type EvolutionStage =
  | "DISCOVERED"
  | "VERIFYING"
  | "RESEARCHING"
  | "EXPERIMENTAL"
  | "SHADOW_CANDIDATE"
  | "ACTIVE"
  | "DECAYING"
  | "REJECTED"
  | "RETIRED"
  | "ARCHIVED";

export type CapabilityDomain =
  | "WORLD_SOURCE"
  | "MARKET_DATA"
  | "NEWS_MACRO"
  | "ENTITY_GRAPH"
  | "CROSS_ASSET"
  | "SENSOR"
  | "MODEL"
  | "ALPHA"
  | "RESEARCH"
  | "CODEGEN"
  | "EXPERIMENT"
  | "PORTFOLIO"
  | "EXIT"
  | "OBSERVABILITY";

export type CapabilityHealth = "HEALTHY" | "DEGRADED" | "STALE" | "MISSING" | "DISABLED";

export interface CapabilitySnapshot {
  capabilityId: string;
  observedAt: string;
  domain: CapabilityDomain;
  name: string;
  version: string | null;
  stage: EvolutionStage;
  health: CapabilityHealth;
  description: string;
  sourceIds: string[];
  dependencies: string[];
  limitations: string[];
  evidenceRefs: string[];
  metadata: Record<string, unknown>;
  evidenceClass: typeof EVOLUTION_EVIDENCE_CLASS;
  shadowOnly: true;
  liveExecution: false;
}

export interface WorldSourceCandidate {
  sourceId: string;
  discoveredAt: string;
  canonicalUri: string;
  provider: string;
  sourceKind: string;
  authorityClass: "OFFICIAL_PRIMARY" | "INDEPENDENT_PROFESSIONAL" | "COMMUNITY" | "UNKNOWN";
  accessMode: "PUBLIC_NO_KEY" | "API_KEY_REQUIRED" | "LICENSED_REQUIRED" | "UNAVAILABLE";
  stage: EvolutionStage;
  freshnessSeconds: number | null;
  corroborationRequired: boolean;
  manipulationRisk: number;
  rationale: string;
  metadata: Record<string, unknown>;
}

export interface EvolutionHypothesis {
  hypothesisId: string;
  createdAt: string;
  title: string;
  problemStatement: string;
  proposedMechanism: string;
  targetCapabilities: string[];
  evidenceRefs: string[];
  counterEvidenceRefs: string[];
  measurableSuccessCriteria: string[];
  stage: EvolutionStage;
  uncertainty: number;
}

export interface EvolutionCodeCandidate {
  candidateId: string;
  hypothesisId: string;
  createdAt: string;
  parentCommit: string;
  changedPaths: string[];
  testPlan: string[];
  contaminationDeclaration: string;
  stage: EvolutionStage;
  shadowOnly: true;
  liveExecution: false;
}

export type ProtectedScope = "DIP" | "CI_SECURITY" | "AUTH_SECRETS" | "DATABASE_CONTROL_PLANE";

export interface ProtectedPathMatch {
  path: string;
  scope: ProtectedScope;
  reason: string;
}

const DIP_PATTERNS = [
  /(^|\/)dip([/_-]|$)/i,
  /(^|\/)brian[-_]dip([/_-]|$)/i,
  /(^|\/)brian_dip([/_-]|$)/i,
  /(^|\/)monster-coins-pro\/dip[^/]*$/i,
  /brian[-_]dip/i,
];

const CI_SECURITY_PATTERNS = [
  /^\.github\//i,
  /(^|\/)CODEOWNERS$/i,
];

const AUTH_SECRET_PATTERNS = [
  /(^|\/)\.env(?:\.|$)/i,
  /(^|\/)(secrets?|credentials?)([._/-]|$)/i,
  /(^|\/)cron_auth\.ts$/i,
  /(^|\/)dashboard_auth([._/-]|$)/i,
];

const DB_CONTROL_PATTERNS = [
  /^supabase\/migrations\//i,
];

function normalizePath(path: string): string {
  return String(path ?? "").trim().replaceAll("\\", "/").replace(/^\.\//, "");
}

function firstMatch(path: string, patterns: RegExp[]): boolean {
  return patterns.some((pattern) => pattern.test(path));
}

export function protectedPathMatch(inputPath: string): ProtectedPathMatch | null {
  const path = normalizePath(inputPath);
  if (!path) return { path, scope: "CI_SECURITY", reason: "empty/invalid change path is not eligible for autonomous application" };
  if (firstMatch(path, DIP_PATTERNS)) {
    return { path, scope: "DIP", reason: "DIP is a separate experiment and treasury; Evolution OS may not mutate it" };
  }
  if (firstMatch(path, CI_SECURITY_PATTERNS)) {
    return { path, scope: "CI_SECURITY", reason: "self-coding may not alter its own CI/review enforcement" };
  }
  if (firstMatch(path, AUTH_SECRET_PATTERNS)) {
    return { path, scope: "AUTH_SECRETS", reason: "self-coding may not alter credential/authentication boundaries" };
  }
  if (firstMatch(path, DB_CONTROL_PATTERNS)) {
    return { path, scope: "DATABASE_CONTROL_PLANE", reason: "schema/control-plane changes require explicit human-reviewed promotion" };
  }
  return null;
}

export function classifyEvolutionChangeSet(paths: string[]): {
  allowed: string[];
  protected: ProtectedPathMatch[];
} {
  const allowed: string[] = [];
  const protectedRows: ProtectedPathMatch[] = [];
  for (const raw of paths) {
    const normalized = normalizePath(raw);
    const match = protectedPathMatch(normalized);
    if (match) protectedRows.push(match);
    else allowed.push(normalized);
  }
  return { allowed, protected: protectedRows };
}

export function assertAutonomousChangeSetAllowed(paths: string[]): void {
  const result = classifyEvolutionChangeSet(paths);
  if (result.protected.length) {
    const detail = result.protected.map((x) => `${x.scope}:${x.path}`).join(", ");
    throw new Error(`EVOLUTION_PROTECTED_SCOPE:${detail}`);
  }
}

const ALLOWED_TRANSITIONS: Record<EvolutionStage, ReadonlySet<EvolutionStage>> = {
  DISCOVERED: new Set(["VERIFYING", "REJECTED"]),
  VERIFYING: new Set(["RESEARCHING", "REJECTED"]),
  RESEARCHING: new Set(["EXPERIMENTAL", "REJECTED"]),
  EXPERIMENTAL: new Set(["SHADOW_CANDIDATE", "REJECTED"]),
  SHADOW_CANDIDATE: new Set(["ACTIVE", "REJECTED"]),
  ACTIVE: new Set(["DECAYING", "RETIRED"]),
  DECAYING: new Set(["ACTIVE", "RETIRED"]),
  REJECTED: new Set(["ARCHIVED"]),
  RETIRED: new Set(["ARCHIVED"]),
  ARCHIVED: new Set(),
};

export function canTransitionEvolutionStage(from: EvolutionStage, to: EvolutionStage): boolean {
  return from === to || ALLOWED_TRANSITIONS[from].has(to);
}

export function assertEvolutionStageTransition(from: EvolutionStage, to: EvolutionStage): void {
  if (!canTransitionEvolutionStage(from, to)) {
    throw new Error(`INVALID_EVOLUTION_TRANSITION:${from}->${to}`);
  }
}

export function finiteProbability(value: number, label = "probability"): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error(`${label} must be in [0,1]`);
  return value;
}
