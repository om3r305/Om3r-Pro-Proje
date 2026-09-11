import {
  assertAutonomousChangeSetAllowed,
  EVOLUTION_EVIDENCE_CLASS,
} from "./evolution_contract.ts";
import {
  buildCodeCandidatePlan,
  type HypothesisCandidate,
} from "./evolution_research.ts";

export const EVOLUTION_SANDBOX_VERSION = "brian.evolution-sandbox.v1";
export const MAX_SANDBOX_CHANGED_FILES = 10;
export const MAX_SANDBOX_PATCH_BYTES = 250_000;

const ALLOWED_PATH_PREFIXES = [
  "supabase/functions/_shared/evolution_candidates/",
  "supabase/functions/brian-evolution-candidate-",
  "tests/evolution_candidates/",
  "docs/evolution_candidates/",
] as const;

export type SandboxEvidenceKind = "GENERATED" | "TYPECHECK" | "UNIT" | "REPLAY" | "STRESS" | "PROSPECTIVE";
export type SandboxReviewVerdict = "BLOCK" | "CONTINUE_TESTING" | "READY_FOR_HUMAN_REVIEW";

export interface SandboxGenerationBrief {
  candidateId: string;
  hypothesisId: string;
  proposedAt: string;
  parentCommit: string;
  branchName: string;
  changedPaths: string[];
  testPlan: string[];
  objective: string;
  constraints: string[];
  successCriteria: string[];
  evidenceRefs: string[];
  contaminationDeclaration: string;
  stage: "EXPERIMENTAL";
  autonomousApplyAllowed: false;
  requiredHumanReview: true;
  evidenceClass: typeof EVOLUTION_EVIDENCE_CLASS;
  shadowOnly: true;
  liveExecution: false;
  metadata: Record<string, unknown>;
}

export interface SandboxArtifactSubmission {
  candidateId: string;
  hypothesisId: string;
  parentCommit: string;
  branchName: string;
  changedPaths: string[];
  patchSha256: string;
  patchBytes: number;
  generatedBy: string;
  generatorRunId: string;
  generatedAt: string;
  metadata?: Record<string, unknown>;
}

export interface SandboxValidationResult {
  valid: boolean;
  reasons: string[];
  normalizedPaths: string[];
}

export interface SandboxReviewState {
  protectedScopeClear: boolean;
  artifactHashVerified: boolean;
  provenanceComplete: boolean;
  typecheckGreen: boolean;
  unitGreen: boolean;
  replayGreen: boolean;
  stressGreen: boolean;
  prospectiveGreen: boolean;
  leakageClear: boolean;
}

export interface SandboxReviewDecision {
  verdict: SandboxReviewVerdict;
  reasons: string[];
  autonomousApplyAllowed: false;
  requiredHumanReview: true;
  canonicalMutation: false;
}

function slug(value: string, max = 48): string {
  const cleaned = String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, max)
    .replace(/-+$/g, "");
  return cleaned || "candidate";
}

function normalizePath(path: string): string {
  return String(path ?? "").trim().replaceAll("\\", "/").replace(/^\.\//, "");
}

function isAllowedSandboxPath(path: string): boolean {
  return ALLOWED_PATH_PREFIXES.some((prefix) => path.startsWith(prefix));
}

function candidateStem(h: HypothesisCandidate): string {
  const capability = h.targetCapabilities[0] ?? h.hypothesisKind.toLowerCase();
  return slug(`${h.hypothesisKind}-${capability}`, 56);
}

export function suggestedSandboxPaths(h: HypothesisCandidate): string[] {
  const stem = candidateStem(h);
  return [
    `supabase/functions/_shared/evolution_candidates/${stem}.ts`,
    `supabase/functions/_shared/evolution_candidates/${stem}.test.ts`,
    `docs/evolution_candidates/${stem}.md`,
  ];
}

export function buildSandboxGenerationBrief(
  h: HypothesisCandidate,
  parentCommit: string,
  changedPaths = suggestedSandboxPaths(h),
): SandboxGenerationBrief {
  const plan = buildCodeCandidatePlan(h, parentCommit, changedPaths);
  const branchName = `evolution-candidate/${slug(h.hypothesisKind, 20)}-${slug(plan.candidateId, 34)}`;
  return {
    candidateId: plan.candidateId,
    hypothesisId: h.hypothesisId,
    proposedAt: h.observedAt,
    parentCommit: plan.parentCommit,
    branchName,
    changedPaths: [...plan.changedPaths],
    testPlan: [...plan.testPlan],
    objective: h.proposedMechanism,
    constraints: [
      "Implement only the bounded challenger/research artifact described by this hypothesis.",
      "Do not modify canonical ALPHA behavior, live execution, secrets, auth, CI, migrations, or protected scopes.",
      "Use only point-in-time evidence available at decision time; future outcomes may be used only for evaluation.",
      "Keep replay, stress, and prospective evidence explicitly separated.",
      "Do not read, mutate, schedule, rebalance, or control DIP resources.",
      "Artifact must remain SHADOW ONLY and must not create authenticated exchange-order or withdrawal surfaces.",
    ],
    successCriteria: [...h.measurableSuccessCriteria],
    evidenceRefs: [...h.evidenceRefs],
    contaminationDeclaration: plan.contaminationDeclaration,
    stage: "EXPERIMENTAL",
    autonomousApplyAllowed: false,
    requiredHumanReview: true,
    evidenceClass: EVOLUTION_EVIDENCE_CLASS,
    shadowOnly: true,
    liveExecution: false,
    metadata: {
      ...plan.metadata,
      sandbox_version: EVOLUTION_SANDBOX_VERSION,
      hypothesis_kind: h.hypothesisKind,
      uncertainty: h.uncertainty,
      priority: h.priority,
      target_capabilities: h.targetCapabilities,
      external_generator_required: true,
      branch_isolation_required: true,
    },
  };
}

export function validateSandboxArtifact(submission: SandboxArtifactSubmission): SandboxValidationResult {
  const reasons: string[] = [];
  const normalizedPaths = [...new Set((submission.changedPaths ?? []).map(normalizePath).filter(Boolean))];

  if (!submission.candidateId.trim()) reasons.push("candidate_id missing");
  if (!submission.hypothesisId.trim()) reasons.push("hypothesis_id missing");
  if (!/^[0-9a-f]{7,64}$/i.test(submission.parentCommit.trim())) reasons.push("parent commit is not a git SHA");
  if (!submission.branchName.startsWith("evolution-candidate/")) reasons.push("candidate branch must use evolution-candidate/ prefix");
  if (!/^[0-9a-f]{64}$/i.test(submission.patchSha256.trim())) reasons.push("patch_sha256 must be 64 hex chars");
  if (!Number.isInteger(submission.patchBytes) || submission.patchBytes <= 0) reasons.push("patch_bytes must be a positive integer");
  if (submission.patchBytes > MAX_SANDBOX_PATCH_BYTES) reasons.push(`patch exceeds ${MAX_SANDBOX_PATCH_BYTES} bytes`);
  if (!submission.generatedBy.trim() || !submission.generatorRunId.trim()) reasons.push("generator provenance incomplete");
  if (!Number.isFinite(Date.parse(submission.generatedAt))) reasons.push("generated_at is invalid");
  if (!normalizedPaths.length) reasons.push("candidate must change at least one file");
  if (normalizedPaths.length > MAX_SANDBOX_CHANGED_FILES) reasons.push(`candidate changes more than ${MAX_SANDBOX_CHANGED_FILES} files`);

  for (const path of normalizedPaths) {
    if (path.startsWith("/") || path.includes("../") || path === "..") reasons.push(`unsafe path:${path}`);
    if (!isAllowedSandboxPath(path)) reasons.push(`outside sandbox allowlist:${path}`);
  }

  try {
    assertAutonomousChangeSetAllowed(normalizedPaths);
  } catch (error) {
    reasons.push(String(error));
  }

  return { valid: reasons.length === 0, reasons, normalizedPaths };
}

export function evaluateSandboxReview(state: SandboxReviewState): SandboxReviewDecision {
  const hardBlocks: string[] = [];
  if (!state.protectedScopeClear) hardBlocks.push("protected-scope validation failed");
  if (!state.artifactHashVerified) hardBlocks.push("artifact hash is not verified");
  if (!state.provenanceComplete) hardBlocks.push("artifact provenance is incomplete");
  if (!state.leakageClear) hardBlocks.push("point-in-time leakage or contamination is unresolved");
  if (hardBlocks.length) {
    return {
      verdict: "BLOCK",
      reasons: hardBlocks,
      autonomousApplyAllowed: false,
      requiredHumanReview: true,
      canonicalMutation: false,
    };
  }

  const pending: string[] = [];
  if (!state.typecheckGreen) pending.push("type-check evidence missing or failing");
  if (!state.unitGreen) pending.push("unit/behavioral tests missing or failing");
  if (!state.replayGreen) pending.push("replay evidence missing or failing");
  if (!state.stressGreen) pending.push("stress/adversarial evidence missing or failing");
  if (!state.prospectiveGreen) pending.push("prospective shadow evidence missing or failing");
  if (pending.length) {
    return {
      verdict: "CONTINUE_TESTING",
      reasons: pending,
      autonomousApplyAllowed: false,
      requiredHumanReview: true,
      canonicalMutation: false,
    };
  }

  return {
    verdict: "READY_FOR_HUMAN_REVIEW",
    reasons: ["all sandbox evidence gates passed; human review is still mandatory before any canonical integration"],
    autonomousApplyAllowed: false,
    requiredHumanReview: true,
    canonicalMutation: false,
  };
}
