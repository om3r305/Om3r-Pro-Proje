import {
  buildSandboxGenerationBrief,
  evaluateSandboxReview,
  MAX_SANDBOX_PATCH_BYTES,
  suggestedSandboxPaths,
  validateSandboxArtifact,
} from "./evolution_sandbox.ts";
import type { HypothesisCandidate } from "./evolution_research.ts";

function hypothesis(): HypothesisCandidate {
  return {
    hypothesisId: "hypothesis|expected-edge|900|2026-09-11t13",
    observedAt: "2026-09-11T13:00:00Z",
    problemStatement: "Canonical actions are negative after cost.",
    proposedMechanism: "Build an isolated expected-net-edge challenger.",
    targetCapabilities: ["alpha.expected-edge", "alpha.compiler"],
    evidenceRefs: ["outcomes:900"],
    counterEvidenceRefs: [],
    measurableSuccessCriteria: ["positive prospective after-cost edge"],
    stage: "RESEARCHING",
    uncertainty: 0.25,
    priority: 0.98,
    hypothesisKind: "EXPECTED_EDGE",
    metadata: {},
  };
}

function validSubmission() {
  const h = hypothesis();
  const brief = buildSandboxGenerationBrief(h, "ba330f4ec4bd3d76b1fe5564d6b4e95cd41ef4f0");
  return {
    candidateId: brief.candidateId,
    hypothesisId: brief.hypothesisId,
    parentCommit: brief.parentCommit,
    branchName: brief.branchName,
    changedPaths: brief.changedPaths,
    patchSha256: "a".repeat(64),
    patchBytes: 12000,
    generatedBy: "external-code-generator",
    generatorRunId: "run-001",
    generatedAt: "2026-09-11T13:05:00Z",
  };
}

Deno.test("sandbox generation brief is isolated and never auto-applies", () => {
  const h = hypothesis();
  const paths = suggestedSandboxPaths(h);
  if (!paths.every((path) => path.includes("evolution_candidates") || path.startsWith("docs/evolution_candidates/"))) {
    throw new Error(JSON.stringify(paths));
  }
  const brief = buildSandboxGenerationBrief(h, "ba330f4ec4bd3d76b1fe5564d6b4e95cd41ef4f0", paths);
  if (!brief.branchName.startsWith("evolution-candidate/")) throw new Error("candidate branch is not isolated");
  if (brief.autonomousApplyAllowed) throw new Error("autonomous apply enabled");
  if (!brief.requiredHumanReview) throw new Error("human review must remain mandatory");
  if (!brief.constraints.some((x) => x.includes("canonical ALPHA"))) throw new Error("canonical fence missing");
});

Deno.test("sandbox accepts a bounded provenance-complete artifact manifest", () => {
  const result = validateSandboxArtifact(validSubmission());
  if (!result.valid) throw new Error(JSON.stringify(result));
});

Deno.test("sandbox rejects protected and non-allowlisted paths", () => {
  for (const path of [
    "supabase/functions/brian-dip-v84-authority-worker/index.ts",
    ".github/workflows/brian-evolution-ci.yml",
    "supabase/functions/brian-control-center/index.ts",
    "supabase/migrations/202609119999_bad.sql",
  ]) {
    const result = validateSandboxArtifact({ ...validSubmission(), changedPaths: [path] });
    if (result.valid) throw new Error(`unsafe path accepted: ${path}`);
  }
});

Deno.test("sandbox rejects oversized or unverifiable patches", () => {
  const oversized = validateSandboxArtifact({ ...validSubmission(), patchBytes: MAX_SANDBOX_PATCH_BYTES + 1 });
  if (oversized.valid) throw new Error("oversized patch accepted");
  const badHash = validateSandboxArtifact({ ...validSubmission(), patchSha256: "not-a-hash" });
  if (badHash.valid) throw new Error("invalid hash accepted");
});

Deno.test("sandbox review blocks contamination before considering test status", () => {
  const review = evaluateSandboxReview({
    protectedScopeClear: true,
    artifactHashVerified: true,
    provenanceComplete: true,
    typecheckGreen: true,
    unitGreen: true,
    replayGreen: true,
    stressGreen: true,
    prospectiveGreen: true,
    leakageClear: false,
  });
  if (review.verdict !== "BLOCK") throw new Error(JSON.stringify(review));
});

Deno.test("sandbox review requires the complete evidence ladder", () => {
  const review = evaluateSandboxReview({
    protectedScopeClear: true,
    artifactHashVerified: true,
    provenanceComplete: true,
    typecheckGreen: true,
    unitGreen: true,
    replayGreen: true,
    stressGreen: false,
    prospectiveGreen: false,
    leakageClear: true,
  });
  if (review.verdict !== "CONTINUE_TESTING") throw new Error(JSON.stringify(review));
});

Deno.test("sandbox can only become ready for human review, never canonical auto-apply", () => {
  const review = evaluateSandboxReview({
    protectedScopeClear: true,
    artifactHashVerified: true,
    provenanceComplete: true,
    typecheckGreen: true,
    unitGreen: true,
    replayGreen: true,
    stressGreen: true,
    prospectiveGreen: true,
    leakageClear: true,
  });
  if (review.verdict !== "READY_FOR_HUMAN_REVIEW") throw new Error(JSON.stringify(review));
  if (review.autonomousApplyAllowed || !review.requiredHumanReview || review.canonicalMutation) {
    throw new Error("unsafe final review state");
  }
});
