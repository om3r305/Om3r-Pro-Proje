import { validateSandboxArtifact } from "./evolution_sandbox.ts";

function submission(changedPaths: string[]) {
  return {
    candidateId: "candidate-engineer-evidence",
    hypothesisId: "hypothesis|engineer-evidence|1",
    parentCommit: "ba330f4ec4bd3d76b1fe5564d6b4e95cd41ef4f0",
    branchName: "evolution-candidate/engineer-evidence",
    changedPaths,
    patchSha256: "a".repeat(64),
    patchBytes: 12000,
    generatedBy: "brian-engineer",
    generatorRunId: "run-engineer-evidence",
    generatedAt: "2026-09-13T12:00:00Z",
  };
}

Deno.test("sandbox accepts only isolated Brian Engineer replay and stress evidence paths", () => {
  const accepted = validateSandboxArtifact(submission([
    "supabase/functions/_shared/evolution_candidates/example.ts",
    "tests/evolution_engineer/replay/example.replay.test.ts",
    "tests/evolution_engineer/stress/example.stress.test.ts",
    "docs/evolution_candidates/example.md",
  ]));
  if (!accepted.valid) throw new Error(JSON.stringify(accepted));

  const rejected = validateSandboxArtifact(submission([
    "tests/evolution_engineer/live/example.test.ts",
  ]));
  if (rejected.valid) throw new Error("non-evidence engineer path must remain outside sandbox allowlist");
});
