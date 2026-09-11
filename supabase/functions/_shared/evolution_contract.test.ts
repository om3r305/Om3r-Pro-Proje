import {
  assertAutonomousChangeSetAllowed,
  assertEvolutionStageTransition,
  canTransitionEvolutionStage,
  classifyEvolutionChangeSet,
  finiteProbability,
  protectedPathMatch,
} from "./evolution_contract.ts";

Deno.test("Evolution autonomous changes reject DIP runtime paths", () => {
  const paths = [
    "supabase/functions/brian-dip-v84-authority-worker/index.ts",
    "supabase/functions/brian_dip_shadow_worker/index.ts",
    "monster-coins-pro/dip-expert-v4-engine.js",
    "supabase/migrations/202609010001_brian_dip_runtime.sql",
    "docs/BRIAN_DIP_TARGET_COST_REVIEW.md",
  ];
  for (const path of paths) {
    const match = protectedPathMatch(path);
    if (!match || match.scope !== "DIP") throw new Error(`expected DIP protection for ${path}`);
  }
});

Deno.test("Evolution allows normal MAIN/ALPHA research paths", () => {
  const paths = [
    "supabase/functions/brian-world-explorer/index.ts",
    "supabase/functions/_shared/evolution_contract.ts",
    "brian2026/evolution/hypothesis.py",
    "monster-coins-pro/evolution.js",
  ];
  const result = classifyEvolutionChangeSet(paths);
  if (result.protected.length !== 0) throw new Error(JSON.stringify(result.protected));
  if (result.allowed.length !== paths.length) throw new Error("allowed path count mismatch");
  assertAutonomousChangeSetAllowed(paths);
});

Deno.test("Evolution self-coding cannot alter CI or auth boundaries", () => {
  const result = classifyEvolutionChangeSet([
    ".github/workflows/brian-ci.yml",
    "supabase/functions/_shared/cron_auth.ts",
    ".env.production",
  ]);
  const scopes = new Set(result.protected.map((x) => x.scope));
  if (!scopes.has("CI_SECURITY")) throw new Error("CI protection missing");
  if (!scopes.has("AUTH_SECRETS")) throw new Error("auth/secret protection missing");
});

Deno.test("Evolution generated migrations require human-reviewed promotion", () => {
  const match = protectedPathMatch("supabase/migrations/202609111300_brian_evolution.sql");
  if (!match || match.scope !== "DATABASE_CONTROL_PLANE") throw new Error("migration should be protected");
});

Deno.test("Evolution lifecycle cannot skip scientific stages", () => {
  if (!canTransitionEvolutionStage("DISCOVERED", "VERIFYING")) throw new Error("valid transition rejected");
  if (!canTransitionEvolutionStage("EXPERIMENTAL", "SHADOW_CANDIDATE")) throw new Error("valid transition rejected");
  if (!canTransitionEvolutionStage("DECAYING", "ACTIVE")) throw new Error("recovery transition rejected");
  if (canTransitionEvolutionStage("DISCOVERED", "ACTIVE")) throw new Error("stage skip accepted");
  if (canTransitionEvolutionStage("RESEARCHING", "ACTIVE")) throw new Error("research->active skip accepted");
  if (canTransitionEvolutionStage("ARCHIVED", "ACTIVE")) throw new Error("archive resurrection accepted");
});

Deno.test("Evolution invalid stage transition fails closed", () => {
  let threw = false;
  try {
    assertEvolutionStageTransition("VERIFYING", "ACTIVE");
  } catch (error) {
    threw = String(error).includes("INVALID_EVOLUTION_TRANSITION");
  }
  if (!threw) throw new Error("invalid transition did not fail closed");
});

Deno.test("Evolution probability inputs are bounded", () => {
  if (finiteProbability(0.75) !== 0.75) throw new Error("valid probability changed");
  for (const value of [-0.01, 1.01, Number.NaN, Number.POSITIVE_INFINITY]) {
    let threw = false;
    try {
      finiteProbability(value);
    } catch {
      threw = true;
    }
    if (!threw) throw new Error(`invalid probability accepted: ${value}`);
  }
});
