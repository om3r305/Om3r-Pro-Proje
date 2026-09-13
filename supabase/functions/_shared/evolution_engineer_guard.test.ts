import { assertEquals } from "jsr:@std/assert@1";
import {
  isDipProtectedPath,
  validateEngineerChangeSet,
} from "./evolution_engineer_guard.ts";

Deno.test("engineer guard blocks every protected DIP path form", () => {
  for (const path of [
    "tests/replay/dip_case.test.ts",
    "supabase/functions/brian-dip-worker/index.ts",
    "supabase/migrations/anything_brian_dip.sql",
    "foo/dip/bar.ts",
  ]) assertEquals(isDipProtectedPath(path), true, path);
});

Deno.test("engineer guard allows ordinary Brian source and tests", () => {
  const result = validateEngineerChangeSet({
    changedPaths: [
      "supabase/functions/_shared/evolution_research.ts",
      "supabase/functions/_shared/evolution_research.test.ts",
    ],
    patchBytes: 4096,
    addedText: "export const value = 1;",
  });
  assertEquals(result.valid, true);
  assertEquals(result.reasons, []);
});

Deno.test("engineer guard blocks its own control plane", () => {
  const result = validateEngineerChangeSet({
    changedPaths: [".github/workflows/brian-engineer.yml"],
    patchBytes: 100,
  });
  assertEquals(result.valid, false);
});

Deno.test("manual infrastructure review may opt into control-plane files but never DIP", () => {
  const infra = validateEngineerChangeSet({
    changedPaths: [".github/workflows/brian-engineer.yml"],
    patchBytes: 100,
    allowControlPlane: true,
  });
  assertEquals(infra.valid, true);
  const protectedResult = validateEngineerChangeSet({
    changedPaths: ["tests/replay/dip_v83_level_lifecycle.test.ts"],
    patchBytes: 100,
    allowControlPlane: true,
  });
  assertEquals(protectedResult.valid, false);
});

Deno.test("engineer guard blocks live execution surface", () => {
  const result = validateEngineerChangeSet({
    changedPaths: ["supabase/functions/_shared/example.ts"],
    patchBytes: 100,
    addedText: "const live_execution = true;",
  });
  assertEquals(result.valid, false);
});
