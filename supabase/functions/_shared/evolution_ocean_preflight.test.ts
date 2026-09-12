import { assessOceanPreflightFreshness } from "./evolution_ocean_preflight.ts";

Deno.test("Ocean preflight accepts fresh successful runtime evidence", () => {
  const result = assessOceanPreflightFreshness("2026-09-11T14:00:00Z", [
    { label: "Treasury snapshot", observedAt: "2026-09-11T13:58:30Z", maxAgeSeconds: 240 },
    { label: "Layer-4 edge", observedAt: "2026-09-11T13:57:30Z", maxAgeSeconds: 360 },
    { label: "Treasury worker", observedAt: "2026-09-11T13:59:10Z", maxAgeSeconds: 240, status: "SUCCESS", requireSuccess: true },
  ]);
  if (!result.ready || result.reasons.length) throw new Error(JSON.stringify(result));
});

Deno.test("Ocean preflight rejects stale evidence even when last status was SUCCESS", () => {
  const result = assessOceanPreflightFreshness("2026-09-11T14:00:00Z", [
    { label: "Researcher", observedAt: "2026-09-11T12:00:00Z", maxAgeSeconds: 4500, status: "SUCCESS", requireSuccess: true },
  ]);
  if (result.ready || !result.reasons.some((reason) => reason.includes("stale"))) throw new Error(JSON.stringify(result));
});

Deno.test("Ocean preflight rejects failed, missing and future-dated probes", () => {
  const result = assessOceanPreflightFreshness("2026-09-11T14:00:00Z", [
    { label: "Orchestrator", observedAt: "2026-09-11T13:59:00Z", maxAgeSeconds: 1200, status: "FAILED", requireSuccess: true },
    { label: "Treasury snapshot", observedAt: null, maxAgeSeconds: 240 },
    { label: "Layer-4 edge", observedAt: "2026-09-11T14:00:30Z", maxAgeSeconds: 360 },
  ]);
  if (result.ready) throw new Error(JSON.stringify(result));
  if (!result.reasons.some((reason) => reason.includes("latest status is FAILED"))) throw new Error(JSON.stringify(result));
  if (!result.reasons.some((reason) => reason.includes("no valid timestamp"))) throw new Error(JSON.stringify(result));
  if (!result.reasons.some((reason) => reason.includes("in the future"))) throw new Error(JSON.stringify(result));
});
