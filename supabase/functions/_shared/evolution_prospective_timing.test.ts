import { assessProspectiveTiming } from "./evolution_prospective_timing.ts";

Deno.test("prospective timing accepts labels created after experiment start and before outcome", () => {
  const result = assessProspectiveTiming({
    experimentStartedAt: "2026-09-11T10:00:00Z",
    decisionObservedAt: "2026-09-11T10:10:00Z",
    labelEvaluatedAt: "2026-09-11T10:11:00Z",
    outcomeResolvedAt: "2026-09-11T10:25:00Z",
    measuredAt: "2026-09-11T11:00:00Z",
    maxLabelLatencySeconds: 120,
  });
  if (!result.clean || result.reasons.length) throw new Error(JSON.stringify(result));
});

Deno.test("prospective timing rejects pre-experiment evidence", () => {
  const result = assessProspectiveTiming({
    experimentStartedAt: "2026-09-11T10:00:00Z",
    decisionObservedAt: "2026-09-11T09:59:00Z",
    labelEvaluatedAt: "2026-09-11T09:59:30Z",
    outcomeResolvedAt: "2026-09-11T10:14:00Z",
    measuredAt: "2026-09-11T11:00:00Z",
  });
  if (result.clean || !result.reasons.some((reason) => reason.includes("predates experiment"))) throw new Error(JSON.stringify(result));
});

Deno.test("prospective timing rejects challenger labels produced after outcome resolution", () => {
  const result = assessProspectiveTiming({
    experimentStartedAt: "2026-09-11T10:00:00Z",
    decisionObservedAt: "2026-09-11T10:10:00Z",
    labelEvaluatedAt: "2026-09-11T10:30:00Z",
    outcomeResolvedAt: "2026-09-11T10:25:00Z",
    measuredAt: "2026-09-11T11:00:00Z",
  });
  if (result.clean || !result.reasons.some((reason) => reason.includes("after outcome"))) throw new Error(JSON.stringify(result));
});

Deno.test("prospective timing rejects expected-edge labels beyond runtime latency budget", () => {
  const result = assessProspectiveTiming({
    experimentStartedAt: "2026-09-11T10:00:00Z",
    decisionObservedAt: "2026-09-11T10:10:00Z",
    labelEvaluatedAt: "2026-09-11T10:12:30Z",
    outcomeResolvedAt: "2026-09-11T10:25:00Z",
    measuredAt: "2026-09-11T11:00:00Z",
    maxLabelLatencySeconds: 120,
  });
  if (result.clean || !result.reasons.some((reason) => reason.includes("latency budget"))) throw new Error(JSON.stringify(result));
});
