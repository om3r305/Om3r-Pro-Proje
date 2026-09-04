import { assert, assertEquals } from "jsr:@std/assert@1";
import { compileAlphaDecision, dedupeIndependentEvidence, type AlphaEvidenceRow } from "./alpha_decision.ts";
import { compileDegradedTopOfBookCost } from "./dynamic_cost.ts";

function row(partial: Partial<AlphaEvidenceRow> & Pick<AlphaEvidenceRow, "observationId" | "independentGroup" | "direction">): AlphaEvidenceRow {
  return {
    observationId: partial.observationId,
    sourceKind: partial.sourceKind ?? "sensor",
    independentGroup: partial.independentGroup,
    direction: partial.direction,
    strength: partial.strength ?? 1,
    confidence: partial.confidence ?? 1,
    reliability: partial.reliability ?? 1,
    observedAt: partial.observedAt ?? "2026-09-04T12:00:00Z",
    horizon: partial.horizon ?? "MICRO_1_5M",
    fresh: partial.fresh ?? true,
    reason: partial.reason ?? "fixture",
  };
}

const cost = compileDegradedTopOfBookCost({
  side: "BUY", notionalUsd: 100, feeBps: 10, spreadBps: 2, assumedSlippageBps: 1, midPrice: 100,
});

Deno.test("alpha compiler: correlated duplicate group counts once", () => {
  const d = dedupeIndependentEvidence([
    row({ observationId: "a", independentGroup: "micro_velocity", direction: 1, strength: 0.8 }),
    row({ observationId: "b", independentGroup: "micro_velocity", direction: 1, strength: 0.6 }),
    row({ observationId: "c", independentGroup: "derivatives_taker", direction: 1, strength: 0.8 }),
  ]);
  assertEquals(d.evidence.length, 2);
  assert(d.ignoredObservationIds.includes("b"));
  assertEquals(d.evidence.map((x) => x.independentGroup), ["derivatives_taker", "intrabar_tape"]);
});

Deno.test("alpha compiler: different micro signals from one intrabar tape count as one independent vote", () => {
  const out = compileAlphaDecision({
    evidenceRows: [
      row({ observationId: "velocity", independentGroup: "micro_velocity", direction: 1 }),
      row({ observationId: "breakout", independentGroup: "micro_breakout", direction: 1 }),
    ],
    costQuote: cost,
  });
  assertEquals(out.independentGroupCount, 1);
  assertEquals(out.supportGroups, ["intrabar_tape"]);
  assertEquals(out.action, "WAIT");
});

Deno.test("alpha compiler: GDELT discovery headline pressure cannot vote direction", () => {
  const out = compileAlphaDecision({
    evidenceRows: [
      row({ observationId: "gdelt", sourceKind: "news_attention", independentGroup: "news_gdelt", direction: 1 }),
      row({ observationId: "deriv", independentGroup: "derivatives_taker", direction: 1 }),
    ],
    costQuote: cost,
  });
  assertEquals(out.independentGroupCount, 1);
  assert(out.ignoredObservationIds.includes("gdelt"));
  assertEquals(out.action, "WAIT");
});

Deno.test("alpha compiler: stale and neutral evidence cannot vote", () => {
  const d = dedupeIndependentEvidence([
    row({ observationId: "stale", independentGroup: "micro_velocity", direction: 1, fresh: false }),
    row({ observationId: "neutral", independentGroup: "derivatives_taker", direction: 0 }),
  ]);
  assertEquals(d.evidence.length, 0);
  assertEquals(d.ignoredObservationIds, ["neutral", "stale"]);
});

Deno.test("alpha compiler: two strong independent aligned groups can open long using existing gate", () => {
  const out = compileAlphaDecision({
    evidenceRows: [
      row({ observationId: "v", independentGroup: "micro_velocity", direction: 1, strength: 0.9, confidence: 0.9, reliability: 0.8 }),
      row({ observationId: "d", independentGroup: "derivatives_taker", direction: 1, strength: 0.9, confidence: 0.9, reliability: 0.8 }),
    ],
    costQuote: cost,
  });
  assertEquals(out.action, "OPEN_LONG");
  assertEquals(out.direction, 1);
  assert(out.evidenceScore >= 0.18);
  assertEquals(out.supportGroups, ["derivatives_taker", "intrabar_tape"]);
});

Deno.test("alpha compiler: one independent group remains WAIT even if very strong", () => {
  const out = compileAlphaDecision({
    evidenceRows: [row({ observationId: "v", independentGroup: "micro_velocity", direction: -1, strength: 1 })],
    costQuote: { ...cost, side: "SELL" },
  });
  assertEquals(out.action, "WAIT");
  assertEquals(out.direction, -1);
});

Deno.test("alpha compiler: conflicts reduce score and can prevent action", () => {
  const out = compileAlphaDecision({
    evidenceRows: [
      row({ observationId: "a", independentGroup: "micro_velocity", direction: 1, strength: 0.5, confidence: 0.8, reliability: 0.5 }),
      row({ observationId: "b", independentGroup: "derivatives_taker", direction: 1, strength: 0.5, confidence: 0.8, reliability: 0.5 }),
      row({ observationId: "c", independentGroup: "derivatives_oi", direction: -1, strength: 1, confidence: 1, reliability: 1 }),
    ],
    costQuote: cost,
  });
  assertEquals(out.action, "WAIT");
  assert(out.conflictGroups.length > 0 || out.supportGroups.length < 2);
});

Deno.test("alpha compiler: matching fresh late-chase veto blocks otherwise actionable evidence", () => {
  const out = compileAlphaDecision({
    evidenceRows: [
      row({ observationId: "a", independentGroup: "micro_velocity", direction: 1 }),
      row({ observationId: "b", independentGroup: "derivatives_taker", direction: 1 }),
    ],
    costQuote: cost,
    intrabarContext: {
      eventId: "late-1", direction: 1, status: "VETOED_LATE_CHASE", lateChase: true,
      observedAt: "2026-09-04T12:00:00Z", fresh: true, reason: "extended",
    },
  });
  assertEquals(out.action, "VETO");
  assertEquals(out.vetoReason, "LATE_CHASE");
  assertEquals(out.lateChaseVetoEventId, "late-1");
});

Deno.test("alpha compiler: opposite-direction late-chase event does not veto", () => {
  const out = compileAlphaDecision({
    evidenceRows: [
      row({ observationId: "a", independentGroup: "micro_velocity", direction: 1 }),
      row({ observationId: "b", independentGroup: "derivatives_taker", direction: 1 }),
    ],
    costQuote: cost,
    intrabarContext: {
      eventId: "late-short", direction: -1, status: "VETOED_LATE_CHASE", lateChase: true,
      observedAt: "2026-09-04T12:00:00Z", fresh: true, reason: "extended short",
    },
  });
  assertEquals(out.action, "OPEN_LONG");
});

Deno.test("alpha compiler: actionable evidence without cost evidence fails closed", () => {
  const out = compileAlphaDecision({
    evidenceRows: [
      row({ observationId: "a", independentGroup: "micro_velocity", direction: -1 }),
      row({ observationId: "b", independentGroup: "derivatives_taker", direction: -1 }),
    ],
    costQuote: null,
  });
  assertEquals(out.action, "VETO");
  assertEquals(out.vetoReason, "COST_UNAVAILABLE");
});

Deno.test("alpha compiler: unfillable L2 cost fails closed", () => {
  const out = compileAlphaDecision({
    evidenceRows: [
      row({ observationId: "a", independentGroup: "micro_velocity", direction: -1 }),
      row({ observationId: "b", independentGroup: "derivatives_taker", direction: -1 }),
    ],
    costQuote: { ...cost, fillable: false, fillRatio: 0.2 },
  });
  assertEquals(out.action, "VETO");
  assertEquals(out.vetoReason, "INSUFFICIENT_VISIBLE_DEPTH");
});
