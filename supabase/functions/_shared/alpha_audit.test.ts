import { assertEquals, assert } from "jsr:@std/assert@1";
import { resolveAlphaAuditHorizon } from "./alpha_audit.ts";

Deno.test("auditor uses immutable decision reference instead of first later tick", () => {
  const resolved = resolveAlphaAuditHorizon({
    observedAt: "2026-09-04T12:00:00Z",
    action: "OPEN_LONG",
    direction: 1,
    referencePrice: 100,
    estimatedRoundTripCostBps: 0,
  }, 300, [
    { observed_at: "2026-09-04T12:00:30Z", observed_mid_price: 110, estimated_round_trip_cost_bps: 0 },
    { observed_at: "2026-09-04T12:05:00Z", observed_mid_price: 120, estimated_round_trip_cost_bps: 0 },
  ]);
  assert(resolved);
  assertEquals(resolved.reference, 100);
  assertEquals(resolved.resolved, 120);
  assert(Math.abs(resolved.gross - 0.2) < 1e-12);
});

Deno.test("auditor fails closed when immutable decision reference is missing", () => {
  const resolved = resolveAlphaAuditHorizon({
    observedAt: "2026-09-04T12:00:00Z",
    action: "WAIT",
    direction: 0,
    referencePrice: null,
    estimatedRoundTripCostBps: 10,
  }, 300, [
    { observed_at: "2026-09-04T12:00:10Z", observed_mid_price: 100, estimated_round_trip_cost_bps: 10 },
    { observed_at: "2026-09-04T12:05:00Z", observed_mid_price: 101, estimated_round_trip_cost_bps: 10 },
  ]);
  assertEquals(resolved, null);
});

Deno.test("auditor rejects a horizon without a near-target resolution point", () => {
  const resolved = resolveAlphaAuditHorizon({
    observedAt: "2026-09-04T12:00:00Z",
    action: "WAIT",
    direction: 0,
    referencePrice: 100,
    estimatedRoundTripCostBps: 10,
  }, 300, [
    { observed_at: "2026-09-04T12:01:00Z", observed_mid_price: 101, estimated_round_trip_cost_bps: 10 },
  ]);
  assertEquals(resolved, null);
});
