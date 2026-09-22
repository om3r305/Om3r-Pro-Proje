// Behavioral tests for the actual deployed brian-derivatives-eye scoring logic.
// Run with: deno test --allow-read supabase/functions/brian-derivatives-eye

import { assertEquals, assertExists } from "jsr:@std/assert@^1.0.0";
import {
  clip, finite, fundingCrowdingSignal, makeObs, oiPriceConfirmationSignal, sign, takerImbalanceSignal,
} from "./logic.ts";

Deno.test("finite: coerces to a finite number or falls back", () => {
  assertEquals(finite("1.5"), 1.5);
  assertEquals(finite(undefined, -1), -1);
  assertEquals(finite(null, 0), 0);
  assertEquals(finite(NaN, 7), 7);
  assertEquals(finite(Infinity, 7), 7);
});

Deno.test("clip: clamps to [0, 1]", () => {
  assertEquals(clip(-1), 0);
  assertEquals(clip(0.3), 0.3);
  assertEquals(clip(2), 1);
});

Deno.test("sign: returns -1/0/1", () => {
  assertEquals(sign(0.01), 1);
  assertEquals(sign(-0.01), -1);
  assertEquals(sign(0), 0);
});

Deno.test("fundingCrowdingSignal: below the 0.0002 noise floor produces no signal", () => {
  assertEquals(fundingCrowdingSignal(0.0001), null);
  assertEquals(fundingCrowdingSignal(-0.0001), null);
  assertEquals(fundingCrowdingSignal(0), null);
});

Deno.test("fundingCrowdingSignal: is contrarian -- positive (long-crowded) funding signals short", () => {
  const s = fundingCrowdingSignal(0.0005);
  assertExists(s);
  assertEquals(s!.direction, -1);
  assertEquals(s!.strength, 0.5);
});

Deno.test("fundingCrowdingSignal: negative (short-crowded) funding signals long, strength saturates at 1", () => {
  const s = fundingCrowdingSignal(-0.01);
  assertExists(s);
  assertEquals(s!.direction, 1);
  assertEquals(s!.strength, 1);
});

Deno.test("oiPriceConfirmationSignal: requires both OI change and price return above threshold", () => {
  assertEquals(oiPriceConfirmationSignal(0.001, 0.01), null);
  assertEquals(oiPriceConfirmationSignal(0.01, 0.0001), null);
  assertEquals(oiPriceConfirmationSignal(0.001, 0.0001), null);
});

Deno.test("oiPriceConfirmationSignal: direction follows price return when both thresholds clear", () => {
  const up = oiPriceConfirmationSignal(0.01, 0.005);
  assertExists(up);
  assertEquals(up!.direction, 1);
  const down = oiPriceConfirmationSignal(0.01, -0.005);
  assertExists(down);
  assertEquals(down!.direction, -1);
});

Deno.test("takerImbalanceSignal: inside the neutral band [0.925, 1.08) produces no signal", () => {
  assertEquals(takerImbalanceSignal(1), null);
  assertEquals(takerImbalanceSignal(0.95), null);
  assertEquals(takerImbalanceSignal(1.05), null);
});

Deno.test("takerImbalanceSignal: at or beyond the band edges is directional", () => {
  const buy = takerImbalanceSignal(1.2);
  assertExists(buy);
  assertEquals(buy!.direction, 1);
  const sell = takerImbalanceSignal(0.8);
  assertExists(sell);
  assertEquals(sell!.direction, -1);
});

Deno.test("makeObs: builds a shadow-only observation row with clipped strength/confidence", async () => {
  const obs = await makeObs(
    "eye-1", "funding-crowding", "crypto:BTCUSDT", "funding_crowding", "derivatives_funding",
    "2026-09-19T00:00:00.000Z", -1, 5, 2, "capture-1", "test reason", { funding_rate: 0.01 },
  );
  assertEquals(obs.direction, -1);
  assertEquals(obs.strength, 1);
  assertEquals(obs.confidence, 1);
  assertEquals(obs.shadow_only, true);
  assertEquals(obs.live_execution, false);
  assertExists(obs.observation_id);
});

Deno.test("makeObs: the observation id is deterministic for identical inputs", async () => {
  const a = await makeObs("eye-1", "t", "a", "f", "g", "2026-09-19T00:00:00.000Z", 1, 0.5, 0.5, "c", "r", {});
  const b = await makeObs("eye-1", "t", "a", "f", "g", "2026-09-19T00:00:00.000Z", 1, 0.5, 0.5, "c", "r", {});
  assertEquals(a.observation_id, b.observation_id);
});
