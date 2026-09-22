// Behavioral tests for the actual deployed brian-sensor-mesh parsing/scoring logic.
// Run with: deno test --allow-read supabase/functions/brian-sensor-mesh

import { assertEquals, assertThrows } from "jsr:@std/assert@^1.0.0";
import {
  type Bar, clip, confidenceFromRadar, finite, mean, parseBars, signalMeanReversion, signalMomentum,
  signalStructure, std, ticketFor,
} from "./logic.ts";

function bar(closeTime: number, o: number, h: number, l: number, c: number, v = 100): Bar {
  return { closeTime, open: o, high: h, low: l, close: c, volume: v };
}

Deno.test("finite: coerces or falls back", () => {
  assertEquals(finite("2.5"), 2.5);
  assertEquals(finite("nan", 9), 9);
  assertEquals(finite(undefined, 1), 1);
});

Deno.test("clip: clamps to a custom [low, high] range, defaulting to [0, 1]", () => {
  assertEquals(clip(-1), 0);
  assertEquals(clip(2), 1);
  assertEquals(clip(-5, -3, 3), -3);
  assertEquals(clip(5, -3, 3), 3);
});

Deno.test("mean/std: empty and single-element inputs are handled without dividing by zero", () => {
  assertEquals(mean([]), 0);
  assertEquals(std([]), 0);
  assertEquals(std([5]), 0);
  assertEquals(mean([1, 2, 3]), 2);
  assertEquals(std([2, 2, 2]), 0);
});

Deno.test("parseBars: normalizes a valid kline payload and drops rows after the cutoff", () => {
  const payload = [
    [0, "1", "1.1", "0.9", "1.05", "10", 1000],
    [0, "1.05", "1.2", "1.0", "1.1", "12", 2000],
    [0, "1.1", "1.3", "1.05", "1.2", "15", 3000],
  ];
  const bars = parseBars(payload, 2000);
  assertEquals(bars.length, 2);
  assertEquals(bars[1], { closeTime: 2000, open: 1.05, high: 1.2, low: 1.0, close: 1.1, volume: 12 });
});

Deno.test("parseBars: a non-positive OHLC value is dropped, not coerced", () => {
  const payload = [[0, "0", "1", "0.9", "1", "10", 1000]];
  assertEquals(parseBars(payload, 5000), []);
});

Deno.test("parseBars: rejects a malformed payload shape instead of silently returning garbage", () => {
  assertThrows(() => parseBars({ not: "an array" }, 1000));
  assertThrows(() => parseBars([["too", "short"]], 1000));
});

Deno.test("signalStructure: needs at least 8 bars of prior context, else neutral", () => {
  const bars = Array.from({ length: 5 }, (_, i) => bar(i, 1, 1, 1, 1));
  assertEquals(signalStructure(bars).direction, 0);
});

Deno.test("signalStructure: a close clearly above the prior range is a bullish breakout", () => {
  const prior = Array.from({ length: 12 }, (_, i) => bar(i, 100, 101, 99, 100));
  const bars = [...prior, bar(12, 100, 110, 100, 108)];
  const s = signalStructure(bars);
  assertEquals(s.direction, 1);
  assertEquals(s.reason, "closed breakout above prior structure");
});

Deno.test("signalStructure: a close clearly below the prior range is a bearish breakdown", () => {
  const prior = Array.from({ length: 12 }, (_, i) => bar(i, 100, 101, 99, 100));
  const bars = [...prior, bar(12, 100, 100, 90, 92)];
  const s = signalStructure(bars);
  assertEquals(s.direction, -1);
});

Deno.test("signalMomentum: needs at least 6 closes, else neutral", () => {
  const bars = Array.from({ length: 3 }, (_, i) => bar(i, 1, 1, 1, 1 + i * 0.001));
  assertEquals(signalMomentum(bars).direction, 0);
});

Deno.test("signalMomentum: a clear 4-bar impulse above the adaptive threshold is directional", () => {
  const closes = [100, 100.05, 100.1, 100.15, 100.2, 100.25, 105, 108, 112, 118];
  const bars = closes.map((c, i) => bar(i, c, c, c, c));
  const s = signalMomentum(bars);
  assertEquals(s.direction, 1);
});

Deno.test("signalMeanReversion: a flat series (zero std) is neutral, not a divide-by-zero", () => {
  const bars = Array.from({ length: 12 }, (_, i) => bar(i, 100, 100, 100, 100));
  assertEquals(signalMeanReversion(bars), { direction: 0, strength: 0, reason: "flat mean-reversion context" });
});

Deno.test("signalMeanReversion: a large upward stretch signals reversion down", () => {
  const closes = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 130];
  const bars = closes.map((c, i) => bar(i, c, c, c, c));
  const s = signalMeanReversion(bars);
  assertEquals(s.direction, -1);
});

Deno.test("ticketFor: known templates return their configured ticket size, unknown falls back to 5", () => {
  assertEquals(ticketFor("structure-fast"), 5.0);
  assertEquals(ticketFor("mean-reversion-fast"), 3.0);
  assertEquals(ticketFor("unknown-template"), 5);
});

Deno.test("confidenceFromRadar: blends liquidity, activity, and spread quality into [0, 1]", () => {
  const wide = confidenceFromRadar({ symbol: "BTCUSDT", liquidity_score: 1, activity_score: 1 }, { bid: 1, ask: 1.1, mid: 1.05, spreadBps: 500 });
  const tight = confidenceFromRadar({ symbol: "BTCUSDT", liquidity_score: 1, activity_score: 1 }, { bid: 1, ask: 1.0001, mid: 1.00005, spreadBps: 1 });
  assertEquals(tight > wide, true);
  assertEquals(clip(tight), tight);
});
