import assert from "node:assert/strict";
import type { Struct } from "../_shared/dip_v8.ts";
import {
  microReclaimReady,
  selectEconomicTarget,
} from "./decision.ts";

function s(
  tf: string,
  trend: Struct["trend"],
  high: number,
  low: number,
  highLabel = "HH",
  lowLabel = "HL",
  pivots: Struct["pivots"] = [],
): Struct {
  return {
    tf,
    lastClose: (high + low) / 2,
    atr: 1.7,
    pivots,
    lastHigh: { i: 1, t: 1, p: high, kind: "H", label: highLabel },
    lastLow: { i: 2, t: 2, p: low, kind: "L", label: lowLabel },
    trend,
    bos: null,
    choch: null,
    sweep: null,
    failedBreak: null,
    equalHigh: null,
    equalLow: null,
    fingerprint: tf,
  };
}

Deno.test("micro reclaim promotes the observed HH/HL + 5m up + strong flow state", () => {
  const s1 = s("1m", "UP", 2492.85, 2488),
    s5 = s("5m", "UP", 2501.74, 2486.98);
  assert.equal(microReclaimReady(s1, s5, .84, 2490.82), true);
});

Deno.test("micro reclaim stays closed with weak flow or without 5m alignment", () => {
  const s1 = s("1m", "UP", 2492.85, 2488),
    s5 = s("5m", "UP", 2501.74, 2486.98);
  assert.equal(microReclaimReady(s1, s5, .10, 2490.82), false);
  assert.equal(
    microReclaimReady(s1, { ...s5, trend: "RANGE" }, .84, 2490.82),
    false,
  );
});

Deno.test("micro reclaim requires confirmed HH/HL rather than an unclassified bounce", () => {
  const s1 = s("1m", "UP", 2492.85, 2488, "LH", "HL"),
    s5 = s("5m", "UP", 2501.74, 2486.98);
  assert.equal(microReclaimReady(s1, s5, .84, 2490.82), false);
});

Deno.test("target ladder skips uneconomic nearby highs for a known farther structural high", () => {
  const s1 = s("1m", "UP", 2492.85, 2488, "HH", "HL", [
      { i: 1, t: 1, p: 2492.85, kind: "H", label: "HH" },
      { i: 2, t: 2, p: 2498.60, kind: "H", label: "LH" },
    ]),
    s5 = s("5m", "UP", 2501.74, 2486.98, "HH", "HL", [
      { i: 3, t: 3, p: 2501.74, kind: "H", label: "HH" },
    ]),
    s15 = s("15m", "RANGE", 2500, 2441.68, "HH", "LL", [
      { i: 4, t: 4, p: 2500, kind: "H", label: "HH" },
      { i: 5, t: 5, p: 2505.46, kind: "H", label: "HH" },
      { i: 6, t: 6, p: 2507.99, kind: "H", label: "HH" },
    ]);
  assert.equal(selectEconomicTarget("UP", 2490.82, 22.04, [s1, s5, s15]), 2505.46);
});

Deno.test("UP target never points below entry and WAIT has no pseudo target", () => {
  const s1 = s("1m", "UP", 2489.4, 2486.98, "HH", "HL", [
    { i: 1, t: 1, p: 2489.4, kind: "H", label: "HH" },
  ]);
  assert.equal(selectEconomicTarget("UP", 2494.32, 22.04, [s1]), null);
  assert.equal(selectEconomicTarget("WAIT", 2494.32, 22.04, [s1]), null);
});
