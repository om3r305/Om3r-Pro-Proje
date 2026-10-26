import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@1";
import { groupSessions, orbResults, orbRulesFingerprint, parseYahooBars, type Bar } from "./orb_shadow.ts";

// 2026-10-22 is EDT (UTC-4): 09:30 New York = 13:30 UTC.
const T0 = Date.UTC(2026, 9, 22, 13, 30) / 1000;
const bar = (k: number, o: number, h: number, l: number, c: number, v = 1000): Bar => ({ t: T0 + k * 300, o, h, l, c, v });
// Opening range 99..101 (mid 100), then a long breakout close 101.5 at 09:45.
function day(after: Bar[]): Bar[] {
  const or = [bar(0, 100, 101, 99, 100.5), bar(1, 100.5, 100.8, 99.5, 100), bar(2, 100, 100.9, 99.2, 100.6)];
  const flat: Bar[] = [];
  for (let k = 3 + after.length; k <= 77; k++) flat.push(bar(k, 101.6, 101.7, 101.55, 101.6));
  return [...or, ...after, ...flat];
}

Deno.test("sessions are grouped in New York time; outside-session bars are ignored", () => {
  const s = groupSessions([bar(-1, 1, 1, 1, 1), bar(0, 1, 1, 1, 1), bar(77, 1, 1, 1, 1), bar(78, 1, 1, 1, 1)]);
  assertEquals([...s.keys()], ["2026-10-22"]);
  assertEquals([...s.get("2026-10-22")!.keys()], ["09:30", "15:55"]);
});

Deno.test("1R target hit: long from the first close above the range, net of cost", () => {
  const r = orbResults(day([bar(3, 100.9, 101.6, 100.9, 101.5), bar(4, 101.5, 103.2, 101.4, 103)]), "etf");
  const a = r.find((x) => x.variant === "ORB_1R")!;
  assertEquals([a.status, a.direction, a.exitReason], ["TRADE", 1, "TARGET"]);
  assertEquals(a.entryPrice, 101.5);
  assertEquals(a.stopPrice, 100);
  assertEquals(a.targetPrice, 103);
  assertAlmostEquals(a.netBps!, (103 / 101.5 - 1) * 1e4 - 12, 1e-9);
  const b = r.find((x) => x.variant === "ORB_CLOSE")!;
  assertEquals(b.exitReason, "TIME", "no target: held to the 15:55 close");
  assertEquals(r.find((x) => x.variant === "ORB_IN_PLAY_1R")!.status, "INSUFFICIENT_DATA", "needs 10 prior sessions");
});

Deno.test("conservative fills: stop wins a bar that touches both; a gap through the stop exits at the open", () => {
  const both = orbResults(day([bar(3, 100.9, 101.6, 100.9, 101.5), bar(4, 101.5, 103.5, 99.8, 101)]), "equity").find((x) => x.variant === "ORB_1R")!;
  assertEquals([both.exitReason, both.exitPrice], ["STOP", 100]);
  const gap = orbResults(day([bar(3, 100.9, 101.6, 100.9, 101.5), bar(4, 99.0, 99.5, 98.8, 99.2)]), "equity").find((x) => x.variant === "ORB_1R")!;
  assertEquals([gap.exitReason, gap.exitPrice], ["STOP", 99]);
  assert(gap.netBps! < -18);
});

Deno.test("short breakouts and no-trigger days", () => {
  const s = orbResults(day([bar(3, 99.3, 99.4, 98.5, 98.6)]), "equity").find((x) => x.variant === "ORB_1R")!;
  assertEquals([s.status, s.direction, s.stopPrice], ["TRADE", -1, 100]);
  const none = orbResults(day([bar(3, 100, 100.5, 99.5, 100)]).map((b) => ({ ...b, h: Math.min(b.h, 100.95), l: Math.max(b.l, 99.05), c: 100, o: 100 })), "equity");
  assertEquals(none.find((x) => x.variant === "ORB_1R")!.status, "NO_TRIGGER");
});

Deno.test("in-play filter compares opening-range volume with the median of the previous 10 sessions", () => {
  const sessions: Bar[] = [];
  for (let d = 0; d < 11; d++) {
    const shift = (d - 10) * 86400; // all inside EDT (US DST ends 2026-11-01)
    const vol = d === 10 ? 5000 : 1000;
    for (const b of day([bar(3, 100.9, 101.6, 100.9, 101.5)])) sessions.push({ ...b, t: b.t + shift, v: vol });
  }
  const r = orbResults(sessions, "etf").filter((x) => x.variant === "ORB_IN_PLAY_1R");
  assertEquals(r.length, 11);
  assertEquals(r.slice(0, 10).every((x) => x.status === "INSUFFICIENT_DATA"), true);
  assertEquals(r[10].status, "TRADE");
  assertAlmostEquals(r[10].relVolume!, 5, 1e-9);
});

Deno.test("Yahoo payload parsing drops null rows; fingerprint is stable", async () => {
  const p = parseYahooBars({ chart: { result: [{ meta: { exchangeTimezoneName: "America/New_York" }, timestamp: [1, 2],
    indicators: { quote: [{ open: [1, null], high: [2, 2], low: [0.5, 1], close: [1.5, 1], volume: [10, 10] }] } }] } });
  assertEquals(p.bars.length, 1);
  assertEquals(p.tz, "America/New_York");
  assertEquals((await orbRulesFingerprint()).length, 64);
});
