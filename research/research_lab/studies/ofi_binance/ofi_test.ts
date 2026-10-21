// Offline tests for the OFI collector core. deno test --allow-read --allow-write research/research_lab/studies/ofi_binance/ofi_test.ts

import { eventContribution, parseBookTicker, SecondAggregator, type Top } from "./ofi.ts";
import { preflight } from "./main.ts";
import { Recorder } from "./recorder.ts";
import { assertAllowed, RULES, rulesFingerprint } from "./rules.ts";

function eq(a: unknown, b: unknown, m = "") {
  if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${m} expected ${JSON.stringify(b)} got ${JSON.stringify(a)}`);
}
function ok(c: unknown, m: string) {
  if (!c) throw new Error(`assertion failed: ${m}`);
}
const top = (bid: number, bidQty: number, ask: number, askQty: number, updateId = 1): Top => ({ bid, bidQty, ask, askQty, updateId });

Deno.test("e_n matches every case described in the paper (section 2.1)", () => {
  const p = top(100, 5, 101, 7);
  eq(eventContribution(p, top(100, 8, 101, 7)), 3, "bid size up at same price: qb_n - qb_{n-1}");
  eq(eventContribution(p, top(100, 2, 101, 7)), -3, "bid size down at same price");
  eq(eventContribution(p, top(100.5, 4, 101, 7)), 4, "bid price up: + qb_n (price-improving limit order)");
  eq(eventContribution(p, top(99.5, 9, 101, 7)), -5, "bid price down: - qb_{n-1} (removed size)");
  eq(eventContribution(p, top(100, 5, 101, 10)), -3, "ask size up at same price: supply increase");
  eq(eventContribution(p, top(100, 5, 101, 4)), 3, "ask size down at same price");
  eq(eventContribution(p, top(100, 5, 100.5, 6)), -6, "ask price down: - qa_n");
  eq(eventContribution(p, top(100, 5, 101.5, 2)), 7, "ask price up: + qa_{n-1}");
  eq(eventContribution(p, p), 0, "no change -> 0");
});

Deno.test("parser accepts combined-stream bookTicker and rejects crossed/invalid books", () => {
  const msg = { stream: "btcusdt@bookTicker", data: { u: 10, s: "BTCUSDT", b: "67000.10", B: "1.5", a: "67000.20", A: "2.0" } };
  eq(parseBookTicker(msg), { symbol: "BTCUSDT", bid: 67000.1, bidQty: 1.5, ask: 67000.2, askQty: 2, updateId: 10 });
  eq(parseBookTicker({ data: { u: 1, s: "BTCUSDT", b: "10", B: "1", a: "9", A: "1" } }), null, "crossed");
  eq(parseBookTicker({ data: { u: 1, s: "BTCUSDT", b: "x", B: "1", a: "9", A: "1" } }), null, "NaN");
  eq(parseBookTicker({ data: { u: 1, b: "1", B: "1", a: "2", A: "1" } }), null, "no symbol");
  eq(parseBookTicker(null), null);
});

Deno.test("aggregator: first message only sets state; OFI summed per second; quiet seconds carried while alive", () => {
  const a = new SecondAggregator("BTCUSDT", 5000);
  const t0 = 1_000_000_000; // ms, exactly on a second boundary
  eq(a.push(top(100, 5, 101, 7, 1), t0 + 100), [], "nothing closed yet");
  a.push(top(100, 8, 101, 7, 2), t0 + 300); // +3
  a.push(top(100.5, 4, 101, 7, 3), t0 + 900); // +4
  const closed = a.push(top(100.5, 4, 101, 9, 4), t0 + 2500); // second t0 closed, t0+1 quiet, ask +2 size -> -2
  eq(closed.length, 2, "two seconds closed");
  eq(closed[0].ofi, 7); eq(closed[0].events, 2); ok(closed[0].stateReset, "first bucket flags the reset");
  eq(closed[0].mid, (100.5 + 101) / 2);
  eq(closed[1].events, 0, "quiet second carried"); eq(closed[1].ofi, 0); eq(closed[1].mid, (100.5 + 101) / 2);
  const later = a.flush(t0 + 3100);
  eq(later.length, 1); eq(later[0].ofi, -2); eq(later[0].stateReset, false);
});

Deno.test("aggregator: silence > maxSilence is a gap; no e_n across it", () => {
  const a = new SecondAggregator("BTCUSDT", 5000);
  const t0 = 2_000_000_000;
  a.push(top(100, 5, 101, 7, 1), t0);
  a.push(top(100, 6, 101, 7, 2), t0 + 200); // +1
  const out = a.flush(t0 + 20_000);
  ok(out.length >= 1 && out.length <= 5, `only seconds within the silence window are emitted (${out.length})`);
  // Next message after the gap must only re-initialise the state (big size change NOT counted).
  const resumed = a.push(top(100, 50, 101, 7, 3), t0 + 20_100);
  eq(resumed, []);
  const after = a.push(top(100, 51, 101, 7, 4), t0 + 21_200);
  eq(after.length, 1);
  eq(after[0].ofi, 0, "the re-initialising message contributes nothing");
  ok(after[0].stateReset, "bucket after a gap is flagged");
});

Deno.test("aggregator: reconnect reset and out-of-order update ids", () => {
  const a = new SecondAggregator("ETHUSDT", 5000);
  const t0 = 3_000_000_000;
  a.push(top(10, 1, 11, 1, 5), t0);
  a.push(top(10, 3, 11, 1, 4), t0 + 100); // older update id -> ignored
  a.reset();
  a.push(top(10, 9, 11, 1, 6), t0 + 200); // state only
  a.push(top(10, 10, 11, 1, 7), t0 + 300); // +1
  const out = a.flush(t0 + 1000);
  eq(out.length, 1);
  eq(out[0].ofi, 1, "stale message ignored, reset message not counted");
  eq(out[0].events, 1);
});

Deno.test("guards: host allow-list, frozen rules, approval required", async () => {
  let threw = false;
  try { assertAllowed("wss://stream.binance.com:9443/ws"); } catch { threw = true; }
  ok(threw, "only data-stream.binance.vision / data-api.binance.vision");
  ok(assertAllowed("wss://data-stream.binance.vision/stream?streams=btcusdt@bookTicker"), "allowed");
  ok(Object.isFrozen(RULES), "rules frozen");
  const fp = await rulesFingerprint();
  eq(fp, await rulesFingerprint(), "stable fingerprint");
  const good = { offsetMs: 300, uncertaintyMs: 4 };
  eq(preflight({ skew: good, freeBytes: 10 * 1024 ** 3, fp, approved: fp }).ok, true);
  eq(preflight({ skew: good, freeBytes: 10 * 1024 ** 3, fp, approved: null }).ok, false, "no approval");
  eq(preflight({ skew: good, freeBytes: 10 * 1024 ** 3, fp, approved: "x" }).ok, false, "wrong approval");
  eq(preflight({ skew: { offsetMs: 0, uncertaintyMs: 80 }, freeBytes: 10 * 1024 ** 3, fp, approved: fp }).ok, false, "clock");
  eq(preflight({ skew: good, freeBytes: 100, fp, approved: fp }).ok, false, "disk");
});

Deno.test("recorder: header once per file, byte cap enforced", async () => {
  const dir = await Deno.makeTempDir();
  const r = new Recorder({ dir, capBytes: 400, rulesVersion: "v", rulesFingerprint: "f" });
  ok(await r.write("bucket", "BTCUSDT", [{ a: 1 }], Date.UTC(2026, 9, 21)), "first write");
  ok(await r.write("bucket", "BTCUSDT", [{ a: 2 }], Date.UTC(2026, 9, 21)), "second write");
  const lines = (await Deno.readTextFile(`${dir}/2026-10-21.BTCUSDT.bucket.jsonl`)).trim().split("\n");
  eq(lines.length, 3, "header + 2 records");
  ok(JSON.parse(lines[0]).header === true, "header first");
  ok(!(await r.write("bucket", "BTCUSDT", [{ big: "x".repeat(500) }], Date.UTC(2026, 9, 21))), "cap refuses");
  await Deno.remove(dir, { recursive: true });
});
