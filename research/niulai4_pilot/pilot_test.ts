import { applyMessage, type BookState, bestAsk, bestBid, topLevels, walkAsks } from "./book.ts";
import { type Capture, evaluate, type Observation, summarize, type WindowInfo } from "./evaluate.ts";
import { estimateSkew, evaluatePreflight, reconnectDelayMs, StopGuard } from "./guards.ts";
import { readJsonl, Recorder } from "./recorder.ts";
import { assertPilotHost, RULES, RULES_VERSION, rulesFingerprint } from "./rules.ts";
import { SpotSeries, spotDirection } from "./spot.ts";
import { clusterBootstrapMean } from "./stats.ts";
import { ActivityWatcher, captureTimes } from "./watcher.ts";
import { activeAndNext, inSamplingTail, windowAt, windowFromSlug, windowMatchesMarket, withinActiveHours } from "./windows.ts";

function assert(c: unknown, m: string): asserts c {
  if (!c) throw new Error(`assertion failed: ${m}`);
}
function eq<T>(a: T, b: T, m: string) {
  if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${m}: expected ${JSON.stringify(b)} got ${JSON.stringify(a)}`);
}
const near = (a: number | null | undefined, b: number, m: string, tol = 1e-6) => {
  if (a === null || a === undefined || Math.abs(a - b) > tol) throw new Error(`${m}: expected ${b} got ${a}`);
};

// ---------------------------------------------------------------- rules / hosts
Deno.test("rules are frozen, fingerprinted, and the host allow-list refuses everything else", async () => {
  const a = await rulesFingerprint();
  assert(a === (await rulesFingerprint()) && /^[0-9a-f]{64}$/.test(a), "stable sha256 fingerprint");
  assert(Object.isFrozen(RULES) && Object.isFrozen(RULES.delaysSeconds), "rules frozen");
  eq(RULES.primary, { delaySeconds: 5, baseline: "B1_SPOT_DIRECTION" }, "primary comparison is pre-registered");
  assert(RULES.feeRateFallback === null, "no assumed fee rate");
  assertPilotHost("wss://ws-subscriptions-clob.polymarket.com/ws/market");
  assertPilotHost("https://api.exchange.coinbase.com/products/BTC-USD/ticker");
  for (const bad of ["http://data-api.polymarket.com/x", "https://evil.example/x", "wss://user:pw@ws-subscriptions-clob.polymarket.com/x", "https://qbcjuxhvhwagvqbjyemo.supabase.co/rest/v1/x"]) {
    let threw = false;
    try {
      assertPilotHost(bad);
    } catch {
      threw = true;
    }
    assert(threw, `refused ${bad}`);
  }
});

// ---------------------------------------------------------------- windows
Deno.test("window arithmetic, slug parsing, hours, tail and market cross-check", () => {
  const t = Date.parse("2026-10-20T12:03:30Z");
  const w = windowAt(t);
  eq([w.startSec, w.endSec, w.slug], [Date.parse("2026-10-20T12:00:00Z") / 1000, Date.parse("2026-10-20T12:05:00Z") / 1000, `btc-updown-5m-${Date.parse("2026-10-20T12:00:00Z") / 1000}`], "window");
  eq(activeAndNext(t)[1].startSec, w.endSec, "next window");
  eq(windowFromSlug(w.slug), w, "slug round trip");
  for (const bad of ["eth-updown-5m-1792479000", "btc-updown-5m-17924790", "btc-updown-5m-1792479001", "btc-updown-5m-x"]) eq(windowFromSlug(bad), null, bad);
  assert(withinActiveHours(Date.parse("2026-10-20T12:00:00Z")) && withinActiveHours(Date.parse("2026-10-20T23:59:59Z")), "inside hours");
  assert(!withinActiveHours(Date.parse("2026-10-20T11:59:59Z")) && !withinActiveHours(Date.parse("2026-10-21T00:00:00Z")), "outside hours");
  assert(inSamplingTail(Date.parse("2026-10-20T12:03:00Z"), w) && !inSamplingTail(Date.parse("2026-10-20T12:02:59Z"), w), "tail = last 120 s");
  assert(!inSamplingTail(Date.parse("2026-10-20T12:05:00Z"), w), "window end excluded");
  assert(windowMatchesMarket(w, "2026-10-20T12:00:00Z", "2026-10-20T12:05:00Z"), "market agrees");
  assert(!windowMatchesMarket(w, "2026-10-20T12:00:00Z", "2026-10-20T12:15:00Z") && !windowMatchesMarket(w, null, undefined), "mismatch skipped");
});

// ---------------------------------------------------------------- book
const UP = "111";
const DN = "222";
function snapshot(asset: string, bids: [string, string][], asks: [string, string][]) {
  return { event_type: "book", asset_id: asset, timestamp: "1792479793785", bids: bids.map(([price, size]) => ({ price, size })), asks: asks.map(([price, size]) => ({ price, size })) };
}
Deno.test("book reducer: snapshot, absolute-size deltas, removal, unsynced deltas are not applied", () => {
  const s: BookState = new Map();
  const tracked = new Set([UP, DN]);
  const early = applyMessage(s, { event_type: "price_change", price_changes: [{ asset_id: UP, price: "0.5", size: "10", side: "SELL" }] }, tracked);
  eq(early.unsyncedDeltas, 1, "delta before snapshot is held back");
  assert(!s.has(UP), "no partial book");
  applyMessage(s, [snapshot(UP, [["0.48", "100"], ["0.47", "50"]], [["0.99", "9"], ["0.52", "20"], ["0.53", "40"]]), snapshot("999", [], [])], tracked);
  assert(!s.has("999"), "untracked asset ignored");
  const b = s.get(UP)!;
  eq([bestBid(b), bestAsk(b)], [0.48, 0.52], "best levels regardless of wire order");
  applyMessage(s, { event_type: "price_change", price_changes: [{ asset_id: UP, price: "0.52", size: "0", side: "SELL" }, { asset_id: UP, price: "0.51", size: "7", side: "SELL" }, { asset_id: UP, price: "0.49", size: "3", side: "BUY" }] }, tracked);
  eq([bestBid(b), bestAsk(b)], [0.49, 0.51], "size 0 removes, absolute size sets");
  eq(topLevels(b, 2).asks, [{ price: 0.51, size: 7 }, { price: 0.53, size: 40 }], "asks ascending");
  const junk = applyMessage(s, [{ event_type: "last_trade_price" }, null, { event_type: "price_change", price_changes: [{ asset_id: UP, price: "2", size: "1", side: "SELL" }] }], tracked);
  eq(junk.ignored, 3, "unknown events, null and impossible prices are ignored");
});

Deno.test("walking the asks: fixed size, tolerance above best ask, no partial fills", () => {
  const asks = [{ price: 0.52, size: 4 }, { price: 0.53, size: 4 }, { price: 0.54, size: 10 }, { price: 0.60, size: 100 }];
  const f = walkAsks(asks, 10, 0.02);
  assert(f.status === "FILLED", "fills within +0.02");
  near(f.avgPrice, (4 * 0.52 + 4 * 0.53 + 2 * 0.54) / 10, "average price");
  near(f.slippage, f.avgPrice - 0.52, "slippage vs best ask");
  eq(walkAsks(asks, 20, 0.02).status, "UNFILLABLE", "would need 0.60, beyond tolerance");
  eq(walkAsks([], 10, 0.02).status, "UNFILLABLE", "no asks");
  eq(walkAsks([{ price: 0.52, size: 9.99 }], 10, 0.02).status, "UNFILLABLE", "partial fill is not a fill");
});

// ---------------------------------------------------------------- spot
Deno.test("spot proxy: point-in-time value, 60 s TWAP with coverage rule, direction", () => {
  const s = new SpotSeries();
  const t0 = 1_000_000;
  s.add({ tMs: t0, mid: 100, source: "binance" });
  s.add({ tMs: t0 + 30_000, mid: 110, source: "binance" });
  s.add({ tMs: t0 + 45_000, mid: 999, source: "coinbase" });
  eq(s.at(t0 + 29_999), 100, "value at time");
  eq(s.at(t0 - 1), null, "nothing before first tick");
  near(s.twap(t0 + 60_000), (100 * 30 + 110 * 30) / 60, "twap holds each tick until the next");
  eq(s.twap(t0 + 59_999), null, "no tick at window start -> null, not a guess");
  eq(spotDirection(s, t0 + 60_000, t0), "Up", "twap 105 >= open 100");
  eq(spotDirection(s, t0 + 60_000, t0 - 5), null, "unknown open -> no decision");
  s.add({ tMs: t0 + 61_000, mid: Number.NaN, source: "binance" });
  eq(s.at(t0 + 61_000), 110, "invalid ticks dropped");
});

// ---------------------------------------------------------------- evaluation
const START = 1_792_480_800; // 2026-10-20T13:20:00Z, inside active hours
const slug = `btc-updown-5m-${START}`;
function obs(id: string, outcome: "Up" | "Down", seenOffsetSec: number, lagSec = 26, s = START): Observation {
  const firstSeenMs = (s + seenOffsetSec) * 1000;
  return { obsId: id, windowSlug: `btc-updown-5m-${s}`, windowStartSec: s, windowEndSec: s + 300, outcome, tradeTsMs: firstSeenMs - lagSec * 1000, firstSeenMs };
}
const book = (upAsk: number, dnAsk: number) => ({ Up: [{ price: upAsk, size: 1000 }], Down: [{ price: dnAsk, size: 1000 }] });
function caps(o: Observation, upAsk = 0.6, dnAsk = 0.42, spotDir: "Up" | "Down" | null = "Down", mids = { Up: 0.59, Down: 0.41 }): Capture[] {
  return captureTimes(o.firstSeenMs).map((c) => ({ obsId: o.obsId, delaySeconds: c.delaySeconds, capturedMs: c.atMs, asks: book(upAsk, dnAsk), mids, spotDirection: spotDir }));
}

Deno.test("evaluation: follow vs baselines, fee per level, late and unfillable, settlement", () => {
  const o = obs("o1", "Up", 240); // seen 60 s before end
  const r = evaluate([o], caps(o), [{ slug, feeRate: 0.07, winner: "Up" }]);
  const get = (s: string, d: number) => r.entries.find((e) => e.strategy === s && e.delaySeconds === d)!;
  const f5 = get("FOLLOW", 5);
  eq([f5.status, f5.outcome], ["SETTLED", "Up"], "follow buys NIULAI4's outcome");
  near(f5.cost, 6, "10 x 0.60");
  near(f5.fee, 10 * 0.07 * 0.6 * 0.4, "C x rate x p(1-p)");
  near(f5.net, 10 - 6 - 0.168, "payoff - cost - fee");
  eq(get("B0_FAVORITE", 5).outcome, "Up", "favorite = higher mid");
  eq([get("B1_SPOT_DIRECTION", 5).outcome, get("B1_SPOT_DIRECTION", 5).net! < 0], ["Down", true], "spot baseline lost");
  eq(get("FOLLOW", 60).status, "LATE", "60 s after first-seen is past the window end");
  const thin = evaluate([o], caps(o, 0.6, 0.42).map((c) => ({ ...c, asks: { Up: [{ price: 0.6, size: 3 }], Down: [{ price: 0.42, size: 3 }] } })), [{ slug, feeRate: 0.07, winner: "Up" }]);
  assert(thin.entries.filter((e) => e.delaySeconds === 5).every((e) => e.status === "UNFILLABLE" || e.status === "NO_DECISION"), "thin book -> UNFILLABLE");
  near(r.latencySeconds.p50, 26, "latency from first-seen minus public timestamp");
});

Deno.test("evaluation never assumes: no capture, unsynced book, missing fee, unresolved", () => {
  const o = obs("o2", "Down", 100);
  const none = evaluate([o], [], [{ slug, feeRate: 0.07, winner: "Down" }]);
  assert(none.entries.filter((e) => e.delaySeconds < 60).every((e) => e.status === "NO_CAPTURE"), "no capture recorded");
  const unsynced = evaluate([o], caps(o).map((c) => ({ ...c, asks: null, mids: null })), [{ slug, feeRate: 0.07, winner: "Down" }]);
  assert(unsynced.entries.every((e) => e.status === "BOOK_UNSYNCED" || e.status === "LATE"), "book not synced");
  const nofee = evaluate([o], caps(o), [{ slug, feeRate: null, winner: "Down" }]);
  assert(nofee.entries.filter((e) => e.strategy === "FOLLOW").every((e) => e.status === "NO_FEE_RATE"), "fee rate never assumed");
  const unres = evaluate([o], caps(o), [{ slug, feeRate: 0.07, winner: null }]);
  const f = unres.entries.find((e) => e.strategy === "FOLLOW" && e.delaySeconds === 0)!;
  eq([f.status, f.net], ["UNRESOLVED", undefined], "unresolved window has no return");
  const nodir = evaluate([o], caps(o, 0.6, 0.42, null, { Up: 0.5, Down: 0.5 }), [{ slug, feeRate: 0.07, winner: "Down" }]);
  assert(["B0_FAVORITE", "B1_SPOT_DIRECTION"].every((s) => nodir.entries.filter((e) => e.strategy === s).every((e) => e.status === "NO_DECISION")), "tie / unknown spot -> no decision");
});

Deno.test("capital is reserved until the window settles and released after it", () => {
  // 100 USDC bankroll; each entry costs 10 x 0.95 + fee ~ 9.53. Ten fit, the eleventh does not.
  const many = Array.from({ length: 11 }, (_, i) => obs(`m${i}`, "Up", 200 + i));
  const c = many.flatMap((o) => caps(o, 0.95, 0.06, "Up", { Up: 0.94, Down: 0.06 }));
  const r = evaluate(many, c, [{ slug, feeRate: 0.07, winner: "Up" }]);
  const f0 = r.entries.filter((e) => e.strategy === "FOLLOW" && e.delaySeconds === 0);
  eq(f0.filter((e) => e.status === "SETTLED").length, 10, "ten entries fit the bankroll");
  eq(f0.filter((e) => e.status === "REJECTED_CAPITAL").length, 1, "the eleventh is rejected");
  // A later window: capital from the first (settled at its end) is free again.
  const nextStart = START + 300;
  const later = obs("later", "Up", 10, 26, nextStart);
  const r2 = evaluate([...many, later], [...c, ...caps(later, 0.95, 0.06, "Up", { Up: 0.94, Down: 0.06 })], [{ slug, feeRate: 0.07, winner: "Up" }, { slug: `btc-updown-5m-${nextStart}`, feeRate: 0.07, winner: "Up" }]);
  eq(r2.entries.find((e) => e.obsId === "later" && e.strategy === "FOLLOW" && e.delaySeconds === 0)!.status, "SETTLED", "released capital is reusable");
});

Deno.test("random baseline is deterministic and independent of other rows' presence", () => {
  const o = obs("r1", "Up", 100);
  const a = evaluate([o], caps(o), [{ slug, feeRate: 0.07, winner: "Up" }]).entries.filter((e) => e.strategy === "B2_RANDOM").map((e) => e.outcome);
  const b = evaluate([o], caps(o), [{ slug, feeRate: 0.07, winner: "Up" }]).entries.filter((e) => e.strategy === "B2_RANDOM").map((e) => e.outcome);
  eq(a, b, "same seed, same draws");
});

Deno.test("summary: primary pairs only rows settled under BOTH strategies; edge needs >=1000 windows and >=14 days", () => {
  const os = Array.from({ length: 30 }, (_, i) => obs(`s${i}`, i % 2 ? "Up" : "Down", 150, 26, START + 300 * i));
  const ws: WindowInfo[] = os.map((o, i) => ({ slug: o.windowSlug, feeRate: 0.07, winner: i % 3 ? "Up" : "Down" }));
  const cs = os.flatMap((o) => caps(o));
  const s = summarize(os, evaluate(os, cs, ws), RULES_VERSION);
  eq(s.primary.pairedObservations, 30, "paired rows");
  eq(s.edge.verdict, "INSUFFICIENT_DATA", "30 windows is far from enough");
  assert(s.edge.reason.includes(">= 1000"), "reason states the bar");
  eq(s.feasibility.verdict, "INSUFFICIENT_DATA", "fewer than 100 attempts is not a feasibility verdict");
  assert(s.tradeSignal === false, "never a trade signal");
  eq(s.windows, 30, "independent windows counted");
});

Deno.test("feasibility verdicts follow the pre-registered thresholds", () => {
  const os = Array.from({ length: 120 }, (_, i) => obs(`f${i}`, "Up", 100, 26, START + 300 * i));
  const ws: WindowInfo[] = os.map((o) => ({ slug: o.windowSlug, feeRate: 0.07, winner: "Up" }));
  const fillable = summarize(os, evaluate(os, os.flatMap((o) => caps(o)), ws), RULES_VERSION);
  eq(fillable.feasibility.verdict, "PASS", "all fillable at d=5");
  const thinCaps = os.flatMap((o) => caps(o).map((c) => ({ ...c, asks: { Up: [{ price: 0.6, size: 1 }], Down: [{ price: 0.42, size: 1 }] } })));
  eq(summarize(os, evaluate(os, thinCaps, ws), RULES_VERSION).feasibility.verdict, "FAIL", "nothing fillable");
});

// ---------------------------------------------------------------- stats
Deno.test("cluster bootstrap: dependent rows widen the interval; <2 clusters gives no interval", () => {
  const clustered = Array.from({ length: 40 }, (_, i) => ({ cluster: `w${Math.floor(i / 8)}`, value: Math.floor(i / 8) % 2 ? 1 : -1 }));
  const naive = clustered.map((r, i) => ({ cluster: `u${i}`, value: r.value }));
  const c = clusterBootstrapMean(clustered, { seed: 3 });
  const n = clusterBootstrapMean(naive, { seed: 3 });
  assert(c.upper! - c.lower! > (n.upper! - n.lower!) * 1.5, "clustered interval is wider");
  const one = clusterBootstrapMean([{ cluster: "x", value: 1 }, { cluster: "x", value: 2 }]);
  eq([one.lower, one.upper, one.mean], [null, null, 1.5], "single cluster: point only");
  eq(clusterBootstrapMean([]).mean, null, "empty");
});

// ---------------------------------------------------------------- guards
Deno.test("stop rules: access denied is final, repeated 429 stops, unstable ws stops, first reason wins", () => {
  let t = 0;
  const g = new StopGuard(0, () => t);
  g.onHttpStatus(200);
  for (let i = 0; i < RULES.maxConsecutiveRateLimited - 1; i++) g.onHttpStatus(429);
  eq(g.stopped, null, "below the 429 limit");
  g.onHttpStatus(200);
  for (let i = 0; i < RULES.maxConsecutiveRateLimited - 1; i++) g.onHttpStatus(429);
  eq(g.stopped, null, "a success resets the 429 counter");
  g.onHttpStatus(429);
  eq(g.stopped, "RATE_LIMITED", "five in a row stops");
  g.onHttpStatus(403);
  eq(g.stopped, "RATE_LIMITED", "first reason is never overwritten");
  const g2 = new StopGuard(0, () => t);
  g2.onHttpStatus(451);
  eq(g2.stopped, "ACCESS_DENIED", "geo/access refusal stops at once");
  const g3 = new StopGuard(0, () => t);
  for (let i = 0; i < RULES.wsMaxReconnectsPer10Min; i++) { t += 1000; g3.onWsReconnect(); }
  eq(g3.stopped, null, "five reconnects tolerated");
  t += 1000;
  g3.onWsReconnect();
  eq(g3.stopped, "WS_UNSTABLE", "sixth within 10 minutes stops");
  const g4 = new StopGuard(0, () => RULES.pilotDays * 86_400_000);
  g4.checkDuration();
  eq(g4.stopped, "DURATION_ELAPSED", "duration");
  const g5 = new StopGuard(0, () => 0);
  g5.onIdentity("UNREACHABLE");
  eq(g5.stopped, null, "an outage is not an identity failure");
  g5.onIdentity("WALLET_CHANGED");
  eq(g5.stopped, "IDENTITY_UNVERIFIED", "wallet change stops");
  eq([reconnectDelayMs(0), reconnectDelayMs(3), reconnectDelayMs(20)], [1000, 8000, 60_000], "backoff 1 s -> 60 s");
});

Deno.test("clock skew estimate uses the tightest sample; preflight refuses without approval or sync", () => {
  const s = estimateSkew([{ localSendMs: 0, localRecvMs: 200, serverMs: 500 }, { localSendMs: 1000, localRecvMs: 1020, serverMs: 1170 }])!;
  eq(s, { offsetMs: 160, uncertaintyMs: 10 }, "offset from the 20 ms RTT sample");
  eq(estimateSkew([]), null, "no samples");
  const base = { skew: s, freeDiskBytes: 5 * 1024 ** 3, identity: "VERIFIED", rulesFingerprint: "abc", approvedFingerprint: "abc" };
  assert(evaluatePreflight(base).ok, "all good");
  assert(!evaluatePreflight({ ...base, approvedFingerprint: null }).ok, "no approval file -> refuse");
  assert(!evaluatePreflight({ ...base, approvedFingerprint: "old" }).ok, "approval for other rules -> refuse");
  assert(!evaluatePreflight({ ...base, skew: { offsetMs: 0, uncertaintyMs: 80 } }).ok, "uncertain clock -> refuse");
  assert(!evaluatePreflight({ ...base, freeDiskBytes: 100 * 1024 ** 2 }).ok, "low disk -> refuse");
  assert(!evaluatePreflight({ ...base, identity: "UNREACHABLE" }).ok, "identity must verify at start");
});

// ---------------------------------------------------------------- recorder
Deno.test("recorder writes headed JSONL and refuses to pass its byte cap", async () => {
  const dir = await Deno.makeTempDir();
  const rec = new Recorder({ dir, capBytes: 600, rulesVersion: "v", rulesFingerprint: "fp", now: () => Date.parse("2026-10-20T13:00:00Z") });
  assert(await rec.write("observation", { a: 1 }), "first write");
  assert(await rec.write("observation", { a: 2 }), "second write");
  const rows = await readJsonl(`${dir}/2026-10-20.observation.jsonl`);
  eq(rows.map((r) => r.a), [1, 2], "records without header");
  const text = await Deno.readTextFile(`${dir}/2026-10-20.observation.jsonl`);
  assert(text.split("\n")[0].includes('"rulesFingerprint":"fp"'), "header carries the rules fingerprint");
  let refused = false;
  for (let i = 0; i < 20; i++) if (!(await rec.write("observation", { pad: "x".repeat(40) }))) refused = true;
  assert(refused && rec.bytesWritten <= 600, "cap enforced");
  await Deno.remove(dir, { recursive: true });
});

// ---------------------------------------------------------------- watcher
Deno.test("watcher: first poll only primes, later trades become observations once, others ignored", async () => {
  const W = RULES.wallet;
  const row = (tx: string, slug: string, outcome: string, side = "BUY") => ({
    proxyWallet: W, timestamp: START + 200, conditionId: "0xc", type: "TRADE", size: 10, usdcSize: 5,
    transactionHash: `0x${tx.padStart(64, "0")}`, price: 0.5, asset: "123", side, slug, outcome, name: "NIULAI4",
  });
  let body: unknown[] = [row("1", slug, "Up")];
  let now = (START + 226) * 1000;
  const fetchImpl = (() => Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))) as typeof fetch;
  const w = new ActivityWatcher(W, () => now, { fetchImpl, sleep: () => Promise.resolve() });
  const first = await w.tick();
  eq(first.newObservations.length, 0, "pre-existing trades never become observations");
  body = [row("1", slug, "Up"), row("2", slug, "Down"), row("3", "eth-updown-5m-1792480800", "Up"), row("4", slug, "Up", "SELL")];
  now += 2000;
  const second = await w.tick();
  eq(second.newObservations.map((o) => [o.outcome, o.firstSeenMs - o.tradeTsMs]), [["Down", 28_000]], "only the new BTC 5m BUY, with measured lag");
  eq(second.ignoredOtherMarkets, 2, "ETH and SELL ignored");
  const third = await w.tick();
  eq(third.newObservations.length, 0, "never emitted twice");
  eq(captureTimes(1000).map((c) => c.atMs), [1000, 2000, 6000, 16000, 31000, 61000], "capture schedule");
});

Deno.test("w32tm offsets: median of samples across servers, spread-based uncertainty, needs >= 3 samples", async () => {
  const { parseW32tmOffsets } = await import("./guards.ts");
  const a = "The current time is 20.10.2026 09:09:39.\n09:09:39, +00.2024137s\n09:09:41, +00.2024563s\n09:09:43, +00.2020798s\n";
  const b = "09:09:47, +00.2057290s\n09:09:49, +00.2076877s\n";
  eq(parseW32tmOffsets([a, b]), { offsetMs: 202, uncertaintyMs: 4 }, "two servers agree within a few ms");
  eq(parseW32tmOffsets(["09:00:00, -00.0150000s\n09:00:02, -00.0160000s\n09:00:04, -00.0140000s"]), { offsetMs: -15, uncertaintyMs: 2 }, "negative offsets");
  eq(parseW32tmOffsets(["error 0x800705B4", "09:00:00, +00.1s"]), null, "too few samples");
});
