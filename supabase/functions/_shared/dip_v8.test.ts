import assert from "node:assert/strict";
import {
  calibrate,
  closeLong,
  ENGINE_VERSION,
  evaluatePath,
  initialRuntime,
  parseBars,
  POLICY_VERSION,
  readBook,
  readFlow,
  type Segment,
  sizeLong,
  validateSession,
} from "./dip_v8.ts";
import { marketJson, pricePath } from "./dip_v8_market.ts";
import {
  classifyPivots,
  structure,
} from "../brian-dip-shadow-worker/structure.ts";
const minute = 60_000, base = 1_788_800_040_000;
const bar = (start: number, high = 101, low = 99.5): Segment => ({
  start,
  end: start + minute,
  o: 100,
  h: high,
  l: low,
  c: 100,
  kind: "BAR",
});
const evaluate = (segments: Segment[], more = {}) =>
  evaluatePath({
    direction: "UP",
    target: 103,
    stop: 99,
    start: base,
    due: base + 2 * minute,
    now: base + 3 * minute,
    end: base + minute,
    entry: 100,
    segments,
    ...more,
  });
Deno.test("sealed intrabar target resolves even after price returns to 100", () => {
  assert.equal(evaluate([bar(base, 104)]).reason, "TARGET_FIRST");
});
Deno.test("creation-minute timestamped trades are not skipped", () => {
  const s: Segment = {
    start: base + 20_000,
    end: base + minute,
    o: 100,
    h: 104,
    l: 100,
    c: 104,
    kind: "TRADES",
    points: [{ id: 1, t: base + 25_000, p: 104 }],
  };
  const r = evaluate([s], { start: base + 20_000 });
  assert.equal(r.reason, "TARGET_FIRST");
  assert.equal(r.eventAt, base + 25_000);
});
Deno.test("same bar two barriers is explicitly ambiguous, never a definitive hit", () => {
  const r = evaluate([bar(base, 104, 98)]);
  assert.equal(r.reason, "AMBIGUOUS");
  assert.equal(r.hit, null);
});
Deno.test("future open bar and deadline remainder cannot enter result", () => {
  assert.equal(
    evaluate([bar(base, 104)], { now: base + 20_000 }).reason,
    "INDETERMINATE",
  );
  assert.equal(
    evaluate([bar(base, 104)], { end: base + 20_000, due: base + 20_000 })
      .reason,
    "INDETERMINATE",
  );
});
Deno.test("missing or shuffled price interval is not a loss or win", () => {
  assert.equal(evaluate([bar(base + minute)]).reason, "INDETERMINATE");
  assert.equal(evaluate([]).reason, "INDETERMINATE");
});
Deno.test("stop gap uses observed worse open; favorable gap does not inflate target fill", () => {
  assert.equal(evaluate([{ ...bar(base, 101, 95), o: 95 }]).price, 95);
  assert.equal(evaluate([{ ...bar(base, 105), o: 104 }]).price, 103);
});
Deno.test("expiry uses last available price within sealed deadline", () => {
  const r = evaluate([bar(base)], { due: base + minute });
  assert.equal(r.reason, "EXPIRED_NO_BARRIER");
  assert.equal(r.eventAt, base + minute);
});
Deno.test("partial-minute path is queried exactly through deadline minus one millisecond", async () => {
  const requests: URL[] = [];
  const fetcher = ((url: string | URL | Request) => {
    const u = new URL(String(url));
    requests.push(u);
    return Promise.resolve(
      Response.json([{ a: 1, T: base + 25_000, p: "100" }]),
    );
  }) as typeof fetch;
  const p = await pricePath(
    base + 20_000,
    base + 40_000,
    base + 3 * minute,
    100,
    fetcher,
  );
  assert.equal(
    requests[0].searchParams.get("startTime"),
    String(base + 20_000),
  );
  assert.equal(requests[0].searchParams.get("endTime"), String(base + 39_999));
  assert.equal(p.segments.length, 1);
  assert.equal(p.end, base + 40_000);
});
Deno.test("market adapter cannot call a private order or unknown route even with an injected fetch", async () => {
  let calls = 0;
  const f = (() => {
    calls++;
    return Promise.resolve(Response.json({}));
  }) as typeof fetch;
  await assert.rejects(() => marketJson("/api/v3/order", f), /FORBIDDEN/);
  await assert.rejects(() => marketJson("/fapi/v1/order", f), /FORBIDDEN/);
  assert.equal(calls, 0);
});
Deno.test("minimum notional cannot exceed the cold-start cap", () => {
  const rules = {
    minNotional: 10,
    minQty: .0001,
    maxQty: 10000,
    stepSize: .0001,
  };
  assert.equal(sizeLong(50, 100, 99, calibrate([]), 10, 1, rules), null);
  const x = sizeLong(1000, 100, 99, calibrate([]), 10, 1, rules)!;
  assert.ok(x.notional <= 80);
  assert.equal(x.actual_fraction, x.notional / 1000);
  assert.equal(sizeLong(1000, 100, 100.5, calibrate([]), 10, 1, rules), null);
});
Deno.test("forty losses produce zero measured probability; ambiguous labels do not count as losses", () => {
  const c = calibrate([...Array.from({ length: 40 }, () => ({ hit: false })), {
    hit: null,
  }]);
  assert.equal(c.p, 0);
  assert.equal(c.samples, 40);
  assert.equal(c.ambiguous, 1);
  assert.equal(calibrate([], true).unavailable, true);
});
Deno.test("V8 config fails closed on real mode, SHORT, browser execution and non-ETH", () => {
  const cfg = {
    engine_version: ENGINE_VERSION,
    policy_version: POLICY_VERSION,
    symbols: ["ETHUSDT"],
    shadow_only: true,
    live_execution: false,
    browser_execution: false,
    server_authoritative: true,
    allow_shadow_short: false,
    max_shadow_leverage: 1,
    execution_mode: "SHADOW_PAPER",
  };
  validateSession(cfg);
  for (
    const patch of [
      { live_execution: true },
      { shadow_only: false },
      { allow_shadow_short: true },
      { browser_execution: true },
      { symbols: ["XRPUSDT", "ETHUSDT"] },
      { engine_version: "v7" },
    ]
  ) assert.throws(() => validateSession({ ...cfg, ...patch }));
});
Deno.test("closed-candle timestamp, warm-up, ordering and trade/book freshness are checked", () => {
  const rows = Array.from(
    { length: 61 },
    (
      _,
      i,
    ) => [
      base + i * minute,
      "100",
      "101",
      "99",
      "100",
      "1",
      base + (i + 1) * minute - 1,
    ],
  );
  assert.equal(parseBars(rows, "1m", base + 60 * minute + 20_000).length, 60);
  assert.throws(
    () => parseBars(rows.slice(0, 10), "1m", base + 60 * minute),
    /WARMUP/,
  );
  const shuffled = [...rows];
  [shuffled[20], shuffled[21]] = [shuffled[21], shuffled[20]];
  assert.throws(
    () => parseBars(shuffled, "1m", base + 60 * minute),
    /GAP_OR_ORDER/,
  );
  assert.throws(
    () => readBook({ bids: [[101, 1]], asks: [[100, 1]] }, base),
    /CROSSED/,
  );
  assert.throws(
    () =>
      readFlow(
        Array.from(
          { length: 10 },
          (_, i) => ({ a: i, T: base, p: 100, q: 1, m: false }),
        ),
        base + 60_000,
      ),
    /STALE/,
  );
});
Deno.test("confirmed pivot requires three right closed candles and preserves native structure", async () => {
  const rows = [1, 2, 3, 6, 3, 2, 1].map((h, i) => ({
    t: i * minute,
    ct: (i + 1) * minute - 1,
    o: 1,
    h,
    l: .5,
    c: 1,
    v: 1,
  }));
  assert.equal(
    classifyPivots(rows.slice(0, 6)).filter((p) => p.kind === "H").length,
    0,
  );
  assert.equal(classifyPivots(rows).filter((p) => p.kind === "H").length, 1);
  assert.equal((await structure("1m", rows)).lastHigh?.p, 6);
});
Deno.test("close uses frozen position fees, balanced cash, one terminal transition and ambiguity flag", () => {
  const rt = initialRuntime(1000);
  rt.cash = 899.9;
  rt.pos = {
    side: "LONG",
    position_id: "p",
    thesis_id: "t",
    episode_id: "e",
    setup: "SWEEP_RECLAIM",
    regime: "RANGE",
    entry: 100,
    qty: 1,
    notional: 100,
    target: 103,
    stop: 99,
    opened_at: new Date(base).toISOString(),
    due_at: new Date(base + 2 * minute).toISOString(),
    fees_open: .1,
    fee_bps: 10,
    slippage_bps: 1,
    spread_bps: 2,
    venue: "SPOT",
    policy_version: POLICY_VERSION,
    checked_until: base,
    market_price: 100,
    actual_fraction: .1,
  };
  const event = closeLong(
    rt,
    evaluate([bar(base, 104, 98)]),
    "fp",
    base,
    base + minute,
  )!;
  assert.equal(rt.pos, null);
  assert.equal(rt.trades, 1);
  assert.equal(rt.lastLock?.episode_id, "e");
  assert.ok(Math.abs(rt.cash - rt.start - rt.realized) < 1e-9);
  assert.equal(
    (event.metadata as Record<string, unknown>).exit_reason,
    "AMBIGUOUS_CONSERVATIVE_STOP",
  );
  assert.equal(
    closeLong(rt, evaluate([bar(base, 104)]), "fp", base, base + minute),
    null,
  );
});
