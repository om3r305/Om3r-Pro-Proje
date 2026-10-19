import {
  collectPredictionCycle,
  DEFAULT_COLLECTOR_CONFIG,
  type PredictionCollectorConfig,
  selectMarkets,
  toSnapshotRow,
} from "./prediction_collector.ts";
import { dataActivityUrl, gammaEventsKeysetUrl, normalizeGammaMarket, type PredictionMarket } from "./prediction_market.ts";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(`assertion failed: ${msg}`);
}
function eq<T>(a: T, b: T, msg: string) {
  if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${msg}: expected ${JSON.stringify(b)} got ${JSON.stringify(a)}`);
}

const NOW = Date.parse("2026-10-19T19:30:00.000Z");
const WALLET = "0x44832d0d2ec11187c1e77d786feb15f6a50254c6";
const noSleep = () => Promise.resolve();

function rawMarket(id: string, over: Record<string, unknown> = {}) {
  return {
    id,
    question: `Question ${id}?`,
    conditionId: `0x${id.padStart(64, "0")}`,
    outcomes: '["Yes","No"]',
    outcomePrices: '["0.4","0.6"]',
    clobTokenIds: `["${id}1","${id}2"]`,
    liquidityNum: 50_000,
    volume24hr: 1000,
    active: true,
    closed: false,
    enableOrderBook: true,
    updatedAt: "2026-10-19T19:20:00Z",
    ...over,
  };
}

function trade(i: number, ts: number) {
  return {
    proxyWallet: WALLET,
    timestamp: ts,
    conditionId: "0xd4ca8d49b356d404cd01aef74285132ea95012e31dc4c92d863704c035058d22",
    type: "TRADE",
    size: 10 + i,
    usdcSize: 1,
    transactionHash: `0x${i.toString(16).padStart(64, "a")}`,
    price: 0.5,
    asset: "457",
    side: "BUY",
    slug: "btc-updown-5m",
    outcome: "Up",
    name: "NIULAI4",
  };
}

interface FakeRoutes {
  events?: (tag: string) => { status: number; body?: unknown };
  search?: () => { status: number; body?: unknown };
  probe?: () => { status: number; body?: unknown };
  activity?: (start: number, offset: number, limit: number) => { status: number; body?: unknown };
}

function fakeFetch(routes: FakeRoutes, log: string[]) {
  return ((input: string | URL | Request) => {
    const url = new URL(String(input));
    log.push(url.pathname + url.search);
    let r: { status: number; body?: unknown } = { status: 404 };
    if (url.pathname === "/events/keyset") r = routes.events?.(url.searchParams.get("tag_slug")!) ?? r;
    else if (url.pathname === "/public-search") r = routes.search?.() ?? r;
    else if (url.pathname === "/activity" && url.searchParams.get("type") === "TRADE") {
      r = routes.activity?.(Number(url.searchParams.get("start")), Number(url.searchParams.get("offset")), Number(url.searchParams.get("limit"))) ?? r;
    } else if (url.pathname === "/activity") r = routes.probe?.() ?? r;
    return Promise.resolve(new Response(r.body === undefined ? "" : JSON.stringify(r.body), { status: r.status }));
  }) as typeof fetch;
}

function cfg(over: Partial<PredictionCollectorConfig> = {}): PredictionCollectorConfig {
  return { ...DEFAULT_COLLECTOR_CONFIG, tagSlugs: ["fed"], ...over };
}

const verifiedRoutes: FakeRoutes = {
  search: () => ({ status: 200, body: { profiles: [{ name: "NIULAI4", proxyWallet: WALLET }, { name: "Niulai", proxyWallet: "0x3ef47f3553283b1857c023637d35636fcb9ffb21" }] } }),
  probe: () => ({ status: 200, body: [trade(0, 1792436684)] }),
};

Deno.test("events url validates tags and caps limit; activity url resumes by time, oldest first", () => {
  const u = new URL(gammaEventsKeysetUrl({ tagSlug: "fed", limit: 1000, active: true, closed: false }));
  eq(u.pathname, "/events/keyset", "events keyset path");
  eq(u.searchParams.get("tag_slug"), "fed", "tag");
  eq(u.searchParams.get("limit"), "100", "limit clamp");
  let threw = false;
  try {
    gammaEventsKeysetUrl({ tagSlug: "fed&x=1", limit: 1 });
  } catch {
    threw = true;
  }
  assert(threw, "tag injection refused");
  const a = new URL(dataActivityUrl({ wallet: WALLET, startSeconds: 100, limit: 500, offset: 500 }));
  eq(a.searchParams.get("sortDirection"), "ASC", "oldest first");
  eq(a.searchParams.get("start"), "100", "time resume");
  eq(a.searchParams.get("type"), "TRADE", "trades only");
  for (const bad of [
    () => dataActivityUrl({ wallet: "0x4483", startSeconds: 1, limit: 1, offset: 0 }),
    () => dataActivityUrl({ wallet: WALLET, startSeconds: -1, limit: 1, offset: 0 }),
    () => dataActivityUrl({ wallet: WALLET, startSeconds: 1, limit: 500, offset: 4600 }),
  ]) {
    let t = false;
    try {
      bad();
    } catch {
      t = true;
    }
    assert(t, "invalid activity request refused before any network call");
  }
});

Deno.test("selection: open, book-enabled, KNOWN liquidity above floor, de-duplicated, by volume, bounded", () => {
  const rx = "2026-10-19T19:30:00.000Z";
  const m = (id: string, over: Record<string, unknown> = {}) => {
    const r = normalizeGammaMarket(rawMarket(id, over), rx);
    if (!r.ok) throw new Error(r.reason);
    return r.value;
  };
  const markets: PredictionMarket[] = [
    m("1", { volume24hr: 10 }),
    m("2", { volume24hr: 500 }),
    m("2", { volume24hr: 9999 }), // duplicate id from a second tag: first one wins
    m("3", { liquidityNum: undefined, liquidity: undefined }), // unknown liquidity never passes
    m("4", { liquidityNum: 9_999 }),
    m("5", { closed: true }),
    m("6", { enableOrderBook: false }),
    m("7", { volume24hr: undefined }),
    m("8", { volume24hr: 500 }),
  ];
  eq(selectMarkets(markets, 10, 10_000).map((x) => x.marketId), ["2", "8", "1", "7"], "order and filters");
  eq(selectMarkets(markets, 2, 10_000).map((x) => x.marketId), ["2", "8"], "bounded");
});

Deno.test("snapshot ids are stable for an unchanged market and freshness is measured, not invented", () => {
  const a = normalizeGammaMarket(rawMarket("9"), "2026-10-19T19:30:00.000Z");
  const b = normalizeGammaMarket(rawMarket("9"), "2026-10-19T19:45:00.000Z");
  assert(a.ok && b.ok, "normalize");
  eq(toSnapshotRow(a.value).snapshot_id, toSnapshotRow(b.value).snapshot_id, "same source update -> same snapshot");
  eq(toSnapshotRow(a.value).source_freshness_seconds, 600, "freshness from source time");
  const noTime = normalizeGammaMarket(rawMarket("9", { updatedAt: undefined }), "2026-10-19T19:30:12.000Z");
  assert(noTime.ok, "normalize");
  const row = toSnapshotRow(noTime.value);
  eq(row.source_freshness_seconds, null, "unknown source time -> unknown freshness");
  assert(row.snapshot_id.includes("|rx:2026-10-19T19:30"), "receipt-minute key is explicitly marked");
});

Deno.test("cycle without wallet tracking: bounded requests, rows built, no identity calls", async () => {
  const log: string[] = [];
  const batch = await collectPredictionCycle(cfg({ tagSlugs: ["fed", "cpi", "fed"] }), null, {
    fetchImpl: fakeFetch({
      events: (tag) => ({
        status: 200,
        body: { events: [{ id: `e-${tag}`, title: `Event ${tag}`, markets: [rawMarket(tag === "fed" ? "1" : "2"), { id: "bad" }] }] },
      }),
    }, log),
    sleep: noSleep,
    now: () => NOW,
  });
  eq(log.length, 2, "duplicate tag requested once; no identity or activity calls");
  eq(batch.markets.map((m) => m.market_key).sort(), ["polymarket:1", "polymarket:2"], "markets");
  eq(batch.markets.find((m) => m.market_id === "1")?.event_id, "e-fed", "event context kept");
  eq(batch.snapshots.length, 2, "one snapshot per kept market");
  eq(batch.trades.length, 0, "no trades");
  eq(batch.wallet, null, "no wallet update");
  eq(batch.summary.marketsRejected, 2, "malformed markets counted");
  eq(batch.health.map((h) => h.status), ["DEGRADED", "DEGRADED"], "rejections surface as DEGRADED health");
  eq(new Set(batch.health.map((h) => h.health_id)).size, batch.health.length, "health ids unique");
  eq(batch.summary.identity, "SKIPPED", "identity skipped");
});

Deno.test("a blocked tag is reported and does not stop other tags", async () => {
  const batch = await collectPredictionCycle(cfg({ tagSlugs: ["fed", "cpi"] }), null, {
    fetchImpl: fakeFetch({
      events: (tag) => tag === "fed" ? { status: 451 } : { status: 200, body: { events: [{ id: "e", markets: [rawMarket("2")] }] } },
    }, []),
    sleep: noSleep,
    now: () => NOW,
  });
  eq(batch.health.map((h) => h.status), ["BLOCKED", "OK"], "per-source status");
  eq(batch.markets.length, 1, "other tag still collected");
});

Deno.test("wallet tracking: verified identity, backfill from the 24h floor, watermark = newest trade read", async () => {
  const starts: number[] = [];
  const batch = await collectPredictionCycle(cfg({ walletTracking: true }), null, {
    fetchImpl: fakeFetch({
      events: () => ({ status: 200, body: { events: [] } }),
      ...verifiedRoutes,
      activity: (start) => {
        starts.push(start);
        return { status: 200, body: [trade(1, 1792436000), trade(2, 1792436684), trade(2, 1792436684), { type: "TRADE", side: "HOLD" }] };
      },
    }, []),
    sleep: noSleep,
    now: () => NOW,
  });
  eq(starts, [Math.floor((NOW - 24 * 3_600_000) / 1000)], "first run backfills a bounded 24h window");
  eq(batch.summary.identity, "VERIFIED", "identity");
  eq(batch.trades.length, 2, "duplicates collapsed, malformed rejected");
  eq(batch.wallet?.status, "VERIFIED", "wallet verified");
  eq(batch.wallet?.watermark, new Date(1792436684 * 1000).toISOString(), "watermark");
  assert(batch.health.some((h) => h.source === "polymarket-data-activity-coverage" && h.status === "DEGRADED"), "rejected row surfaces in coverage health");
});

Deno.test("wallet tracking resumes inclusively from the stored watermark", async () => {
  const starts: number[] = [];
  await collectPredictionCycle(cfg({ walletTracking: true }), { wallet: WALLET, watermark: "2026-10-19T19:04:44.000Z" }, {
    fetchImpl: fakeFetch({
      events: () => ({ status: 200, body: { events: [] } }),
      ...verifiedRoutes,
      activity: (start) => {
        starts.push(start);
        return { status: 200, body: [] };
      },
    }, []),
    sleep: noSleep,
    now: () => NOW,
  });
  eq(starts, [Math.floor(Date.parse("2026-10-19T19:04:44.000Z") / 1000)], "inclusive resume second");
});

Deno.test("a full activity window is paged, bounded, and reported as truncated", async () => {
  const offsets: number[] = [];
  const batch = await collectPredictionCycle(cfg({ walletTracking: true, activityPageSize: 2, maxActivityPages: 3 }), null, {
    fetchImpl: fakeFetch({
      events: () => ({ status: 200, body: { events: [] } }),
      ...verifiedRoutes,
      activity: (_s, offset) => {
        offsets.push(offset);
        return { status: 200, body: [trade(offset, 1792430000 + offset), trade(offset + 1, 1792430000 + offset + 1)] };
      },
    }, []),
    sleep: noSleep,
    now: () => NOW,
  });
  eq(offsets, [0, 2, 4], "pages bounded by maxActivityPages");
  eq(batch.trades.length, 6, "all read trades kept");
  assert(batch.summary.activityTruncated, "truncation flagged");
  eq(batch.wallet?.watermark, new Date((1792430000 + 5) * 1000).toISOString(), "watermark advances to what was read");
});

Deno.test("a failed activity page never advances the watermark", async () => {
  let call = 0;
  const batch = await collectPredictionCycle(cfg({ walletTracking: true, activityPageSize: 2, maxActivityPages: 3 }), null, {
    fetchImpl: fakeFetch({
      events: () => ({ status: 200, body: { events: [] } }),
      ...verifiedRoutes,
      activity: () => (++call === 1 ? { status: 200, body: [trade(1, 1792430001), trade(2, 1792430002)] } : { status: 503 }),
    }, []),
    sleep: noSleep,
    now: () => NOW,
    maxAttempts: 1,
  });
  eq(batch.wallet?.watermark, null, "no watermark after a gap");
  eq(batch.wallet?.status, "VERIFIED", "identity unaffected");
});

Deno.test("identity failure downgrades a known wallet; an outage does not", async () => {
  const known = { wallet: WALLET, watermark: "2026-10-19T19:00:00.000Z" };
  const lookalike = await collectPredictionCycle(cfg({ walletTracking: true }), known, {
    fetchImpl: fakeFetch({
      events: () => ({ status: 200, body: { events: [] } }),
      search: () => ({ status: 200, body: { profiles: [{ name: "niulai4", proxyWallet: WALLET }] } }),
    }, []),
    sleep: noSleep,
    now: () => NOW,
  });
  eq(lookalike.wallet?.status, "UNVERIFIED", "downgraded");
  eq(lookalike.trades.length, 0, "no trades collected for an unverified identity");

  const log: string[] = [];
  const outage = await collectPredictionCycle(cfg({ walletTracking: true }), known, {
    fetchImpl: fakeFetch({ events: () => ({ status: 200, body: { events: [] } }), search: () => ({ status: 503 }) }, log),
    sleep: noSleep,
    now: () => NOW,
    maxAttempts: 1,
  });
  eq(outage.wallet, null, "source outage is not evidence against the identity");
  eq(outage.summary.identity, "UNREACHABLE", "reported as unreachable");
  assert(!log.some((l) => l.includes("type=TRADE")), "no trade fetch without verification");
});

Deno.test("no identity guessing: no exact candidate means no probe and no wallet row", async () => {
  const log: string[] = [];
  const batch = await collectPredictionCycle(cfg({ walletTracking: true }), null, {
    fetchImpl: fakeFetch({
      events: () => ({ status: 200, body: { events: [] } }),
      search: () => ({ status: 200, body: { profiles: [{ name: "Niulai", proxyWallet: WALLET }] } }),
    }, log),
    sleep: noSleep,
    now: () => NOW,
  });
  eq(batch.wallet, null, "nothing to downgrade, nothing to store");
  eq(batch.summary.identity, "UNVERIFIED", "unverified");
  assert(!log.some((l) => l.startsWith("/activity")), "never probes a non-matching wallet");
});

Deno.test("a cycle never produces more trades than one batch RPC accepts", async () => {
  const offsets: number[] = [];
  const batch = await collectPredictionCycle(cfg({ walletTracking: true, activityPageSize: 500, maxActivityPages: 9 }), null, {
    fetchImpl: fakeFetch({
      events: () => ({ status: 200, body: { events: [] } }),
      ...verifiedRoutes,
      activity: (_s, offset, limit) => {
        offsets.push(offset);
        return { status: 200, body: Array.from({ length: limit }, (_, i) => trade(offset + i, 1792400000 + offset + i)) };
      },
    }, []),
    sleep: noSleep,
    now: () => NOW,
  });
  eq(offsets, [0, 500], "pages clamped to the 1000-trade RPC bound");
  eq(batch.trades.length, 1000, "bounded");
  assert(batch.summary.activityTruncated, "remaining trades flagged for the next cycle");
});

Deno.test("resolution capture: bounded, keeps only genuine resolutions, reports the rest as pending", async () => {
  const { collectResolutions, MAX_RESOLUTION_LOOKUPS } = await import("./prediction_collector.ts");
  const bodies: Record<string, unknown> = {
    "1": { id: "1", closed: true, umaResolutionStatus: "resolved", outcomes: '["Yes","No"]', outcomePrices: '["0","1"]', closedTime: "2026-10-29 05:00:00+00" },
    "2": { id: "2", closed: false, umaResolutionStatus: null, outcomes: '["Yes","No"]', outcomePrices: '["0.4","0.6"]' },
    "3": { id: "3", closed: true, umaResolutionStatus: "disputed", outcomes: '["Yes","No"]', outcomePrices: '["1","0"]' },
  };
  const seen: string[] = [];
  const fetchImpl = ((input: string | URL | Request) => {
    const id = new URL(String(input)).pathname.split("/").pop()!;
    seen.push(id);
    if (id === "4") return Promise.resolve(new Response("", { status: 503 }));
    return Promise.resolve(new Response(JSON.stringify(bodies[id]), { status: 200 }));
  }) as typeof fetch;
  const out = await collectResolutions(["1", "2", "3", "4", "1", "x/../y"], { fetchImpl, sleep: noSleep, now: () => NOW, maxAttempts: 1 });
  eq(seen, ["1", "2", "3", "4"], "deduplicated, invalid ids never requested");
  eq(out.rows.map((r) => [r.market_key, r.yes_outcome]), [["polymarket:1", 0]], "only the genuine resolution is kept");
  eq(out.pending.map((p) => p.reason), ["NOT_CLOSED", "UMA_STATUS_disputed", "FETCH_ERROR"], "the rest stay pending with a reason");
  eq(new Set(out.health.map((h) => h.health_id)).size, out.health.length, "health ids unique");
  const many = await collectResolutions(Array.from({ length: 50 }, (_, i) => String(100 + i)), {
    fetchImpl: (() => Promise.resolve(new Response(JSON.stringify(bodies["2"]), { status: 200 }))) as typeof fetch,
    sleep: noSleep, now: () => NOW,
  });
  eq(many.requests, MAX_RESOLUTION_LOOKUPS, "lookups per cycle are bounded");
});
