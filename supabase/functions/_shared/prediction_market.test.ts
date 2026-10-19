import {
  assertAllowedPredictionUrl,
  fetchPredictionJson,
  gammaMarketsKeysetUrl,
  iterateGammaMarkets,
  normalizeGammaMarket,
  normalizeWalletActivityRow,
  parseRetryAfterMs,
  partitionWalletActivity,
  PREDICTION_SAFETY,
  verifyProfileIdentity,
} from "./prediction_market.ts";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(`assertion failed: ${msg}`);
}
function eq<T>(a: T, b: T, msg: string) {
  if (JSON.stringify(a) !== JSON.stringify(b)) {
    throw new Error(`${msg}: expected ${JSON.stringify(b)} got ${JSON.stringify(a)}`);
  }
}

const RECEIVED = "2026-10-19T19:30:00.000Z";
const WALLET = "0x44832d0d2ec11187c1e77d786feb15f6a50254c6";
const TX = "0xd8134735ad1bc7596ff05a87bb19b4d3766b7981d69f35cebf30af8fb3d581dc";

// Shapes below are trimmed copies of real public API responses captured 2026-10-19.
const gammaMarket = {
  id: "2589810",
  question: "Will the Fed decrease interest rates by 50+ bps after the October 2026 meeting?",
  conditionId: "0xa69ef420d8b4075ad8ba8611d02c63cb2ece46f5c8c643af223c272689b0c96f",
  slug: "fed-decreases-interest-rates-by-50-bps-after-october-2026-meeting",
  outcomes: '["Yes", "No"]',
  outcomePrices: '["0.0005", "0.9995"]',
  clobTokenIds: '["1701", "6172"]',
  endDate: "2026-10-29T03:59:00Z",
  liquidityNum: 1188890.47367,
  volume24hr: 1249317.17,
  bestBid: null,
  bestAsk: 0.001,
  spread: 0.001,
  updatedAt: "2026-10-19T19:18:46.966071Z",
  active: true,
  closed: false,
  enableOrderBook: true,
};

const tradeRow = {
  proxyWallet: WALLET,
  timestamp: 1792436684,
  conditionId: "0xd4ca8d49b356d404cd01aef74285132ea95012e31dc4c92d863704c035058d22",
  type: "TRADE",
  size: 10,
  usdcSize: 0.7,
  transactionHash: TX,
  price: 0.07,
  asset: "45780744039970711522644952415639884331285827806483628311008451591801847762772",
  side: "BUY",
  outcomeIndex: 0,
  title: "Bitcoin Up or Down - October 19, 3:00PM-3:05PM ET",
  slug: "btc-updown-5m-1792436400",
  outcome: "Up",
  name: "NIULAI4",
  pseudonym: "Utter-Paramecium",
};

const redeemRow = { ...tradeRow, type: "REDEEM", side: "", price: 0, usdcSize: 322.86522 };

function response(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), { status, headers });
}
const noSleep = () => Promise.resolve();

Deno.test("url allow-list refuses non-polymarket hosts, http, credentials and garbage", () => {
  assertAllowedPredictionUrl("https://gamma-api.polymarket.com/markets/keyset");
  assertAllowedPredictionUrl("https://clob.polymarket.com/book?token_id=1");
  for (const bad of [
    "http://gamma-api.polymarket.com/markets",
    "https://evil.example.com/markets",
    "https://polymarket.com.evil.example.com/x",
    "https://user:pw@gamma-api.polymarket.com/x",
    "not a url",
  ]) {
    let threw = false;
    try {
      assertAllowedPredictionUrl(bad);
    } catch {
      threw = true;
    }
    assert(threw, `must refuse ${bad}`);
  }
});

Deno.test("Retry-After parses seconds and HTTP dates, rejects junk", () => {
  eq(parseRetryAfterMs("3", 0), 3000, "seconds");
  eq(parseRetryAfterMs("1.5", 0), 1500, "fractional seconds");
  eq(parseRetryAfterMs(null, 0), null, "absent");
  eq(parseRetryAfterMs("soon", 0), null, "junk");
  const now = Date.parse("2026-10-19T19:00:00Z");
  eq(parseRetryAfterMs("Mon, 19 Oct 2026 19:00:10 GMT", now), 10_000, "http date");
  eq(parseRetryAfterMs("Mon, 19 Oct 2026 18:00:00 GMT", now), 0, "past date clamps to zero");
});

Deno.test("429 is retried honouring Retry-After, then succeeds", async () => {
  const sleeps: number[] = [];
  let calls = 0;
  const result = await fetchPredictionJson("https://gamma-api.polymarket.com/markets/keyset", {
    fetchImpl: () => {
      calls++;
      return Promise.resolve(calls < 3 ? response(429, "", { "retry-after": "7" }) : response(200, { ok: 1 }));
    },
    sleep: (ms) => {
      sleeps.push(ms);
      return Promise.resolve();
    },
    now: () => Date.parse(RECEIVED),
  });
  eq(result.status, "OK", "status");
  eq(result.attempts, 3, "attempts");
  eq(sleeps, [7000, 7000], "Retry-After wins over the 500/1000ms backoff");
  eq(result.receivedAt, RECEIVED, "receivedAt comes from the injected clock");
});

Deno.test("persistent 5xx exhausts attempts and reports ERROR with the last HTTP status", async () => {
  let calls = 0;
  const result = await fetchPredictionJson("https://data-api.polymarket.com/activity", {
    fetchImpl: () => {
      calls++;
      return Promise.resolve(response(503, ""));
    },
    sleep: noSleep,
    maxAttempts: 3,
  });
  eq(calls, 3, "calls");
  eq(result.status, "ERROR", "status");
  eq(result.httpStatus, 503, "http status");
  eq(result.body, null, "no body is invented");
});

Deno.test("403/451 means BLOCKED: no retry, no circumvention", async () => {
  for (const code of [403, 451, 401]) {
    let calls = 0;
    const result = await fetchPredictionJson("https://clob.polymarket.com/book?token_id=1", {
      fetchImpl: () => {
        calls++;
        return Promise.resolve(response(code, ""));
      },
      sleep: noSleep,
    });
    eq(calls, 1, `calls for ${code}`);
    eq(result.status, "BLOCKED", `status for ${code}`);
  }
});

Deno.test("network errors and invalid JSON are reported, never thrown", async () => {
  const net = await fetchPredictionJson("https://clob.polymarket.com/time", {
    fetchImpl: () => Promise.reject(new Error("boom")),
    sleep: noSleep,
    maxAttempts: 2,
  });
  eq(net.status, "ERROR", "network status");
  eq(net.detail, "boom", "network detail");
  const bad = await fetchPredictionJson("https://clob.polymarket.com/time", {
    fetchImpl: () => Promise.resolve(response(200, "<html>not json</html>")),
    sleep: noSleep,
  });
  eq(bad.status, "ERROR", "invalid json status");
  eq(bad.body, null, "invalid json body");
});

Deno.test("fetch refuses a disallowed url before any request is made", async () => {
  let calls = 0;
  let threw = false;
  try {
    await fetchPredictionJson("https://evil.example.com/x", {
      fetchImpl: () => {
        calls++;
        return Promise.resolve(response(200, {}));
      },
    });
  } catch {
    threw = true;
  }
  assert(threw && calls === 0, "disallowed host must throw with zero requests");
});

Deno.test("gamma market normalizes JSON-string arrays and keeps missing values null, not zero", () => {
  const r = normalizeGammaMarket(gammaMarket, RECEIVED);
  assert(r.ok, "market should normalize");
  const m = r.value;
  eq(m.marketId, "2589810", "id");
  eq(m.outcomes.map((o) => o.label), ["Yes", "No"], "labels");
  eq(m.outcomes.map((o) => o.impliedProbability), [0.0005, 0.9995], "probabilities");
  eq(m.outcomes.map((o) => o.clobTokenId), ["1701", "6172"], "token ids");
  eq(m.bestBid, null, "absent bestBid stays null, not 0");
  eq(m.bestAsk, 0.001, "bestAsk");
  eq(m.endDate, "2026-10-29T03:59:00.000Z", "endDate iso");
  eq(m.evidenceClass, "OBSERVED_VERIFIED", "evidence class");
  eq(m.safety, PREDICTION_SAFETY, "safety envelope");
  assert(m.safety.liveExecution === false && m.safety.canChangeLiveDecisions === false, "no live authority");
});

Deno.test("gamma market: out-of-range or missing prices become null; length mismatches are rejected", () => {
  const noPrices = normalizeGammaMarket({ ...gammaMarket, outcomePrices: undefined }, RECEIVED);
  assert(noPrices.ok, "missing prices still normalizes");
  eq(noPrices.value.outcomes.map((o) => o.impliedProbability), [null, null], "missing prices are null");
  const bad = normalizeGammaMarket({ ...gammaMarket, outcomePrices: '["1.7", ""]' }, RECEIVED);
  assert(bad.ok, "bad prices still normalizes");
  eq(bad.value.outcomes.map((o) => o.impliedProbability), [null, null], "1.7 and empty are not probabilities");
  assert(!normalizeGammaMarket({ ...gammaMarket, outcomePrices: '["0.5"]' }, RECEIVED).ok, "length mismatch rejected");
  assert(!normalizeGammaMarket({ ...gammaMarket, clobTokenIds: '["1"]' }, RECEIVED).ok, "token mismatch rejected");
  assert(!normalizeGammaMarket({ ...gammaMarket, outcomes: "not json" }, RECEIVED).ok, "bad outcomes rejected");
  assert(!normalizeGammaMarket({ ...gammaMarket, id: undefined }, RECEIVED).ok, "no id rejected");
  assert(!normalizeGammaMarket({ ...gammaMarket, question: "  " }, RECEIVED).ok, "blank question rejected");
  assert(!normalizeGammaMarket(null, RECEIVED).ok, "null rejected");
});

Deno.test("keyset url uses after_cursor (the legacy `cursor` name is silently ignored by the API)", () => {
  const first = new URL(gammaMarketsKeysetUrl({ limit: 100, active: true, closed: false }));
  eq(first.pathname, "/markets/keyset", "path is the non-deprecated keyset endpoint");
  eq(first.searchParams.get("after_cursor"), null, "no cursor on page one");
  const next = new URL(gammaMarketsKeysetUrl({ limit: 9999, afterCursor: "abc" }));
  eq(next.searchParams.get("after_cursor"), "abc", "after_cursor");
  eq(next.searchParams.get("cursor"), null, "never the ignored name");
  eq(next.searchParams.get("limit"), "500", "limit is clamped");
});

Deno.test("market iteration follows next_cursor, stops at the end, and refuses a repeated cursor", async () => {
  const pages: Record<string, unknown> = {
    "": { markets: [{ id: "1" }], next_cursor: "c1" },
    c1: { markets: [{ id: "2" }], next_cursor: "c2" },
    c2: { markets: [{ id: "3" }] },
  };
  const requested: (string | null)[] = [];
  const fetchImpl = (input: string | URL | Request) => {
    const cursor = new URL(String(input)).searchParams.get("after_cursor");
    requested.push(cursor);
    return Promise.resolve(response(200, pages[cursor ?? ""]));
  };
  const ids: string[] = [];
  for await (const page of iterateGammaMarkets({ limit: 1, maxPages: 10 }, { fetchImpl: fetchImpl as typeof fetch, sleep: noSleep })) {
    for (const m of page.markets) ids.push((m as { id: string }).id);
  }
  eq(ids, ["1", "2", "3"], "all pages read in order");
  eq(requested, [null, "c1", "c2"], "cursor chain");

  // A server that keeps returning the same cursor must not loop forever.
  let loopCalls = 0;
  const looping = () => {
    loopCalls++;
    return Promise.resolve(response(200, { markets: [{ id: "x" }], next_cursor: "same" }));
  };
  for await (const _ of iterateGammaMarkets({ limit: 1, maxPages: 50 }, { fetchImpl: looping as typeof fetch, sleep: noSleep })) {
    // drain
  }
  assert(loopCalls <= 3, `repeated cursor must stop iteration (calls=${loopCalls})`);

  // maxPages bounds a long chain, and a failed page ends iteration without inventing markets.
  let bounded = 0;
  const endless = () => {
    bounded++;
    return Promise.resolve(response(200, { markets: [], next_cursor: `c${bounded}` }));
  };
  for await (const _ of iterateGammaMarkets({ limit: 1, maxPages: 4 }, { fetchImpl: endless as typeof fetch, sleep: noSleep })) {
    // drain
  }
  eq(bounded, 4, "maxPages bound");
  const failedPages: string[] = [];
  for await (
    const page of iterateGammaMarkets({ limit: 1, maxPages: 5 }, {
      fetchImpl: (() => Promise.resolve(response(403, ""))) as typeof fetch,
      sleep: noSleep,
    })
  ) failedPages.push(page.fetch.status);
  eq(failedPages, ["BLOCKED"], "blocked page yields once and stops");
});

Deno.test("wallet trade row normalizes with a stable idempotency key", () => {
  const a = normalizeWalletActivityRow(tradeRow, RECEIVED);
  const b = normalizeWalletActivityRow({ ...tradeRow, proxyWallet: WALLET.toUpperCase().replace("0X", "0x") }, "2026-10-19T20:00:00.000Z");
  assert(a.ok && b.ok, "both normalize");
  eq(a.value.idempotencyKey, b.value.idempotencyKey, "re-ingest and address case do not change the key");
  eq(a.value.tradedAt, "2026-10-19T19:04:44.000Z", "epoch seconds converted");
  eq(a.value.side, "BUY", "side");
  eq(a.value.price, 0.07, "price");
  eq(a.value.evidenceClass, "OBSERVED_VERIFIED", "observed, not hypothesis");
  const otherFill = normalizeWalletActivityRow({ ...tradeRow, size: 150, price: 0.06 }, RECEIVED);
  assert(otherFill.ok, "second fill normalizes");
  assert(otherFill.value.idempotencyKey !== a.value.idempotencyKey, "a different fill in the same tx never collides");
});

Deno.test("wallet rows: invalid fields are rejected, never defaulted", () => {
  const cases: [string, Record<string, unknown>][] = [
    ["redeem is not a trade", { type: "REDEEM" }],
    ["bad wallet", { proxyWallet: "0x4483" }],
    ["bad tx", { transactionHash: "0xabc" }],
    ["empty side", { side: "" }],
    ["price above 1", { price: 1.2 }],
    ["missing price", { price: undefined }],
    ["zero size", { size: 0 }],
    ["missing timestamp", { timestamp: undefined }],
    ["missing asset", { asset: undefined }],
  ];
  for (const [label, patch] of cases) {
    assert(!normalizeWalletActivityRow({ ...tradeRow, ...patch }, RECEIVED).ok, label);
  }
});

Deno.test("activity partition keeps settlements apart from trades and de-duplicates", () => {
  const p = partitionWalletActivity([tradeRow, tradeRow, redeemRow, { ...tradeRow, side: "HOLD" }], RECEIVED);
  eq(p.trades.length, 1, "duplicate trade collapsed");
  eq(p.settlements, 1, "redeem counted as settlement, not trade");
  eq(p.rejected.length, 1, "malformed row reported, not silently dropped");
  eq(p.rejected[0].index, 3, "rejected index");
});

Deno.test("identity: VERIFIED only when search, wallet and activity agree", () => {
  const search = [
    { name: "NIULAI4", proxyWallet: WALLET },
    { name: "Niulai", proxyWallet: "0x3ef47f3553283b1857c023637d35636fcb9ffb21" },
  ];
  const v = verifyProfileIdentity("NIULAI4", search, [tradeRow, redeemRow]);
  assert(v.status === "VERIFIED", "should verify");
  eq(v.wallet, WALLET, "wallet comes from the API, not from the caller");
});

Deno.test("identity: never guesses -- every doubtful case stays UNVERIFIED or AMBIGUOUS", () => {
  const good = [{ name: "NIULAI4", proxyWallet: WALLET }];
  const wrongName = verifyProfileIdentity("NIULAI4", [{ name: "niulai4", proxyWallet: WALLET }], [tradeRow]);
  eq(wrongName.status, "UNVERIFIED", "case-different lookalike is not a match");
  const none = verifyProfileIdentity("NIULAI4", [{ name: "Niulai", proxyWallet: WALLET }], [tradeRow]);
  eq(none.status, "UNVERIFIED", "prefix/substring match is not a match");
  const truncated = verifyProfileIdentity("NIULAI4", [{ name: "NIULAI4", proxyWallet: "0x4483...54c6" }], [tradeRow]);
  eq(truncated.status, "UNVERIFIED", "a shortened wallet is never completed");
  const ambiguous = verifyProfileIdentity(
    "NIULAI4",
    [{ name: "NIULAI4", proxyWallet: WALLET }, { name: "NIULAI4", proxyWallet: "0x3ef47f3553283b1857c023637d35636fcb9ffb21" }],
    [tradeRow],
  );
  eq(ambiguous.status, "AMBIGUOUS", "two exact names with different wallets");
  eq(verifyProfileIdentity("NIULAI4", good, []).status, "UNVERIFIED", "no activity to cross-check");
  eq(verifyProfileIdentity("NIULAI4", good, null).status, "UNVERIFIED", "activity unavailable");
  eq(verifyProfileIdentity("NIULAI4", null, [tradeRow]).status, "UNVERIFIED", "search unavailable");
  const otherWalletActivity = { ...tradeRow, proxyWallet: "0x3ef47f3553283b1857c023637d35636fcb9ffb21" };
  eq(verifyProfileIdentity("NIULAI4", good, [tradeRow, otherWalletActivity]).status, "UNVERIFIED", "mixed wallets in activity");
  const otherName = { ...tradeRow, name: "someone-else" };
  eq(verifyProfileIdentity("NIULAI4", good, [otherName]).status, "UNVERIFIED", "activity names a different user");
  const anonymous = { ...tradeRow, name: undefined };
  eq(verifyProfileIdentity("NIULAI4", good, [anonymous]).status, "UNVERIFIED", "activity never repeats the name");
});

// Optional live smoke test: only runs when explicitly requested (PREDICTION_LIVE=1 with --allow-env
// --allow-net), so CI -- which runs with no permissions -- never depends on the network.
function liveRequested(): boolean {
  try {
    return Deno.env.get("PREDICTION_LIVE") === "1";
  } catch {
    return false;
  }
}
Deno.test({
  name: "LIVE: public endpoints answer and NIULAI4 verifies (set PREDICTION_LIVE=1)",
  ignore: !liveRequested(),
  fn: async () => {
    const search = await fetchPredictionJson<{ profiles?: unknown[] }>(
      "https://gamma-api.polymarket.com/public-search?q=NIULAI4&search_profiles=true",
    );
    eq(search.status, "OK", "search status");
    const wallet = (search.body?.profiles?.[0] as { proxyWallet?: string } | undefined)?.proxyWallet ?? "";
    const activity = await fetchPredictionJson<unknown[]>(
      `https://data-api.polymarket.com/activity?user=${wallet}&type=TRADE&limit=5`,
    );
    eq(activity.status, "OK", "activity status");
    const verdict = verifyProfileIdentity("NIULAI4", search.body?.profiles ?? null, activity.body);
    assert(verdict.status === "VERIFIED", `live identity: ${JSON.stringify(verdict)}`);
    let seen = 0;
    for await (const page of iterateGammaMarkets({ limit: 5, maxPages: 2, active: true, closed: false })) {
      for (const raw of page.markets) {
        const m = normalizeGammaMarket(raw, page.fetch.receivedAt);
        if (m.ok) seen++;
      }
    }
    assert(seen > 0, "live markets normalize");
  },
});

Deno.test("taker fee follows the official C x feeRate x p x (1-p) shape, 5-decimal rounding, makers free", async () => {
  const { polymarketTakerFeeUsdc } = await import("./prediction_market.ts");
  eq(polymarketTakerFeeUsdc(100, 0.5, 0.07), 1.75, "crypto rate at p=0.5");
  eq(polymarketTakerFeeUsdc(100, 0.95, 0.07), 0.3325, "near p=1 the fee shrinks (a proportional model would say 6.65)");
  eq(polymarketTakerFeeUsdc(100, 0.5, 0), 0, "fee-free category");
  eq(polymarketTakerFeeUsdc(100, 0.5, 0.07, "MAKER"), 0, "makers are never charged");
  eq(polymarketTakerFeeUsdc(1, 0.9999, 0.04), 0, "below the 0.00001 minimum is not charged");
  eq(polymarketTakerFeeUsdc(2, 0.123, 0.05), 0.01079, "rounded to 5 decimals (0.0107871)");
  for (const bad of [() => polymarketTakerFeeUsdc(-1, 0.5, 0.07), () => polymarketTakerFeeUsdc(1, 1.2, 0.07), () => polymarketTakerFeeUsdc(1, 0.5, 1)]) {
    let threw = false;
    try {
      bad();
    } catch {
      threw = true;
    }
    assert(threw, "invalid fee inputs are refused, not defaulted");
  }
});

Deno.test("resolution is accepted only when closed, UMA-resolved and prices are exactly one 1", async () => {
  const { normalizeGammaResolution, gammaMarketByIdUrl } = await import("./prediction_market.ts");
  const resolved = {
    id: "507414", closed: true, umaResolutionStatus: "resolved", outcomes: '["Yes", "No"]', outcomePrices: '["1", "0"]',
    closedTime: "2024-09-18 21:17:18+00",
  };
  const r = normalizeGammaResolution(resolved, RECEIVED);
  assert(r.ok, "resolved market normalizes");
  eq(r.value.yesOutcome, 1, "Yes won");
  eq(r.value.winningLabel, "Yes", "winning label");
  eq(r.value.closedTime, "2024-09-18T21:17:18.000Z", "closedTime parsed from Gamma's space/+00 format");
  const no = normalizeGammaResolution({ ...resolved, outcomePrices: '["0", "1"]' }, RECEIVED);
  assert(no.ok && no.value.yesOutcome === 0, "No won");
  const cases: [string, Record<string, unknown>][] = [
    ["still open", { closed: false }],
    ["proposed, not final", { umaResolutionStatus: "proposed" }],
    ["disputed", { umaResolutionStatus: "disputed" }],
    ["missing status", { umaResolutionStatus: undefined }],
    ["50/50 split", { outcomePrices: '["0.5", "0.5"]' }],
    ["near-final price is not final", { outcomePrices: '["0.9995", "0.0005"]' }],
    ["two winners", { outcomePrices: '["1", "1"]' }],
    ["length mismatch", { outcomePrices: '["1"]' }],
    ["empty price", { outcomePrices: '["1", ""]' }],
  ];
  for (const [label, patch] of cases) assert(!normalizeGammaResolution({ ...resolved, ...patch }, RECEIVED).ok, label);
  const multi = normalizeGammaResolution({ ...resolved, outcomes: '["Trump", "Harris", "Other"]', outcomePrices: '["1", "0", "0"]' }, RECEIVED);
  assert(multi.ok && multi.value.yesOutcome === null, "non Yes/No market is kept but excluded from binary calibration");
  eq(gammaMarketByIdUrl("2589810"), "https://gamma-api.polymarket.com/markets/2589810", "by-id url");
  let threw = false;
  try {
    gammaMarketByIdUrl("1/../x");
  } catch {
    threw = true;
  }
  assert(threw, "market id injection refused");
});
