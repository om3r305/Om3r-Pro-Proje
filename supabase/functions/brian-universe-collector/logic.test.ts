// Behavioral tests for the actual deployed brian-universe-collector filtering/scoring logic.
// Run with: deno test --allow-read supabase/functions/brian-universe-collector

import { assertEquals, assertThrows } from "jsr:@std/assert@^1.0.0";
import {
  buildEligibleRows, clip01, CONFIG, diffEligibility, finiteNumber, indexRows, rankPercentiles, scoreCandidates,
  type UniverseConfig,
} from "./logic.ts";

Deno.test("finiteNumber: coerces or falls back", () => {
  assertEquals(finiteNumber("42"), 42);
  assertEquals(finiteNumber("nan", -1), -1);
  assertEquals(finiteNumber(undefined, 0), 0);
});

Deno.test("clip01: clamps to [0, 1]", () => {
  assertEquals(clip01(-1), 0);
  assertEquals(clip01(0.5), 0.5);
  assertEquals(clip01(3), 1);
});

Deno.test("rankPercentiles: smallest is 0, largest is 1, ties break by input order", () => {
  assertEquals(rankPercentiles([]), []);
  assertEquals(rankPercentiles([5]), [0]);
  assertEquals(rankPercentiles([10, 30, 20]), [0, 1, 0.5]);
  assertEquals(rankPercentiles([1, 1]), [0, 1]);
});

Deno.test("indexRows: indexes an array-of-objects response by symbol, ignoring rows without one", () => {
  const payload = [{ symbol: "BTCUSDT", x: 1 }, { symbol: "ETHUSDT", x: 2 }, { x: 3 }];
  const map = indexRows(payload);
  assertEquals(map.size, 2);
  assertEquals(map.get("BTCUSDT"), { symbol: "BTCUSDT", x: 1 });
});

Deno.test("indexRows: rejects a non-array payload instead of silently returning an empty map", () => {
  assertThrows(() => indexRows({ not: "an array" }));
});

function exchangeSymbol(symbol: string, base: string, quote = "USDT", overrides: Record<string, unknown> = {}) {
  return { symbol, baseAsset: base, quoteAsset: quote, status: "TRADING", isSpotTradingAllowed: true, ...overrides };
}
function tickerRow(overrides: Record<string, unknown> = {}) {
  return { lastPrice: "100", quoteVolume: "10000000", count: "5000", priceChangePercent: "2.5", highPrice: "105", lowPrice: "95", ...overrides };
}

Deno.test("buildEligibleRows: a fully qualifying symbol passes every gate", () => {
  const exchangeSymbols = [exchangeSymbol("BTCUSDT", "BTC")];
  const tickerMap = new Map([["BTCUSDT", tickerRow()]]);
  const bookMap = new Map([["BTCUSDT", { bidPrice: "99.9", askPrice: "100.1" }]]);
  const rows = buildEligibleRows(exchangeSymbols, tickerMap, bookMap);
  assertEquals(rows.length, 1);
  assertEquals(rows[0].symbol, "BTCUSDT");
  assertEquals(rows[0].spread_bps !== null, true);
});

Deno.test("buildEligibleRows: rejects non-TRADING status, disabled spot trading, wrong quote asset, and excluded base assets", () => {
  const tickerMap = new Map([
    ["BTCUSDT", tickerRow()], ["ETHBTC", tickerRow()], ["USDCUSDT", tickerRow()], ["SOLUSDT", tickerRow()],
  ]);
  const exchangeSymbols = [
    exchangeSymbol("BTCUSDT", "BTC", "USDT", { status: "BREAK" }),
    exchangeSymbol("ETHBTC", "ETH", "BTC"),
    exchangeSymbol("USDCUSDT", "USDC", "USDT"),
    exchangeSymbol("SOLUSDT", "SOL", "USDT", { isSpotTradingAllowed: false }),
  ];
  const rows = buildEligibleRows(exchangeSymbols, tickerMap, new Map());
  assertEquals(rows.length, 0);
});

Deno.test("buildEligibleRows: rejects below-threshold liquidity/activity/price and an inverted high/low range", () => {
  const config: UniverseConfig = CONFIG;
  const tickerMap = new Map([
    ["LOWVOLUSDT", tickerRow({ quoteVolume: "1" })],
    ["LOWTRADEUSDT", tickerRow({ count: "1" })],
    ["ZEROUSDT", tickerRow({ lastPrice: "0" })],
    ["INVERTEDUSDT", tickerRow({ highPrice: "90", lowPrice: "95" })],
  ]);
  const exchangeSymbols = [
    exchangeSymbol("LOWVOLUSDT", "LOWVOL"), exchangeSymbol("LOWTRADEUSDT", "LOWTRADE"),
    exchangeSymbol("ZEROUSDT", "ZERO"), exchangeSymbol("INVERTEDUSDT", "INVERTED"),
  ];
  assertEquals(buildEligibleRows(exchangeSymbols, tickerMap, new Map(), config).length, 0);
});

Deno.test("buildEligibleRows: a missing book-ticker row degrades gracefully to a null spread, not an exclusion", () => {
  const exchangeSymbols = [exchangeSymbol("BTCUSDT", "BTC")];
  const tickerMap = new Map([["BTCUSDT", tickerRow()]]);
  const rows = buildEligibleRows(exchangeSymbols, tickerMap, new Map());
  assertEquals(rows.length, 1);
  assertEquals(rows[0].spread_bps, null);
});

Deno.test("scoreCandidates: a symbol dominant on every axis scores highest and is sorted first", () => {
  const rows = [
    { symbol: "WEAK", base_asset: "WEAK", last_price: 1, quote_volume: 5_000_000, trades_24h: 1_000, price_change_pct: 0.1, range_pct: 1, spread_bps: 50 as number | null },
    { symbol: "STRONG", base_asset: "STRONG", last_price: 1, quote_volume: 500_000_000, trades_24h: 200_000, price_change_pct: 15, range_pct: 20, spread_bps: 1 as number | null },
  ];
  const candidates = scoreCandidates(rows);
  assertEquals(candidates[0].symbol, "STRONG");
  assertEquals(candidates[0].radar_score > candidates[1].radar_score, true);
  assertEquals(candidates[0].reasons.includes("high relative liquidity"), true);
  assertEquals(candidates[0].reasons.includes("tight top-of-book spread"), true);
});

Deno.test("scoreCandidates: a null spread falls back to a neutral 0.5 spread quality, not a throw", () => {
  const rows = [{ symbol: "X", base_asset: "X", last_price: 1, quote_volume: 10_000_000, trades_24h: 5_000, price_change_pct: 1, range_pct: 2, spread_bps: null }];
  const candidates = scoreCandidates(rows);
  assertEquals(candidates[0].spread_quality, 0.5);
});

Deno.test("diffEligibility: no prior snapshot means not comparable and both diffs are empty", () => {
  const d = diffEligibility(null, ["BTCUSDT", "ETHUSDT"]);
  assertEquals(d, { comparable: false, newlyObserved: [], disappeared: [] });
});

Deno.test("diffEligibility: detects newly observed and disappeared symbols against a prior snapshot", () => {
  const d = diffEligibility(["BTCUSDT", "ETHUSDT"], ["BTCUSDT", "SOLUSDT"]);
  assertEquals(d.comparable, true);
  assertEquals(d.newlyObserved, ["SOLUSDT"]);
  assertEquals(d.disappeared, ["ETHUSDT"]);
});
