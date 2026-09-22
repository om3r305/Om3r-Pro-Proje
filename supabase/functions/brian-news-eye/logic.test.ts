// Behavioral tests for the actual deployed brian-news-eye parsing/scoring logic.
// Run with: deno test --allow-read supabase/functions/brian-news-eye

import { assertEquals } from "jsr:@std/assert@^1.0.0";
import { asset, clip, parseSeen, pressure, sign } from "./logic.ts";

Deno.test("parseSeen: parses GDELT's compact YYYYMMDDTHHMMSSZ into ISO-8601", () => {
  assertEquals(parseSeen("20260919T143000Z"), "2026-09-19T14:30:00.000Z");
});

Deno.test("parseSeen: undefined, garbage, and malformed values all return null, not a throw", () => {
  assertEquals(parseSeen(undefined), null);
  assertEquals(parseSeen(""), null);
  assertEquals(parseSeen("not a date"), null);
  assertEquals(parseSeen("2026-09-19"), null);
});

Deno.test("asset: routes crypto tickers, commodities and macro keywords, first match wins", () => {
  assertEquals(asset("Bitcoin surges past $100k"), "BTCUSDT");
  assertEquals(asset("BTC rally continues"), "BTCUSDT");
  assertEquals(asset("Ethereum upgrade goes live"), "ETHUSDT");
  assertEquals(asset("Solana network outage"), "SOLUSDT");
  assertEquals(asset("XRP lawsuit update"), "XRPUSDT");
  assertEquals(asset("Ripple wins court case"), "XRPUSDT");
  assertEquals(asset("Binance Coin burn event"), "BNBUSDT");
  assertEquals(asset("Gold hits record high"), "GOLD");
  assertEquals(asset("Silver demand rises"), "SILVER");
  assertEquals(asset("Oil prices fall on WTI data"), "OIL");
  assertEquals(asset("Federal Reserve holds interest rates"), "MACRO");
  assertEquals(asset("ECB inflation outlook"), "MACRO");
});

Deno.test("asset: an unmatched headline falls back to the generic crypto market bucket", () => {
  assertEquals(asset("Random unrelated headline about weather"), "CRYPTO_MARKET");
});

Deno.test("pressure: positive-only words score +1, negative-only score -1, mixed or neither score 0", () => {
  assertEquals(pressure("Major exchange partnership announced, adoption surges"), 1);
  assertEquals(pressure("Exchange hacked, funds stolen in exploit"), -1);
  assertEquals(pressure("Approval granted despite lawsuit ongoing"), 0);
  assertEquals(pressure("Completely neutral headline with no signal words"), 0);
});

Deno.test("clip: clamps to [0, 1] inclusive", () => {
  assertEquals(clip(-1), 0);
  assertEquals(clip(0.5), 0.5);
  assertEquals(clip(2), 1);
});

Deno.test("sign: returns -1/0/1", () => {
  assertEquals(sign(10), 1);
  assertEquals(sign(-10), -1);
  assertEquals(sign(0), 0);
});
