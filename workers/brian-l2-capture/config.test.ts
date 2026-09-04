import { assertEquals, assertThrows } from "jsr:@std/assert@^1.0.0";
import {
  combinedDepthWsUrl,
  depthSnapshotUrl,
  parseSymbols,
} from "./config.ts";

Deno.test("L2 config: symbols are normalized, unique and deterministic", () => {
  assertEquals(parseSymbols("xrpUSDT, btcusdt,ETHUSDT"), ["BTCUSDT", "ETHUSDT", "XRPUSDT"]);
  assertThrows(() => parseSymbols("BTCUSDT,btcusdt"), Error, "duplicates");
  assertThrows(() => parseSymbols("BTC-USDT"), Error, "invalid Binance symbol");
});

Deno.test("L2 config: builds Binance public combined diff-depth and REST snapshot URLs", () => {
  assertEquals(
    combinedDepthWsUrl("wss://stream.binance.com:9443/", ["BTCUSDT", "XRPUSDT"]),
    "wss://stream.binance.com:9443/stream?streams=btcusdt@depth@100ms/xrpusdt@depth@100ms",
  );
  assertEquals(
    depthSnapshotUrl("https://api.binance.com/", "BTCUSDT", 1000),
    "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=1000",
  );
});
