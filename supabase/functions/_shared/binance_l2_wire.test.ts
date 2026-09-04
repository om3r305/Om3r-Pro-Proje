import { assertEquals, assertThrows } from "jsr:@std/assert@^1.0.0";
import {
  extractExactUnsignedInteger,
  isBinanceServerShutdownRaw,
  parseBinanceCombinedDepthRaw,
  parseBinanceDepthSnapshotRaw,
} from "./binance_l2_wire.ts";

const HUGE = "9223372036854775807";

Deno.test("Binance L2 wire: combined diff preserves int64 update IDs as exact strings", () => {
  const raw = `{"stream":"btcusdt@depth@100ms","data":{"e":"depthUpdate","E":1770000000000,"s":"BTCUSDT","U":${HUGE},"u":${HUGE},"b":[["100.00000000","1.2300"]],"a":[["101.00000000","2.3400"]]}}`;
  const parsed = parseBinanceCombinedDepthRaw(raw);
  assertEquals(parsed.data.U, HUGE);
  assertEquals(parsed.data.u, HUGE);
  assertEquals(parsed.data.b[0], ["100.00000000", "1.2300"]);
});

Deno.test("Binance L2 wire: REST snapshot preserves exact lastUpdateId", () => {
  const raw = `{"lastUpdateId":${HUGE},"bids":[["100.0","1"]],"asks":[["101.0","2"]]}`;
  assertEquals(parseBinanceDepthSnapshotRaw(raw).lastUpdateId, HUGE);
});

Deno.test("Binance L2 wire: combined stream name must match payload symbol and cadence", () => {
  const raw = `{"stream":"ethusdt@depth@100ms","data":{"e":"depthUpdate","E":1770000000000,"s":"BTCUSDT","U":1,"u":2,"b":[],"a":[]}}`;
  assertThrows(() => parseBinanceCombinedDepthRaw(raw), Error, "lineage mismatch");
});

Deno.test("Binance L2 wire: serverShutdown is transport control, not a depth event", () => {
  assertEquals(isBinanceServerShutdownRaw('{"stream":"!serverShutdown","data":{"e":"serverShutdown","E":1770000000000}}'), true);
  assertEquals(isBinanceServerShutdownRaw('{"e":"serverShutdown","E":1770000000000}'), true);
  assertEquals(isBinanceServerShutdownRaw('{"e":"depthUpdate"}'), false);
});

Deno.test("Binance L2 wire: missing or ambiguous exact integer tokens fail closed", () => {
  assertThrows(() => extractExactUnsignedInteger("{}", "U"), Error, "found 0");
  assertThrows(() => extractExactUnsignedInteger('{"U":1,"nested":{"U":2}}', "U"), Error, "found 2");
});

Deno.test("Binance L2 wire: an escaped U token elsewhere cannot impersonate a missing data.U field", () => {
  const raw = '{"stream":"btcusdt@depth@100ms","data":{"e":"depthUpdate","E":1770000000000,"s":"BTCUSDT","u":2,"b":[],"a":[],"note":"\\\"U\\\":1"}}';
  assertThrows(() => parseBinanceCombinedDepthRaw(raw), Error, "depth U must be an unsigned numeric JSON integer field");
});

Deno.test("Binance L2 wire: an escaped lastUpdateId token cannot impersonate the snapshot field", () => {
  const raw = '{"bids":[],"asks":[],"note":"\\\"lastUpdateId\\\":123"}';
  assertThrows(() => parseBinanceDepthSnapshotRaw(raw), Error, "snapshot lastUpdateId must be an unsigned numeric JSON integer field");
});

Deno.test("Binance L2 wire: depth levels are exact two-string tuples", () => {
  const extra = '{"stream":"btcusdt@depth@100ms","data":{"e":"depthUpdate","E":1770000000000,"s":"BTCUSDT","U":1,"u":2,"b":[["100","1","EXTRA"]],"a":[]}}';
  assertThrows(() => parseBinanceCombinedDepthRaw(extra), Error, "exactly a [price,quantity] tuple");

  const numeric = '{"lastUpdateId":2,"bids":[[100,"1"]],"asks":[]}';
  assertThrows(() => parseBinanceDepthSnapshotRaw(numeric), Error, "price/quantity must be strings");
});
