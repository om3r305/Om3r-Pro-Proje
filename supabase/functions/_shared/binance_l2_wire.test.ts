import { assertEquals, assertThrows } from "jsr:@std/assert@^1.0.0";
import {
  extractExactUnsignedInteger,
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
  const parsed = parseBinanceDepthSnapshotRaw(raw);
  assertEquals(parsed.lastUpdateId, HUGE);
});

Deno.test("Binance L2 wire: combined stream name must match payload symbol and cadence", () => {
  const raw = `{"stream":"ethusdt@depth@100ms","data":{"e":"depthUpdate","E":1770000000000,"s":"BTCUSDT","U":1,"u":2,"b":[],"a":[]}}`;
  assertThrows(() => parseBinanceCombinedDepthRaw(raw), Error, "lineage mismatch");
});

Deno.test("Binance L2 wire: missing or ambiguous exact integer tokens fail closed", () => {
  assertThrows(() => extractExactUnsignedInteger("{}", "U"), Error, "found 0");
  assertThrows(() => extractExactUnsignedInteger('{"U":1,"nested":{"U":2}}', "U"), Error, "found 2");
});
