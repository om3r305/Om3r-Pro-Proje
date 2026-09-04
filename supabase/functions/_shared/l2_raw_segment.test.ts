import { assert, assertEquals, assertThrows } from "jsr:@std/assert@^1.0.0";
import { buildRawL2Segment, type RawL2Item } from "./l2_raw_segment.ts";

function item(arrivalSeq: number, rawJson: string, overrides: Partial<RawL2Item> = {}): RawL2Item {
  return {
    collectorSessionId: "session-a",
    arrivalSeq,
    connectionGeneration: 1,
    syncGeneration: 1,
    source: "binance_spot_diff_depth_ws",
    symbol: "BTCUSDT",
    collectorReceivedAt: `2026-09-04T10:00:0${Math.min(arrivalSeq, 9)}.000Z`,
    rawJson,
    ...overrides,
  };
}

Deno.test("raw L2 segment preserves exact raw integer/decimal lexemes and total arrival order", () => {
  const huge = "9223372036854775807";
  const segment = buildRawL2Segment([
    item(1, `{"e":"depthUpdate","U":${huge},"u":${huge},"b":[["0.10000000","1.2300"]],"a":[]}`),
    item(2, `{"lastUpdateId":${huge},"bids":[["0.10000000","1.2300"]],"asks":[]}`, {
      source: "binance_spot_rest_depth_snapshot",
    }),
  ]);
  assertEquals(segment.firstArrivalSeq, 1);
  assertEquals(segment.lastArrivalSeq, 2);
  assertEquals(segment.messageCount, 2);
  assert(segment.ndjson.includes(`"U":${huge}`));
  assert(segment.ndjson.includes(`"lastUpdateId":${huge}`));
  assert(segment.ndjson.includes('"0.10000000"'));
  assert(segment.ndjson.indexOf('"arrival_seq":1') < segment.ndjson.indexOf('"arrival_seq":2'));
});

Deno.test("raw L2 segment retains malformed provider text with null unknown lineage", () => {
  const segment = buildRawL2Segment([
    item(1, "{bad", { symbol: null, syncGeneration: null }),
  ]);
  assert(segment.ndjson.includes('"raw_valid_json":false'));
  assert(segment.ndjson.includes('"raw_text":"{bad"'));
  assert(segment.ndjson.includes('"symbol":null'));
  assert(segment.ndjson.includes('"sync_generation":null'));
});

Deno.test("raw L2 segment rejects mixed sessions and duplicate/gapped sequence", () => {
  assertThrows(() => buildRawL2Segment([
    item(1, "{}"), item(2, "{}", { collectorSessionId: "session-b" }),
  ]), Error, "cannot mix");
  assertThrows(() => buildRawL2Segment([item(2, "{}"), item(2, "{}")]), Error, "contiguous");
  assertThrows(() => buildRawL2Segment([item(10, "{}"), item(12, "{}")] ), Error, "expected 11");
});
