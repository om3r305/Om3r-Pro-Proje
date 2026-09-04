// Exact Binance Spot L2 wire parsing helpers.
//
// JSON.parse coerces JSON integer tokens to JS Number before a reviver can see them. Binance
// update IDs are int64-shaped values, so a future value above Number.MAX_SAFE_INTEGER could be
// silently rounded. We therefore keep the original raw JSON and extract U/u/lastUpdateId as exact
// digit strings, while still JSON-parsing the surrounding structure for normal schema checks.

import type {
  BinanceRawDepthDiff,
  BinanceRawDepthSnapshot,
} from "./l2_book.ts";

export interface BinanceCombinedDepthMessage {
  stream: string;
  data: BinanceRawDepthDiff;
}

function objectRecord(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

/** Exact top-level-ish unsigned integer token extraction. Binance's depth message/snapshot schema
 * contains each target key exactly once. Refuse ambiguous/missing input instead of guessing. */
export function extractExactUnsignedInteger(rawJson: string, key: string): string {
  if (!/^[A-Za-z][A-Za-z0-9]*$/.test(key)) throw new Error("invalid JSON key selector");
  const expression = new RegExp(`"${key}"\\s*:\\s*(\\d+)`, "g");
  const matches = [...rawJson.matchAll(expression)];
  if (matches.length !== 1) {
    throw new Error(`expected exactly one numeric ${key} token, found ${matches.length}`);
  }
  return matches[0][1];
}

function exactLevelArray(value: unknown, name: string): [string, string][] {
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  return value.map((entry, index) => {
    if (!Array.isArray(entry) || entry.length < 2) {
      throw new Error(`${name}[${index}] must be a [price,quantity] tuple`);
    }
    if (typeof entry[0] !== "string" || typeof entry[1] !== "string") {
      throw new Error(`${name}[${index}] price/quantity must be strings`);
    }
    return [entry[0], entry[1]];
  });
}

export function parseBinanceCombinedDepthRaw(rawJson: string): BinanceCombinedDepthMessage {
  const outer = objectRecord(JSON.parse(rawJson), "combined depth message");
  if (typeof outer.stream !== "string" || !outer.stream.trim()) throw new Error("combined stream name missing");
  const data = objectRecord(outer.data, "combined depth data");
  if (data.e !== "depthUpdate") throw new Error(`unexpected Binance event type: ${String(data.e)}`);
  if (typeof data.s !== "string" || !data.s.trim()) throw new Error("depth symbol missing");
  if (typeof data.E !== "number" || !Number.isSafeInteger(data.E)) throw new Error("depth event time E must be a safe integer");

  const expectedStream = `${data.s.toLowerCase()}@depth@100ms`;
  if (outer.stream.toLowerCase() !== expectedStream) {
    throw new Error(`combined stream/symbol lineage mismatch: ${outer.stream} vs ${expectedStream}`);
  }

  return {
    stream: outer.stream,
    data: {
      e: "depthUpdate",
      E: data.E,
      s: data.s,
      U: extractExactUnsignedInteger(rawJson, "U"),
      u: extractExactUnsignedInteger(rawJson, "u"),
      b: exactLevelArray(data.b, "b"),
      a: exactLevelArray(data.a, "a"),
    },
  };
}

export function parseBinanceDepthSnapshotRaw(rawJson: string): BinanceRawDepthSnapshot {
  const raw = objectRecord(JSON.parse(rawJson), "depth snapshot");
  return {
    lastUpdateId: extractExactUnsignedInteger(rawJson, "lastUpdateId"),
    bids: exactLevelArray(raw.bids, "bids"),
    asks: exactLevelArray(raw.asks, "asks"),
  };
}
