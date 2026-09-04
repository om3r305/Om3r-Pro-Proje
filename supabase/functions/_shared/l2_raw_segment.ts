// Lossless raw L2 segment framing for Brian 2026 capture.
//
// Each line embeds the venue JSON text verbatim as the value of `raw`. This preserves raw integer
// and decimal lexemes for audit even when a downstream JSON parser would coerce a large integer
// through JS Number. Every raw L2 transport arrival in a collector session is assigned a sequence,
// so a persisted segment must be contiguous as well as ordered. If a malformed message cannot be
// assigned a truthful symbol/sync generation, those raw-only metadata fields remain null rather
// than being fabricated.

export type L2RawSource = "binance_spot_diff_depth_ws" | "binance_spot_rest_depth_snapshot";

export interface RawL2Item {
  collectorSessionId: string;
  arrivalSeq: number;
  connectionGeneration: number;
  syncGeneration: number | null;
  source: L2RawSource;
  symbol: string | null;
  collectorReceivedAt: string;
  rawJson: string;
}

export interface RawL2Segment {
  collectorSessionId: string;
  firstArrivalSeq: number;
  lastArrivalSeq: number;
  messageCount: number;
  observedAt: string;
  ndjson: string;
}

function ensurePositiveSafeInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive safe integer`);
}

function validateRawJson(value: string): void {
  if (!value.trim()) throw new Error("rawJson is required");
  // Validate syntax only. Never reserialize the raw value; its original text is embedded below.
  JSON.parse(value);
}

function line(item: RawL2Item): string {
  const meta = JSON.stringify({
    collector_session_id: item.collectorSessionId,
    arrival_seq: item.arrivalSeq,
    connection_generation: item.connectionGeneration,
    sync_generation: item.syncGeneration,
    source: item.source,
    symbol: item.symbol,
    collector_received_at: item.collectorReceivedAt,
  });
  return `${meta.slice(0, -1)},"raw":${item.rawJson}}`;
}

export function buildRawL2Segment(items: RawL2Item[]): RawL2Segment {
  if (!items.length) throw new Error("cannot build an empty raw L2 segment");
  const session = items[0].collectorSessionId.trim();
  if (!session) throw new Error("collectorSessionId is required");

  let prior: number | null = null;
  let observedAt = "";
  const lines: string[] = [];
  for (const [index, item] of items.entries()) {
    if (item.collectorSessionId.trim() !== session) throw new Error("raw L2 segment cannot mix collector sessions");
    ensurePositiveSafeInteger(item.arrivalSeq, `items[${index}].arrivalSeq`);
    ensurePositiveSafeInteger(item.connectionGeneration, `items[${index}].connectionGeneration`);
    if (item.syncGeneration !== null) {
      ensurePositiveSafeInteger(item.syncGeneration, `items[${index}].syncGeneration`);
    }
    if (prior !== null && item.arrivalSeq !== prior + 1) {
      throw new Error(`raw L2 segment arrival_seq must be contiguous; expected ${prior + 1}, got ${item.arrivalSeq}`);
    }
    prior = item.arrivalSeq;
    if (item.symbol !== null && !item.symbol.trim()) {
      throw new Error(`items[${index}].symbol must be non-empty when present`);
    }
    const receivedMs = Date.parse(item.collectorReceivedAt);
    if (!Number.isFinite(receivedMs)) throw new Error(`items[${index}].collectorReceivedAt must be valid`);
    validateRawJson(item.rawJson);
    if (!observedAt || receivedMs < Date.parse(observedAt)) observedAt = item.collectorReceivedAt;
    lines.push(line(item));
  }

  return {
    collectorSessionId: session,
    firstArrivalSeq: items[0].arrivalSeq,
    lastArrivalSeq: items[items.length - 1].arrivalSeq,
    messageCount: items.length,
    observedAt,
    ndjson: `${lines.join("\n")}\n`,
  };
}
