import {
  compileL2Cost,
  type CostSide,
  type DecimalLevel,
  type DynamicCostQuote,
} from "../dynamic_cost.ts";

export type MarketL2Classification =
  | "HEALTHY"
  | "BLOCKED"
  | "INSUFFICIENT_EVIDENCE";

export interface MarketL2Options {
  decisionAt: string | number;
  side: CostSide;
  requestedNotionalUsd: number;
  feeBps: number;
  maxInputRows?: number;
  maxLevels?: number;
  staleAfterMs?: number;
}

export interface MarketL2Report {
  classification: MarketL2Classification;
  healthy: boolean;
  venue: string | null;
  symbol: string | null;
  decisionAt: string | null;
  book: {
    bestBid: string | null;
    bestAsk: string | null;
    spreadBps: number | null;
    visibleBidLevels: number;
    visibleAskLevels: number;
  };
  cost: DynamicCostQuote | null;
  provenance: {
    collectorSessionId: string | null;
    connectionGeneration: number | null;
    syncGeneration: number | null;
    sourceEventIds: string[];
    arrivalSeq: number[];
    snapshotLastUpdateId: string | null;
    asOf: string | null;
  };
  cadence: {
    sampleCount: number;
    firstArrivalAt: string | null;
    lastArrivalAt: string | null;
    windowMs: number | null;
  };
  blockers: string[];
  invalidEvidenceCount: number;
  futureTelemetry: { futureEvidenceCount: number };
  truncated: boolean;
  shadow_only: true;
  live_execution: false;
  promotionReady: false;
}

type Instant = { text: string; ms: number };
type Level = { price: string; size: string };
type Row = {
  kind: "snapshot" | "diff";
  venue: string;
  symbol: string;
  session: string;
  connection: number;
  generation: number;
  arrival: number;
  eventId: string;
  observedAt: Instant;
  receivedAt: Instant;
  lastUpdateId?: string;
  firstUpdateId?: string;
  finalUpdateId?: string;
  bids: Level[];
  asks: Level[];
  lineage: Record<string, unknown>;
};

const MAX_INPUT_ROWS = 2_000;
const MAX_LEVELS = 1_000;
const MAX_NOTIONAL = 1e12;
const MAX_PRICE_SIZE = 1e12;
const ISO_UTC =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?Z$/;

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function instant(value: unknown): Instant | null {
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value) || value < 0 || value > 8.64e15) {
      return null;
    }
    return { text: new Date(value).toISOString(), ms: value };
  }
  if (typeof value !== "string") return null;
  const match = ISO_UTC.exec(value);
  if (!match) return null;
  const ms = Date.parse(value);
  const fraction = (match[7] ?? "").padEnd(3, "0");
  const text = `${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${
    match[6]
  }.${fraction}Z`;
  return Number.isFinite(ms) && new Date(ms).toISOString() === text
    ? { text, ms }
    : null;
}
function identifier(value: unknown): string | null {
  return typeof value === "string" &&
      /^[A-Za-z][A-Za-z0-9_.:-]{0,127}$/.test(value)
    ? value
    : null;
}
function positiveInt(
  value: unknown,
  fallback: number,
  maximum: number,
): number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0
    ? Math.min(value, maximum)
    : fallback;
}
function integerId(value: unknown): string | null {
  if (typeof value === "string" && /^\d+$/.test(value)) return value;
  if (typeof value === "number" && Number.isSafeInteger(value) && value >= 0) {
    return String(value);
  }
  return null;
}
function decimal(value: unknown): string | null {
  if (typeof value !== "string" || !/^\d+(\.\d+)?$/.test(value)) return null;
  const n = Number(value);
  return Number.isFinite(n) && n > 0 && n <= MAX_PRICE_SIZE ? value : null;
}
function decimalKey(value: string): string {
  const [whole, fraction = ""] = value.split(".");
  const normalizedWhole = whole.replace(/^0+(?=\d)/, "");
  const normalizedFraction = fraction.replace(/0+$/, "");
  return `${normalizedWhole}.${normalizedFraction}`;
}
function compareDecimal(a: string, b: string): number {
  const [aWhole, aFraction = ""] = a.split(".");
  const [bWhole, bFraction = ""] = b.split(".");
  const scale = Math.max(aFraction.length, bFraction.length);
  const left = BigInt(aWhole + aFraction.padEnd(scale, "0"));
  const right = BigInt(bWhole + bFraction.padEnd(scale, "0"));
  return left < right ? -1 : left > right ? 1 : 0;
}
function levels(value: unknown, limit: number): Level[] | null {
  if (!Array.isArray(value) || value.length > limit) return null;
  const result: Level[] = [];
  const byPrice = new Map<string, string>();
  for (const item of value) {
    if (!record(item)) return null;
    const price = decimal(item.price);
    const size =
      typeof item.size === "string" && /^\d+(\.\d+)?$/.test(item.size)
        ? Number(item.size) <= MAX_PRICE_SIZE ? item.size : null
        : null;
    if (!price || !size) return null;
    const key = decimalKey(price);
    const prior = byPrice.get(key);
    if (prior !== undefined && decimalKey(prior) !== decimalKey(size)) {
      return null;
    }
    if (prior === undefined) {
      byPrice.set(key, size);
      result.push({ price, size });
    }
  }
  return result;
}
function stable(value: unknown): string {
  return JSON.stringify(value) ?? "";
}
function baseReport(decisionAt: string | null): MarketL2Report {
  return {
    classification: "BLOCKED",
    healthy: false,
    venue: null,
    symbol: null,
    decisionAt,
    book: {
      bestBid: null,
      bestAsk: null,
      spreadBps: null,
      visibleBidLevels: 0,
      visibleAskLevels: 0,
    },
    cost: null,
    provenance: {
      collectorSessionId: null,
      connectionGeneration: null,
      syncGeneration: null,
      sourceEventIds: [],
      arrivalSeq: [],
      snapshotLastUpdateId: null,
      asOf: null,
    },
    cadence: {
      sampleCount: 0,
      firstArrivalAt: null,
      lastArrivalAt: null,
      windowMs: null,
    },
    blockers: ["missing prospective multi-window shadow A/B evidence"],
    invalidEvidenceCount: 0,
    futureTelemetry: { futureEvidenceCount: 0 },
    truncated: false,
    shadow_only: true,
    live_execution: false,
    promotionReady: false,
  };
}
function sideBook(
  map: Map<string, Level>,
  side: "bid" | "ask",
): DecimalLevel[] {
  return [...map.values()].filter((level) => Number(level.size) > 0).sort((
    a,
    b,
  ) => (side === "bid" ? -1 : 1) * compareDecimal(a.price, b.price));
}

export function compileCapabilityGapMarketL2(
  input: unknown,
  options: MarketL2Options,
): MarketL2Report {
  const decision = instant(options?.decisionAt);
  const report = baseReport(decision?.text ?? null);
  if (!decision) {
    report.blockers.push("invalid decision timestamp");
    return report;
  }
  if (!record(input) || !Array.isArray(input.evidence)) {
    report.blockers.push("input must be an evidence envelope");
    return report;
  }
  if (
    !options || typeof options !== "object" ||
    (options.side !== "BUY" && options.side !== "SELL") ||
    typeof options.requestedNotionalUsd !== "number" ||
    !Number.isFinite(options.requestedNotionalUsd) ||
    options.requestedNotionalUsd <= 0 ||
    options.requestedNotionalUsd > MAX_NOTIONAL ||
    typeof options.feeBps !== "number" ||
    !Number.isFinite(options.feeBps) ||
    options.feeBps < 0
  ) {
    report.blockers.push("invalid cost bounds");
    return report;
  }
  const maxRows = positiveInt(
    options.maxInputRows,
    MAX_INPUT_ROWS,
    MAX_INPUT_ROWS,
  );
  const maxLevels = positiveInt(options.maxLevels, MAX_LEVELS, MAX_LEVELS);
  const raw = [...input.evidence].sort((a, b) =>
    stable(a).localeCompare(stable(b))
  );
  report.truncated = raw.length > maxRows;
  const accepted = raw.slice(0, maxRows);
  if (report.truncated) report.blockers.push("input envelope truncated");
  const rows: Row[] = [];
  const signatures = new Map<string, string>();
  for (const value of accepted) {
    if (!record(value)) {
      report.invalidEvidenceCount++;
      continue;
    }
    const observed = instant(value.observedAt);
    const received = instant(value.receivedAt);
    const future = (observed?.ms ?? 0) > decision.ms ||
      (received?.ms ?? 0) > decision.ms;
    if (future && observed && received) {
      report.futureTelemetry.futureEvidenceCount++;
      continue;
    }
    const venue = identifier(value.venue);
    const symbol = identifier(value.symbol);
    const session = identifier(value.collectorSessionId);
    const connection = value.connectionGeneration;
    const generation = value.syncGeneration;
    const arrival = value.arrivalSeq;
    const eventId = identifier(value.sourceEventId);
    const lineage =
      record(value.sourceLineage) && Object.keys(value.sourceLineage).length > 0
        ? value.sourceLineage
        : null;
    const kind = value.kind === "snapshot" || value.kind === "diff"
      ? value.kind
      : null;
    const bids = levels(
      kind === "snapshot" ? value.bids : value.bidMutations,
      maxLevels,
    );
    const asks = levels(
      kind === "snapshot" ? value.asks : value.askMutations,
      maxLevels,
    );
    const lastUpdateId = kind === "snapshot"
      ? integerId(value.lastUpdateId)
      : null;
    const firstUpdateId = kind === "diff"
      ? integerId(value.firstUpdateId)
      : null;
    const finalUpdateId = kind === "diff"
      ? integerId(value.finalUpdateId)
      : null;
    const valid = !!venue && !!symbol && !!session &&
      Number.isSafeInteger(connection) &&
      connection > 0 && Number.isSafeInteger(generation) && generation > 0 &&
      Number.isSafeInteger(arrival) && arrival > 0 && !!eventId && !!observed &&
      !!received && !!lineage && !!kind && !!bids && !!asks &&
      (kind === "snapshot"
        ? !!lastUpdateId
        : !!firstUpdateId && !!finalUpdateId &&
          BigInt(firstUpdateId!) <= BigInt(finalUpdateId!));
    if (!valid) {
      report.invalidEvidenceCount++;
      continue;
    }
    const row: Row = {
      kind: kind!,
      venue: venue!,
      symbol: symbol!,
      session: session!,
      connection: connection as number,
      generation: generation as number,
      arrival: arrival as number,
      eventId: eventId!,
      observedAt: observed!,
      receivedAt: received!,
      bids: bids!,
      asks: asks!,
      lineage: lineage!,
      ...(lastUpdateId ? { lastUpdateId } : {}),
      ...(firstUpdateId
        ? { firstUpdateId, finalUpdateId: finalUpdateId! }
        : {}),
    };
    const key = `${row.session}|${row.symbol}|${row.arrival}`;
    const fingerprint = stable(row);
    if (signatures.has(key) && signatures.get(key) !== fingerprint) {
      report.blockers.push("conflicting equal-arrival source evidence");
    } else {
      signatures.set(key, fingerprint);
      rows.push(row);
    }
  }
  if (!rows.length) {
    report.blockers.push("no valid decision-time L2 evidence");
    return report;
  }
  rows.sort((a, b) =>
    a.arrival - b.arrival || a.eventId.localeCompare(b.eventId)
  );
  const first = rows[0];
  report.venue = first.venue;
  report.symbol = first.symbol;
  if (
    rows.some((row) =>
      row.venue !== first.venue || row.symbol !== first.symbol ||
      row.session !== first.session || row.connection !== first.connection ||
      row.generation !== first.generation
    )
  ) {
    report.blockers.push("mixed L2 lineage");
    return report;
  }
  const arrivals = rows.map((row) => row.arrival);
  if (arrivals.some((value, index) => value !== index + 1)) {
    report.blockers.push("non-contiguous arrival sequence");
    return report;
  }
  const snapshotIndex = rows.findIndex((row) => row.kind === "snapshot");
  if (snapshotIndex < 0) {
    report.blockers.push("synchronized snapshot is required");
    return report;
  }
  if (snapshotIndex !== 0) {
    report.blockers.push("snapshot must precede diff evidence");
    return report;
  }
  const snapshot = rows[snapshotIndex];
  const bids = new Map<string, Level>();
  const asks = new Map<string, Level>();
  for (const level of snapshot.bids) bids.set(decimalKey(level.price), level);
  for (const level of snapshot.asks) asks.set(decimalKey(level.price), level);
  let updateId = BigInt(snapshot.lastUpdateId!);
  for (const row of rows.slice(snapshotIndex + 1)) {
    if (row.kind === "snapshot") {
      report.blockers.push("multiple snapshots in one decision lineage");
      return report;
    }
    const firstId = BigInt(row.firstUpdateId!);
    const finalId = BigInt(row.finalUpdateId!);
    if (finalId <= updateId) continue;
    if (firstId > updateId + 1n) {
      report.blockers.push("L2 sequence gap");
      return report;
    }
    for (const level of row.bids) {
      if (Number(level.size) === 0) bids.delete(decimalKey(level.price));
      else bids.set(decimalKey(level.price), level);
    }
    for (const level of row.asks) {
      if (Number(level.size) === 0) asks.delete(decimalKey(level.price));
      else asks.set(decimalKey(level.price), level);
    }
    updateId = finalId;
  }
  const bidLevels = sideBook(bids, "bid");
  const askLevels = sideBook(asks, "ask");
  if (!bidLevels.length || !askLevels.length) {
    report.blockers.push("both bid and ask depth are required");
    return report;
  }
  const bestBid = bidLevels[0].price;
  const bestAsk = askLevels[0].price;
  if (compareDecimal(bestAsk, bestBid) < 0) {
    report.blockers.push("crossed L2 book");
    return report;
  }
  const asOf = rows[rows.length - 1].receivedAt;
  const staleAfterMs = typeof options.staleAfterMs === "number" &&
      Number.isFinite(options.staleAfterMs) && options.staleAfterMs > 0
    ? options.staleAfterMs
    : 5 * 60_000;
  if (decision.ms - asOf.ms > staleAfterMs) {
    report.blockers.push("stale L2 evidence");
  }
  report.book = {
    bestBid,
    bestAsk,
    spreadBps: 10_000 * (Number(bestAsk) - Number(bestBid)) /
      ((Number(bestAsk) + Number(bestBid)) / 2),
    visibleBidLevels: bidLevels.length,
    visibleAskLevels: askLevels.length,
  };
  report.cost = compileL2Cost({
    side: options.side,
    notionalUsd: options.requestedNotionalUsd,
    feeBps: options.feeBps,
    bids: bidLevels,
    asks: askLevels,
  });
  report.provenance = {
    collectorSessionId: first.session,
    connectionGeneration: first.connection,
    syncGeneration: first.generation,
    sourceEventIds: rows.map((row) => row.eventId),
    arrivalSeq: arrivals,
    snapshotLastUpdateId: snapshot.lastUpdateId!,
    asOf: asOf.text,
  };
  report.cadence = {
    sampleCount: rows.length,
    firstArrivalAt: rows[0].receivedAt.text,
    lastArrivalAt: asOf.text,
    windowMs: asOf.ms - rows[0].receivedAt.ms,
  };
  if (report.cost && !report.cost.fillable) {
    report.blockers.push("insufficient visible depth");
  }
  if (report.invalidEvidenceCount) {
    report.blockers.push("invalid evidence present");
  }
  report.healthy = report.blockers.length === 1 &&
    report.blockers[0] ===
      "missing prospective multi-window shadow A/B evidence" &&
    !!report.cost?.fillable;
  report.classification = report.healthy
    ? "HEALTHY"
    : report.cost && report.cost.fillable === false
    ? "INSUFFICIENT_EVIDENCE"
    : "BLOCKED";
  return report;
}

export const compileMarketL2CapabilityGap = compileCapabilityGapMarketL2;
