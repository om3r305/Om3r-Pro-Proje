// Brian ALPHA v2 Dynamic Cost/Slippage Compiler.
// Pure/import-safe. No I/O, no Supabase client, no order placement.
// Source L2 decimals stay exact strings in persistence; this derived calculator validates and
// converts only for cost arithmetic. Every result is SHADOW-only metadata.

export type CostSide = "BUY" | "SELL";
export type CostQuality = "L2_OBSERVED" | "DEGRADED_TOP_OF_BOOK";
export type DecimalLevel = { price: string; size: string };

export interface DynamicCostQuote {
  side: CostSide;
  requestedNotionalUsd: number;
  filledNotionalUsd: number;
  filledBaseQty: number;
  fillRatio: number;
  fillable: boolean;
  referenceMid: number;
  bestBid: number;
  bestAsk: number;
  vwap: number | null;
  feeBps: number;
  spreadBps: number;
  halfSpreadBps: number;
  depthSlippageBps: number;
  oneWayCostBps: number;
  estimatedRoundTripCostBps: number;
  quality: CostQuality;
  levelsConsumed: number;
  reason: string;
}

function finitePositive(value: unknown, label: string): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n) || !(n > 0)) throw new Error(`${label} must be a finite positive number`);
  return n;
}

function finiteNonnegative(value: unknown, label: string): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) throw new Error(`${label} must be a finite nonnegative number`);
  return n;
}

function parseLevel(level: DecimalLevel, label: string): { price: number; size: number } {
  if (!level || typeof level.price !== "string" || typeof level.size !== "string") {
    throw new Error(`${label} must preserve venue decimal strings`);
  }
  const price = finitePositive(level.price, `${label}.price`);
  const size = finiteNonnegative(level.size, `${label}.size`);
  return { price, size };
}

function validatedBook(bids: DecimalLevel[], asks: DecimalLevel[]) {
  if (!Array.isArray(bids) || !Array.isArray(asks) || !bids.length || !asks.length) {
    throw new Error("both bid and ask depth are required");
  }
  const parsedBids = bids.map((x, i) => parseLevel(x, `bids[${i}]`)).filter((x) => x.size > 0);
  const parsedAsks = asks.map((x, i) => parseLevel(x, `asks[${i}]`)).filter((x) => x.size > 0);
  if (!parsedBids.length || !parsedAsks.length) throw new Error("book must contain non-zero bid and ask liquidity");

  parsedBids.sort((a, b) => b.price - a.price);
  parsedAsks.sort((a, b) => a.price - b.price);
  const bestBid = parsedBids[0].price;
  const bestAsk = parsedAsks[0].price;
  if (!(bestAsk >= bestBid)) throw new Error("crossed L2 book is not trustworthy");
  const referenceMid = (bestBid + bestAsk) / 2;
  return { parsedBids, parsedAsks, bestBid, bestAsk, referenceMid };
}

/**
 * Walk visible L2 depth for a quote-notional market order.
 *
 * Cost decomposition deliberately avoids double-counting:
 * - halfSpreadBps: mid -> best quote
 * - depthSlippageBps: best quote -> depth-walk VWAP
 * - feeBps: explicit venue fee assumption
 * oneWayCostBps = fee + half spread + depth slippage.
 *
 * This is a shadow estimate, not a promise of execution quality. Queue movement/latency/adverse
 * selection are not inferred from a static book and must be modeled separately later.
 */
export function compileL2Cost(params: {
  side: CostSide;
  notionalUsd: number;
  feeBps: number;
  bids: DecimalLevel[];
  asks: DecimalLevel[];
}): DynamicCostQuote {
  const requested = finitePositive(params.notionalUsd, "notionalUsd");
  const feeBps = finiteNonnegative(params.feeBps, "feeBps");
  const { parsedBids, parsedAsks, bestBid, bestAsk, referenceMid } = validatedBook(params.bids, params.asks);
  const levels = params.side === "BUY" ? parsedAsks : parsedBids;
  const best = params.side === "BUY" ? bestAsk : bestBid;

  let remaining = requested;
  let filledQuote = 0;
  let filledBase = 0;
  let levelsConsumed = 0;
  for (const level of levels) {
    if (remaining <= 1e-12) break;
    const availableQuote = level.price * level.size;
    if (!(availableQuote > 0)) continue;
    const takeQuote = Math.min(remaining, availableQuote);
    const takeBase = takeQuote / level.price;
    filledQuote += takeQuote;
    filledBase += takeBase;
    remaining -= takeQuote;
    levelsConsumed += 1;
  }

  const fillRatio = Math.min(1, Math.max(0, filledQuote / requested));
  const fillable = remaining <= Math.max(1e-8, requested * 1e-10);
  const vwap = filledBase > 0 ? filledQuote / filledBase : null;
  const spreadBps = 10000 * (bestAsk - bestBid) / Math.max(referenceMid, 1e-12);
  const halfSpreadBps = spreadBps / 2;
  let depthSlippageBps = 0;
  if (vwap !== null) {
    depthSlippageBps = params.side === "BUY"
      ? 10000 * Math.max(0, vwap - bestAsk) / bestAsk
      : 10000 * Math.max(0, bestBid - vwap) / bestBid;
  }
  const oneWayCostBps = feeBps + halfSpreadBps + depthSlippageBps;

  return {
    side: params.side,
    requestedNotionalUsd: requested,
    filledNotionalUsd: filledQuote,
    filledBaseQty: filledBase,
    fillRatio,
    fillable,
    referenceMid,
    bestBid,
    bestAsk,
    vwap,
    feeBps,
    spreadBps,
    halfSpreadBps,
    depthSlippageBps,
    oneWayCostBps,
    estimatedRoundTripCostBps: oneWayCostBps * 2,
    quality: "L2_OBSERVED",
    levelsConsumed,
    reason: fillable ? "visible L2 depth fully covers requested notional" : "visible L2 depth is insufficient for requested notional",
  };
}

/** Explicit degraded fallback when a synchronized L2 book is unavailable. */
export function compileDegradedTopOfBookCost(params: {
  side: CostSide;
  notionalUsd: number;
  feeBps: number;
  spreadBps: number;
  assumedSlippageBps: number;
  midPrice: number;
}): DynamicCostQuote {
  const requested = finitePositive(params.notionalUsd, "notionalUsd");
  const feeBps = finiteNonnegative(params.feeBps, "feeBps");
  const spreadBps = finiteNonnegative(params.spreadBps, "spreadBps");
  const slippage = finiteNonnegative(params.assumedSlippageBps, "assumedSlippageBps");
  const mid = finitePositive(params.midPrice, "midPrice");
  const half = spreadBps / 2;
  const bestAsk = mid * (1 + half / 10000);
  const bestBid = mid * (1 - half / 10000);
  const oneWay = feeBps + half + slippage;
  const vwap = params.side === "BUY"
    ? bestAsk * (1 + slippage / 10000)
    : bestBid * (1 - slippage / 10000);

  return {
    side: params.side,
    requestedNotionalUsd: requested,
    filledNotionalUsd: requested,
    filledBaseQty: requested / vwap,
    fillRatio: 1,
    fillable: true,
    referenceMid: mid,
    bestBid,
    bestAsk,
    vwap,
    feeBps,
    spreadBps,
    halfSpreadBps: half,
    depthSlippageBps: slippage,
    oneWayCostBps: oneWay,
    estimatedRoundTripCostBps: oneWay * 2,
    quality: "DEGRADED_TOP_OF_BOOK",
    levelsConsumed: 1,
    reason: "synchronized L2 unavailable; conservative top-of-book fallback",
  };
}
