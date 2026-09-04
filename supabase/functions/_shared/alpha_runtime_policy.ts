import {
  compileDegradedTopOfBookCost,
  type CostSide,
  type DynamicCostQuote,
} from "./dynamic_cost.ts";

export type RuntimeL2Status = "OBSERVED" | "UNAVAILABLE" | "INVALID" | "NOT_REQUESTED";
export type RuntimeReferenceBook = { mid: number; spreadBps: number };

/**
 * Runtime cost selection policy.
 *
 * Only an actually unavailable L2 source (transport/HTTP outage) may degrade to top-of-book.
 * If a 2xx depth snapshot was observed but failed schema/book validation, degrading would turn
 * incomplete/untrustworthy visible depth into fillable=true. INVALID therefore fails closed.
 */
export function selectAlphaRuntimeCost(params: {
  l2Status: RuntimeL2Status;
  observedL2Quote?: DynamicCostQuote | null;
  referenceBook?: RuntimeReferenceBook | null;
  side: CostSide;
  notionalUsd: number;
  feeBps: number;
  fallbackSlippageBps: number;
}): DynamicCostQuote | null {
  if (params.l2Status === "OBSERVED") return params.observedL2Quote ?? null;
  if (params.l2Status !== "UNAVAILABLE" || !params.referenceBook) return null;
  return compileDegradedTopOfBookCost({
    side: params.side,
    notionalUsd: params.notionalUsd,
    feeBps: params.feeBps,
    spreadBps: params.referenceBook.spreadBps,
    assumedSlippageBps: params.fallbackSlippageBps,
    midPrice: params.referenceBook.mid,
  });
}
