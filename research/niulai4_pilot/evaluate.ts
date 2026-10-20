// Offline evaluation of recorded pilot data (pure). Inputs are exactly what the recorder wrote:
// NIULAI4 observations (with Brian's own first-seen time), order-book captures taken at
// first_seen + d, the window's fee rate, and the actual resolution. Nothing here looks at a price
// Brian could not have seen at the decision time.
//
// Strategies, all on the same window, the same decision instant, the same size and cost model:
//   FOLLOW  -- buy the outcome NIULAI4 bought
//   B0_FAVORITE        -- buy the outcome with the higher mid
//   B1_SPOT_DIRECTION  -- buy Up if the spot 60s TWAP proxy >= spot at window start, else Down
//   B2_RANDOM          -- seeded coin flip (cost floor)
// Every strategy keeps its own virtual bankroll with cost + fee reserved until settlement.

import { polymarketTakerFeeUsdc } from "../../supabase/functions/_shared/prediction_market.ts";
import { type Level, walkAsks } from "./book.ts";
import { RULES } from "./rules.ts";
import { clusterBootstrapMean, mulberry32, quantiles } from "./stats.ts";

export type Outcome = "Up" | "Down";
export type Strategy = "FOLLOW" | "B0_FAVORITE" | "B1_SPOT_DIRECTION" | "B2_RANDOM";
export const STRATEGIES: Strategy[] = ["FOLLOW", "B0_FAVORITE", "B1_SPOT_DIRECTION", "B2_RANDOM"];

export interface Observation {
  obsId: string; // idempotency key of the public trade row
  windowSlug: string;
  windowStartSec: number;
  windowEndSec: number;
  outcome: Outcome;
  tradeTsMs: number; // public `timestamp` (second resolution)
  firstSeenMs: number; // Brian's own skew-corrected receive time
}

export interface Capture {
  obsId: string;
  delaySeconds: number;
  capturedMs: number;
  asks: Record<Outcome, Level[]> | null; // null when the book was not synced at that instant
  mids: Record<Outcome, number | null> | null;
  spotDirection: Outcome | null;
}

export interface WindowInfo {
  slug: string;
  feeRate: number | null;
  winner: Outcome | null; // null = not (yet) resolved
}

export type EntryStatus =
  | "SETTLED"
  | "LATE"
  | "NO_CAPTURE"
  | "BOOK_UNSYNCED"
  | "NO_DECISION"
  | "UNFILLABLE"
  | "NO_FEE_RATE"
  | "REJECTED_CAPITAL"
  | "UNRESOLVED";

export interface Entry {
  obsId: string;
  window: string;
  day: string;
  strategy: Strategy;
  delaySeconds: number;
  status: EntryStatus;
  outcome?: Outcome;
  cost?: number;
  fee?: number;
  slippage?: number;
  payoff?: number;
  net?: number;
  returnOnCost?: number;
}

function feeForFill(asks: Level[], shares: number, tolerance: number, rate: number): number {
  // Fee is charged per matched level at that level's price.
  const sorted = [...asks].sort((a, b) => a.price - b.price);
  const limit = sorted[0].price + tolerance + 1e-9;
  let remaining = shares;
  let fee = 0;
  for (const lvl of sorted) {
    if (lvl.price > limit || remaining <= 1e-9) break;
    const take = Math.min(remaining, lvl.size);
    fee += polymarketTakerFeeUsdc(take, lvl.price, rate);
    remaining -= take;
  }
  return Math.round(fee * 1e5) / 1e5;
}

export function evaluate(observations: Observation[], captures: Capture[], windows: WindowInfo[]): {
  entries: Entry[];
  latencySeconds: { p50: number | null; p90: number | null; max: number | null; n: number };
} {
  const capByKey = new Map(captures.map((c) => [`${c.obsId}|${c.delaySeconds}`, c]));
  const winBySlug = new Map(windows.map((w) => [w.slug, w]));
  const rand = mulberry32(RULES.randomSeed);
  const candidates: (Entry & { decisionMs: number; settleMs: number; fill?: ReturnType<typeof walkAsks> })[] = [];
  const obs = [...observations].sort((a, b) => a.firstSeenMs - b.firstSeenMs || a.obsId.localeCompare(b.obsId));

  for (const o of obs) {
    const day = new Date(o.windowStartSec * 1000).toISOString().slice(0, 10);
    const w = winBySlug.get(o.windowSlug);
    for (const d of RULES.delaysSeconds) {
      const decisionMs = o.firstSeenMs + d * 1000;
      const cap = capByKey.get(`${o.obsId}|${d}`);
      // Draw the coin for every (observation, delay) so B2 does not depend on which other rows exist.
      const coin: Outcome = rand() < 0.5 ? "Up" : "Down";
      for (const strategy of STRATEGIES) {
        const base = { obsId: o.obsId, window: o.windowSlug, day, strategy, delaySeconds: d, decisionMs, settleMs: o.windowEndSec * 1000 };
        if (decisionMs >= o.windowEndSec * 1000) {
          candidates.push({ ...base, status: "LATE" });
          continue;
        }
        if (!cap) {
          candidates.push({ ...base, status: "NO_CAPTURE" });
          continue;
        }
        if (!cap.asks || !cap.mids) {
          candidates.push({ ...base, status: "BOOK_UNSYNCED" });
          continue;
        }
        let pick: Outcome | null = null;
        if (strategy === "FOLLOW") pick = o.outcome;
        else if (strategy === "B0_FAVORITE") {
          const u = cap.mids.Up;
          const dn = cap.mids.Down;
          pick = u === null || dn === null || u === dn ? null : u > dn ? "Up" : "Down";
        } else if (strategy === "B1_SPOT_DIRECTION") pick = cap.spotDirection;
        else pick = coin;
        if (pick === null) {
          candidates.push({ ...base, status: "NO_DECISION" });
          continue;
        }
        const fill = walkAsks(cap.asks[pick], RULES.followSizeShares, RULES.priceToleranceAboveBestAsk);
        if (fill.status !== "FILLED") {
          candidates.push({ ...base, status: "UNFILLABLE", outcome: pick });
          continue;
        }
        if (!w || w.feeRate === null) {
          candidates.push({ ...base, status: "NO_FEE_RATE", outcome: pick });
          continue;
        }
        const cost = Math.round(fill.avgPrice * fill.shares * 1e6) / 1e6;
        const fee = feeForFill(cap.asks[pick], fill.shares, RULES.priceToleranceAboveBestAsk, w.feeRate);
        candidates.push({ ...base, status: "SETTLED", outcome: pick, cost, fee, slippage: fill.slippage, fill });
      }
    }
  }

  // Capital: per (strategy, delay) ledger, entries in decision order, reservations released at
  // the window end. An entry that does not fit the remaining cash is REJECTED_CAPITAL.
  const entries: Entry[] = [];
  const groups = new Map<string, typeof candidates>();
  for (const c of candidates) {
    const k = `${c.strategy}|${c.delaySeconds}`;
    const g = groups.get(k) ?? [];
    g.push(c);
    groups.set(k, g);
  }
  for (const g of groups.values()) {
    g.sort((a, b) => a.decisionMs - b.decisionMs || a.obsId.localeCompare(b.obsId));
    let cash = RULES.bankrollUsdc;
    const open: { settleMs: number; reserved: number; credit: number }[] = [];
    for (const c of g) {
      for (let i = open.length - 1; i >= 0; i--) {
        if (open[i].settleMs <= c.decisionMs) {
          cash += open[i].credit;
          open.splice(i, 1);
        }
      }
      const { decisionMs: _d, settleMs: _s, fill: _f, ...entry } = c;
      if (c.status !== "SETTLED") {
        entries.push(entry);
        continue;
      }
      const need = (c.cost ?? 0) + (c.fee ?? 0);
      if (need > cash + 1e-9) {
        entries.push({ ...entry, status: "REJECTED_CAPITAL", cost: undefined, fee: undefined, slippage: undefined });
        continue;
      }
      const winner = winBySlug.get(c.window)?.winner ?? null;
      if (winner === null) {
        // Reserve but do not settle: unresolved windows hold capital and are excluded from returns.
        cash -= need;
        open.push({ settleMs: Number.POSITIVE_INFINITY, reserved: need, credit: 0 });
        entries.push({ ...entry, status: "UNRESOLVED" });
        continue;
      }
      const payoff = c.outcome === winner ? RULES.followSizeShares : 0;
      const net = Math.round((payoff - need) * 1e6) / 1e6;
      cash -= need;
      open.push({ settleMs: c.settleMs, reserved: need, credit: payoff });
      entries.push({ ...entry, payoff, net, returnOnCost: Math.round((net / need) * 1e6) / 1e6 });
    }
  }

  const lags = observations.map((o) => (o.firstSeenMs - o.tradeTsMs) / 1000);
  const [p50, p90] = quantiles(lags, [0.5, 0.9]);
  return { entries, latencySeconds: { p50, p90, max: lags.length ? Math.max(...lags) : null, n: lags.length } };
}

export interface PilotSummary {
  rulesVersion: string;
  observations: number;
  windows: number;
  days: number;
  latencySeconds: ReturnType<typeof evaluate>["latencySeconds"];
  byStrategyDelay: Record<string, Record<EntryStatus | "meanReturnOnCost", number | null>>;
  primary: {
    delaySeconds: number;
    baseline: Strategy;
    pairedObservations: number;
    windowClustered: ReturnType<typeof clusterBootstrapMean>;
    dayClustered: ReturnType<typeof clusterBootstrapMean>;
  };
  feasibility: { fillableShareAtPrimaryDelay: number | null; verdict: "PASS" | "FAIL" | "INSUFFICIENT_DATA" };
  edge: { verdict: "EDGE_SUPPORTED" | "NO_EDGE" | "INSUFFICIENT_DATA"; reason: string };
  tradeSignal: false;
}

export function summarize(observations: Observation[], result: ReturnType<typeof evaluate>, rulesVersion: string): PilotSummary {
  const { entries } = result;
  const by: PilotSummary["byStrategyDelay"] = {};
  for (const s of STRATEGIES) {
    for (const d of RULES.delaysSeconds) {
      const rows = entries.filter((e) => e.strategy === s && e.delaySeconds === d);
      const counts: Record<string, number | null> = {};
      for (const st of ["SETTLED", "LATE", "NO_CAPTURE", "BOOK_UNSYNCED", "NO_DECISION", "UNFILLABLE", "NO_FEE_RATE", "REJECTED_CAPITAL", "UNRESOLVED"]) {
        counts[st] = rows.filter((e) => e.status === st).length;
      }
      const rets = rows.filter((e) => e.status === "SETTLED").map((e) => e.returnOnCost!);
      counts.meanReturnOnCost = rets.length ? Math.round((rets.reduce((a, b) => a + b, 0) / rets.length) * 1e6) / 1e6 : null;
      by[`${s}@${d}s`] = counts as PilotSummary["byStrategyDelay"][string];
    }
  }

  // Primary: per observation, FOLLOW return minus B1 return at the primary delay, only where BOTH
  // settled -- otherwise the comparison would mix different subsets.
  const d = RULES.primary.delaySeconds;
  const baseline = RULES.primary.baseline;
  const key = (e: Entry) => `${e.obsId}`;
  const follow = new Map(entries.filter((e) => e.strategy === "FOLLOW" && e.delaySeconds === d && e.status === "SETTLED").map((e) => [key(e), e]));
  const base = new Map(entries.filter((e) => e.strategy === baseline && e.delaySeconds === d && e.status === "SETTLED").map((e) => [key(e), e]));
  const paired = [...follow.keys()].filter((k) => base.has(k)).map((k) => ({
    window: follow.get(k)!.window,
    day: follow.get(k)!.day,
    diff: follow.get(k)!.returnOnCost! - base.get(k)!.returnOnCost!,
  }));
  const windowClustered = clusterBootstrapMean(paired.map((p) => ({ cluster: p.window, value: p.diff })), { seed: RULES.randomSeed });
  const dayClustered = clusterBootstrapMean(paired.map((p) => ({ cluster: p.day, value: p.diff })), { seed: RULES.randomSeed + 1 });

  const primaryFollow = entries.filter((e) => e.strategy === "FOLLOW" && e.delaySeconds === d);
  const attempted = primaryFollow.filter((e) => e.status !== "NO_CAPTURE" && e.status !== "BOOK_UNSYNCED");
  const fillable = attempted.filter((e) => ["SETTLED", "UNRESOLVED", "REJECTED_CAPITAL", "NO_FEE_RATE"].includes(e.status));
  const share = attempted.length ? fillable.length / attempted.length : null;
  const windows = new Set(observations.map((o) => o.windowSlug)).size;
  const days = new Set(observations.map((o) => new Date(o.windowStartSec * 1000).toISOString().slice(0, 10))).size;

  // Feasibility thresholds from the plan: PASS >= 30% fillable at d=5 with enough attempts;
  // FAIL < 10%; otherwise (or too few attempts) INSUFFICIENT_DATA.
  let feasibility: PilotSummary["feasibility"]["verdict"] = "INSUFFICIENT_DATA";
  if (share !== null && attempted.length >= 100) feasibility = share >= 0.3 ? "PASS" : share < 0.1 ? "FAIL" : "INSUFFICIENT_DATA";

  let edge: PilotSummary["edge"] = { verdict: "INSUFFICIENT_DATA", reason: "" };
  if (windowClustered.clusters < 1000 || dayClustered.clusters < 14) {
    edge = { verdict: "INSUFFICIENT_DATA", reason: `needs >= 1000 independent windows and >= 14 days (have ${windowClustered.clusters} / ${dayClustered.clusters})` };
  } else if (windowClustered.lower !== null && dayClustered.lower !== null && windowClustered.lower > 0 && dayClustered.lower > 0) {
    edge = { verdict: "EDGE_SUPPORTED", reason: "window- and day-clustered 95% lower bounds both > 0" };
  } else if (windowClustered.upper !== null && windowClustered.upper < 0) {
    edge = { verdict: "NO_EDGE", reason: "window-clustered 95% upper bound < 0" };
  } else {
    edge = { verdict: "INSUFFICIENT_DATA", reason: "interval contains 0" };
  }

  return {
    rulesVersion,
    observations: observations.length,
    windows,
    days,
    latencySeconds: result.latencySeconds,
    byStrategyDelay: by,
    primary: { delaySeconds: d, baseline, pairedObservations: paired.length, windowClustered, dayClustered },
    feasibility: { fillableShareAtPrimaryDelay: share, verdict: feasibility },
    edge,
    tradeSignal: false,
  };
}
