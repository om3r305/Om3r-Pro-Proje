// Offline evaluation of recorded pilot data (pure). Inputs are exactly what the recorder wrote:
// NIULAI4 observations (with Brian's own first-seen time), order-book captures, the window's fee
// rate, and the actual resolution. Nothing here uses information Brian did not have at the
// decision instant:
//   * observations are processed in first-seen order and each decision uses only that
//     observation and the capture scheduled for it -- a later-seen trade (e.g. the other side of
//     the same window) can never change an earlier decision;
//   * a capture taken later than scheduled + maxCaptureLagMs, earlier than scheduled, from a book
//     that was not freshly re-synced, or after a socket silence, is rejected (never "repaired");
//   * cost is the sum over the actually walked levels, fee is charged per walked level, and
//     slippage is REPORTED only (it is already inside the cost, never subtracted twice).
//
// Strategies, conditional on the instants at which NIULAI4 trades (same window, same instant,
// same size, same cost model). This answers ONLY "is following realizable and does it beat
// simple rules at those instants". The wallet-independent H2 question is NOT tested here.
//   FOLLOW             -- buy the outcome NIULAI4 bought
//   B0_FAVORITE        -- buy the outcome with the higher mid
//   B1_SPOT_DIRECTION  -- buy Up if the spot 60s TWAP proxy >= spot at window start, else Down
//   B2_RANDOM          -- deterministic coin from hash(obsId, delay): order-independent
// Each (strategy, delay) pair has its own virtual bankroll; cost + fee is reserved until the
// window settles, so repeated entries in one bankroll share capital.

import { polymarketTakerFeeUsdc } from "../../supabase/functions/_shared/prediction_market.ts";
import { type Level, walkAsks } from "./book.ts";
import { RULES } from "./rules.ts";
import { type BootstrapResult, clusterBootstrapMean, quantiles } from "./stats.ts";

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
  firstSeenMs: number; // Brian's own clock-corrected receive time
}

export interface Capture {
  obsId: string;
  delaySeconds: number;
  scheduledMs: number; // firstSeen + d + orderLatency
  capturedMs: number; // when the snapshot was actually taken (local, clock-corrected)
  socketSilenceMs: number | null; // time since the last market-socket message at capture
  asks: Record<Outcome, Level[]> | null; // null when either book was not synced since (re)connect
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
  | "LATE_WINDOW_CLOSED"
  | "NO_CAPTURE"
  | "CAPTURE_TIME_INVALID"
  | "STALE_BOOK"
  | "BOOK_UNSYNCED"
  | "NO_DECISION"
  | "UNFILLABLE"
  | "NO_FEE_RATE"
  | "REJECTED_CAPITAL"
  | "UNRESOLVED";

export const ENTRY_STATUSES: EntryStatus[] = [
  "SETTLED", "LATE_WINDOW_CLOSED", "NO_CAPTURE", "CAPTURE_TIME_INVALID", "STALE_BOOK", "BOOK_UNSYNCED",
  "NO_DECISION", "UNFILLABLE", "NO_FEE_RATE", "REJECTED_CAPITAL", "UNRESOLVED",
];

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

// FNV-1a 32-bit, so the random baseline's coin depends only on (obsId, delay).
function fnv1a(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}
export function coinFor(obsId: string, delaySeconds: number): Outcome {
  return (fnv1a(`${RULES.randomSeed}|${obsId}|${delaySeconds}`) & 1) === 0 ? "Up" : "Down";
}

// Walk the asks once and charge cost and fee per walked level.
function costAndFee(asks: Level[], shares: number, tolerance: number, rate: number): { cost: number; fee: number } {
  const sorted = [...asks].sort((a, b) => a.price - b.price);
  const limit = sorted[0].price + tolerance + 1e-9;
  let remaining = shares;
  let cost = 0;
  let fee = 0;
  for (const lvl of sorted) {
    if (lvl.price > limit || remaining <= 1e-9) break;
    const take = Math.min(remaining, lvl.size);
    cost += take * lvl.price;
    fee += polymarketTakerFeeUsdc(take, lvl.price, rate);
    remaining -= take;
  }
  return { cost: Math.round(cost * 1e6) / 1e6, fee: Math.round(fee * 1e5) / 1e5 };
}

export function scheduledCaptureMs(firstSeenMs: number, delaySeconds: number): number {
  return firstSeenMs + delaySeconds * 1000 + RULES.orderLatencyMs;
}

export function evaluate(observations: Observation[], captures: Capture[], windows: WindowInfo[]): { entries: Entry[] } {
  const capByKey = new Map(captures.map((c) => [`${c.obsId}|${c.delaySeconds}`, c]));
  const winBySlug = new Map(windows.map((w) => [w.slug, w]));
  type Cand = Entry & { decisionMs: number; settleMs: number };
  const candidates: Cand[] = [];
  // First-seen order: what Brian knew first is decided first.
  const obs = [...observations].sort((a, b) => a.firstSeenMs - b.firstSeenMs || a.obsId.localeCompare(b.obsId));

  for (const o of obs) {
    const day = new Date(o.windowStartSec * 1000).toISOString().slice(0, 10);
    const w = winBySlug.get(o.windowSlug);
    for (const d of RULES.delaysSeconds) {
      const scheduled = scheduledCaptureMs(o.firstSeenMs, d);
      const cap = capByKey.get(`${o.obsId}|${d}`);
      for (const strategy of STRATEGIES) {
        const base = { obsId: o.obsId, window: o.windowSlug, day, strategy, delaySeconds: d, decisionMs: scheduled, settleMs: o.windowEndSec * 1000 };
        // The order must reach the venue before the window closes.
        if (scheduled >= o.windowEndSec * 1000) {
          candidates.push({ ...base, status: "LATE_WINDOW_CLOSED" });
          continue;
        }
        if (!cap) {
          candidates.push({ ...base, status: "NO_CAPTURE" });
          continue;
        }
        if (cap.scheduledMs !== scheduled || cap.capturedMs < scheduled || cap.capturedMs > scheduled + RULES.maxCaptureLagMs) {
          candidates.push({ ...base, status: "CAPTURE_TIME_INVALID" });
          continue;
        }
        if (cap.socketSilenceMs === null || cap.socketSilenceMs > RULES.maxSocketSilenceMs) {
          candidates.push({ ...base, status: "STALE_BOOK" });
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
        else pick = coinFor(o.obsId, d);
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
        const { cost, fee } = costAndFee(cap.asks[pick], fill.shares, RULES.priceToleranceAboveBestAsk, w.feeRate);
        candidates.push({ ...base, status: "SETTLED", outcome: pick, cost, fee, slippage: fill.slippage });
      }
    }
  }

  // Capital: one ledger per (strategy, delay); entries in decision order; reservations released
  // only when their window settles. Unresolved windows keep their capital locked.
  const entries: Entry[] = [];
  const groups = new Map<string, Cand[]>();
  for (const c of candidates) {
    const k = `${c.strategy}|${c.delaySeconds}`;
    const g = groups.get(k) ?? [];
    g.push(c);
    groups.set(k, g);
  }
  for (const g of groups.values()) {
    g.sort((a, b) => a.decisionMs - b.decisionMs || a.obsId.localeCompare(b.obsId));
    let cash = RULES.bankrollUsdc;
    const open: { settleMs: number; credit: number }[] = [];
    for (const c of g) {
      for (let i = open.length - 1; i >= 0; i--) {
        if (open[i].settleMs <= c.decisionMs) {
          cash += open[i].credit;
          open.splice(i, 1);
        }
      }
      const { decisionMs: _d, settleMs: _s, ...entry } = c;
      if (c.status !== "SETTLED") {
        entries.push(entry);
        continue;
      }
      const need = (c.cost ?? 0) + (c.fee ?? 0);
      if (need > cash + 1e-9) {
        entries.push({ ...entry, status: "REJECTED_CAPITAL", cost: undefined, fee: undefined, slippage: undefined });
        continue;
      }
      cash -= need;
      const winner = winBySlug.get(c.window)?.winner ?? null;
      if (winner === null) {
        open.push({ settleMs: Number.POSITIVE_INFINITY, credit: 0 });
        entries.push({ ...entry, status: "UNRESOLVED" });
        continue;
      }
      const payoff = c.outcome === winner ? RULES.followSizeShares : 0;
      const net = Math.round((payoff - need) * 1e6) / 1e6;
      open.push({ settleMs: c.settleMs, credit: payoff });
      entries.push({ ...entry, payoff, net, returnOnCost: Math.round((net / need) * 1e6) / 1e6 });
    }
  }
  return { entries };
}

// ---------------------------------------------------------------------------------------------
// Summary. Three questions are reported SEPARATELY and never merged into one success flag:
//   feasibility        -- at the primary delay, how often is following realizable at all?
//   absoluteNet        -- is FOLLOW's own expected net return on cost > 0 (after fees/slippage)?
//   baselineSuperiority-- is FOLLOW better than the pre-registered baseline on the same instants?
// An edge is supported only if BOTH absoluteNet and baselineSuperiority are supported, at the
// pre-registered final analysis, with the pre-registered sample size reached. Follow -2% vs
// baseline -5% is a +3% difference but NOT an edge.
// ---------------------------------------------------------------------------------------------

export type Verdict = "SUPPORTED" | "NOT_SUPPORTED" | "INSUFFICIENT_DATA" | "INTERIM_NO_VERDICT";

export interface SummaryOptions {
  // Only the single pre-registered final analysis may carry a verdict; every earlier report is
  // descriptive (no peeking-based early success).
  finalAnalysis: boolean;
  // Required number of independent windows, frozen from the PILOT before the test period
  // (see requiredWindows). Without it no verdict is possible.
  frozenRequiredWindows: number | null;
}

export interface PilotSummary {
  rulesVersion: string;
  question: "CONDITIONAL_FOLLOW_REALIZABILITY_AND_COMPARISON";
  h2WalletIndependentTiming: { status: "NOT_TESTED"; reason: string };
  observations: number;
  windows: number;
  days: number;
  latency: {
    measured: number;
    secondsP50: number | null;
    secondsP90: number | null;
    secondsMax: number | null;
    seenAfterWindowClosed: number; // missed entirely: the window was over when Brian saw the trade
    lateForPrimaryDelay: number; // order would arrive after the window end at the primary delay
    note: string;
  };
  byStrategyDelay: Record<string, Record<string, number | null | string>>;
  feasibility: { attempts: number; fillableShare: number | null; verdict: "PASS" | "FAIL" | "INCONCLUSIVE_REDESIGN" | "INSUFFICIENT_DATA" };
  absoluteNet: { verdict: Verdict; windowClustered: BootstrapResult; dayClustered: BootstrapResult };
  baselineSuperiority: { baseline: Strategy; pairedObservations: number; verdict: Verdict; windowClustered: BootstrapResult; dayClustered: BootstrapResult };
  power: { minEffect: number; perWindowSdFollow: number | null; perWindowSdDiff: number | null; suggestedRequiredWindows: number | null; frozenRequiredWindows: number | null; note: string };
  edge: { verdict: Verdict; reason: string };
  tradeSignal: false;
}

function perWindowMeans(rows: { window: string; v: number }[]): number[] {
  const m = new Map<string, number[]>();
  for (const r of rows) {
    const g = m.get(r.window) ?? [];
    g.push(r.v);
    m.set(r.window, g);
  }
  return [...m.values()].map((g) => g.reduce((a, b) => a + b, 0) / g.length);
}
function sd(xs: number[]): number | null {
  if (xs.length < 2) return null;
  const mu = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, b) => a + (b - mu) ** 2, 0) / (xs.length - 1));
}
const Z = { 0.05: 1.959964, 0.8: 0.841621 } as const;

// Windows needed so that a two-sided test at alpha with the given power detects minEffect, using
// the standard deviation of PER-WINDOW means (which already folds in within-window dependence and
// varying participation per window). Day dependence is checked separately by the day-clustered
// interval at analysis time.
export function requiredWindows(perWindowSd: number | null, minEffect = RULES.minEffectReturnOnCost): number | null {
  if (perWindowSd === null || !(perWindowSd > 0) || !(minEffect > 0)) return null;
  return Math.ceil(((Z[0.05] + Z[0.8]) * perWindowSd / minEffect) ** 2);
}

function verdictFor(win: BootstrapResult, day: BootstrapResult, gate: { ok: boolean; reason: string }, final: boolean): Verdict {
  if (!final) return "INTERIM_NO_VERDICT";
  if (!gate.ok) return "INSUFFICIENT_DATA";
  if (win.lower !== null && day.lower !== null && win.lower > 0 && day.lower > 0) return "SUPPORTED";
  if (win.upper !== null && win.upper < 0) return "NOT_SUPPORTED";
  return "INSUFFICIENT_DATA";
}

export function summarize(observations: Observation[], result: ReturnType<typeof evaluate>, rulesVersion: string, options: SummaryOptions): PilotSummary {
  const { entries } = result;
  const d = RULES.primary.delaySeconds;
  const baseline = RULES.primary.baseline;

  const by: PilotSummary["byStrategyDelay"] = {};
  for (const s of STRATEGIES) {
    for (const delay of RULES.delaysSeconds) {
      const rows = entries.filter((e) => e.strategy === s && e.delaySeconds === delay);
      const counts: Record<string, number | null | string> = {};
      for (const st of ENTRY_STATUSES) counts[st] = rows.filter((e) => e.status === st).length;
      const settled = rows.filter((e) => e.status === "SETTLED");
      counts.meanReturnOnCost = settled.length ? Math.round((settled.reduce((a, e) => a + e.returnOnCost!, 0) / settled.length) * 1e6) / 1e6 : null;
      counts.meanSlippage = settled.length ? Math.round((settled.reduce((a, e) => a + (e.slippage ?? 0), 0) / settled.length) * 1e6) / 1e6 : null;
      counts.meanFee = settled.length ? Math.round((settled.reduce((a, e) => a + (e.fee ?? 0), 0) / settled.length) * 1e6) / 1e6 : null;
      if (delay === 0) counts.label = "OPTIMISTIC_UPPER_BOUND";
      by[`${s}@${delay}s`] = counts;
    }
  }

  // Feasibility at the primary delay.
  const primaryFollow = entries.filter((e) => e.strategy === "FOLLOW" && e.delaySeconds === d);
  const attempted = primaryFollow.filter((e) => !["NO_CAPTURE", "CAPTURE_TIME_INVALID", "STALE_BOOK", "BOOK_UNSYNCED"].includes(e.status));
  const fillable = attempted.filter((e) => ["SETTLED", "UNRESOLVED", "REJECTED_CAPITAL", "NO_FEE_RATE"].includes(e.status));
  const share = attempted.length ? fillable.length / attempted.length : null;
  const F = RULES.feasibility;
  let feas: PilotSummary["feasibility"]["verdict"] = "INSUFFICIENT_DATA";
  if (share !== null && attempted.length >= F.minAttempts) feas = share >= F.passAtLeast ? "PASS" : share < F.failBelow ? "FAIL" : "INCONCLUSIVE_REDESIGN";

  // Absolute net of FOLLOW at the primary delay.
  const followSettled = primaryFollow.filter((e) => e.status === "SETTLED");
  const absWin = clusterBootstrapMean(followSettled.map((e) => ({ cluster: e.window, value: e.returnOnCost! })), { seed: RULES.randomSeed });
  const absDay = clusterBootstrapMean(followSettled.map((e) => ({ cluster: e.day, value: e.returnOnCost! })), { seed: RULES.randomSeed + 2 });

  // Paired superiority vs the primary baseline: only observations settled under BOTH.
  const baseMap = new Map(entries.filter((e) => e.strategy === baseline && e.delaySeconds === d && e.status === "SETTLED").map((e) => [e.obsId, e]));
  const paired = followSettled.filter((e) => baseMap.has(e.obsId)).map((e) => ({ window: e.window, day: e.day, diff: e.returnOnCost! - baseMap.get(e.obsId)!.returnOnCost! }));
  const supWin = clusterBootstrapMean(paired.map((p) => ({ cluster: p.window, value: p.diff })), { seed: RULES.randomSeed });
  const supDay = clusterBootstrapMean(paired.map((p) => ({ cluster: p.day, value: p.diff })), { seed: RULES.randomSeed + 1 });

  // Power from per-window means (descriptive in the pilot; frozen for the test).
  const sdFollow = sd(perWindowMeans(followSettled.map((e) => ({ window: e.window, v: e.returnOnCost! }))));
  const sdDiff = sd(perWindowMeans(paired.map((p) => ({ window: p.window, v: p.diff }))));
  const suggested = [requiredWindows(sdFollow), requiredWindows(sdDiff)].filter((x): x is number => x !== null);
  const suggestedRequired = suggested.length ? Math.max(...suggested) : null;

  const windows = new Set(observations.map((o) => o.windowSlug)).size;
  const days = new Set(observations.map((o) => new Date(o.windowStartSec * 1000).toISOString().slice(0, 10))).size;
  const gate = options.frozenRequiredWindows === null
    ? { ok: false, reason: "sample size requirement was not frozen before the test" }
    : supWin.clusters < options.frozenRequiredWindows || absWin.clusters < options.frozenRequiredWindows
    ? { ok: false, reason: `need ${options.frozenRequiredWindows} independent windows (have ${Math.min(supWin.clusters, absWin.clusters)})` }
    : days < RULES.minTestDays
    ? { ok: false, reason: `need >= ${RULES.minTestDays} days (have ${days})` }
    : { ok: true, reason: "" };
  const absVerdict = verdictFor(absWin, absDay, gate, options.finalAnalysis);
  const supVerdict = verdictFor(supWin, supDay, gate, options.finalAnalysis);

  let edge: PilotSummary["edge"];
  if (!options.finalAnalysis) edge = { verdict: "INTERIM_NO_VERDICT", reason: "only the pre-registered final analysis may carry a verdict" };
  else if (!gate.ok) edge = { verdict: "INSUFFICIENT_DATA", reason: gate.reason };
  else if (absVerdict === "SUPPORTED" && supVerdict === "SUPPORTED") edge = { verdict: "SUPPORTED", reason: "absolute net > 0 AND better than baseline (window- and day-clustered lower bounds > 0)" };
  else if (absVerdict === "NOT_SUPPORTED") edge = { verdict: "NOT_SUPPORTED", reason: "follower's own net return is negative" };
  else if (supVerdict === "NOT_SUPPORTED") edge = { verdict: "NOT_SUPPORTED", reason: "follower is worse than the baseline" };
  else edge = { verdict: "INSUFFICIENT_DATA", reason: `absolute: ${absVerdict}, vs baseline: ${supVerdict}` };

  const measured = observations.map((o) => (o.firstSeenMs - o.tradeTsMs) / 1000);
  const [p50, p90] = quantiles(measured, [0.5, 0.9]);
  return {
    rulesVersion,
    question: "CONDITIONAL_FOLLOW_REALIZABILITY_AND_COMPARISON",
    h2WalletIndependentTiming: { status: "NOT_TESTED", reason: "deferred: needs a wallet-independent pre-registered decision schedule (see plan section 1)" },
    observations: observations.length,
    windows,
    days,
    latency: {
      measured: measured.length,
      secondsP50: p50,
      secondsP90: p90,
      secondsMax: measured.length ? Math.max(...measured) : null,
      seenAfterWindowClosed: observations.filter((o) => o.firstSeenMs >= o.windowEndSec * 1000).length,
      lateForPrimaryDelay: primaryFollow.filter((e) => e.status === "LATE_WINDOW_CLOSED").length,
      note: "public timestamp has 1 s resolution; latency includes the poll interval",
    },
    byStrategyDelay: by,
    feasibility: { attempts: attempted.length, fillableShare: share, verdict: feas },
    absoluteNet: { verdict: absVerdict, windowClustered: absWin, dayClustered: absDay },
    baselineSuperiority: { baseline, pairedObservations: paired.length, verdict: supVerdict, windowClustered: supWin, dayClustered: supDay },
    power: {
      minEffect: RULES.minEffectReturnOnCost,
      perWindowSdFollow: sdFollow,
      perWindowSdDiff: sdDiff,
      suggestedRequiredWindows: suggestedRequired,
      frozenRequiredWindows: options.frozenRequiredWindows,
      note: "two-sided alpha 0.05, power 0.8 on per-window means; a pilot suggestion must be frozen before the test, never chosen after seeing test data",
    },
    edge,
    tradeSignal: false,
  };
}
