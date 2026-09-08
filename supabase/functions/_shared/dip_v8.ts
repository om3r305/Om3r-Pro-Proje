// DIP-only deterministic contracts. No exchange execution or database dependency.
export const SYMBOL = "ETHUSDT";
export const ENGINE_VERSION = "brian-dip-chart-reader-v8";
export const POLICY_VERSION = "dip-v8-integrity-20260908.1";
export const METRIC_VERSION = "target-before-invalidation-v8.1";
export const RESOLVER_VERSION = "dip-v8-path-20260908.1";
export const MAX_HOLD_MS = 90 * 60_000;
export const MIN_CAL_SAMPLES = 40;
export const MAX_POSITION_FRACTION_CALIBRATING = 0.08;
export const MAX_HEAT = 0.20;
export type J = Record<string, unknown>;
export type Bar = {
  t: number;
  ct: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
};
export type Pivot = {
  i: number;
  t: number;
  p: number;
  kind: "H" | "L";
  label?: string;
};
export type Struct = {
  tf: string;
  lastClose: number;
  atr: number;
  pivots: Pivot[];
  lastHigh: Pivot | null;
  lastLow: Pivot | null;
  trend: "UP" | "DOWN" | "RANGE";
  bos: "UP" | "DOWN" | null;
  choch: "UP" | "DOWN" | null;
  sweep: "BULL" | "BEAR" | null;
  failedBreak: "BULL" | "BEAR" | null;
  equalHigh: number | null;
  equalLow: number | null;
  fingerprint: string;
};
export type Book = {
  bid: number;
  ask: number;
  mid: number;
  spreadBps: number;
  pressure: number;
  receivedAt: number;
};
export type Flow = {
  ofi: number;
  ratio: number;
  prints: number;
  firstAt: number;
  lastAt: number;
};
export type Rules = {
  minNotional: number;
  minQty: number;
  maxQty: number;
  stepSize: number;
};
export type Cal = {
  samples: number;
  hits: number;
  p: number | null;
  lower: number;
  upper: number;
  ambiguous: number;
  unavailable?: boolean;
};
export type Position = {
  side: "LONG";
  position_id: string;
  thesis_id: string;
  episode_id: string;
  setup: string;
  regime: string;
  entry: number;
  qty: number;
  notional: number;
  target: number;
  stop: number;
  opened_at: string;
  due_at: string;
  fees_open: number;
  fee_bps: number;
  slippage_bps: number;
  spread_bps: number;
  venue: "SPOT";
  policy_version: string;
  checked_until: number;
  market_price: number;
  actual_fraction: number;
};
export type Runtime = {
  start: number;
  cash: number;
  realized: number;
  trades: number;
  wins: number;
  losses: number;
  pos: Position | null;
  lastLock: {
    episode_id: string;
    fingerprint: string;
    last5m_t: number;
    reason: string;
  } | null;
  latestThesis: J | null;
  lastOccurrence: string | null;
  lastSnapshotHour: string | null;
  marketCursor: number;
  lastClosedAt: number | null;
};
export const TF_MS: Record<string, number> = {
  "1m": 60_000,
  "5m": 300_000,
  "15m": 900_000,
  "1h": 3_600_000,
  "4h": 14_400_000,
};
export function n(value: unknown, fallback = 0): number {
  const x = Number(value);
  return Number.isFinite(x) ? x : fallback;
}
export function clip(x: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, x));
}
export function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}
export async function hash(s: string): Promise<string> {
  const b = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)),
  );
  return [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
}
export function same(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
}
export function initialRuntime(start: number): Runtime {
  if (!Number.isFinite(start) || start <= 0) {
    throw Error("INVALID_STARTING_EQUITY");
  }
  return {
    start,
    cash: start,
    realized: 0,
    trades: 0,
    wins: 0,
    losses: 0,
    pos: null,
    lastLock: null,
    latestThesis: null,
    lastOccurrence: null,
    lastSnapshotHour: null,
    marketCursor: 0,
    lastClosedAt: null,
  };
}
export function validateSession(config: J): void {
  if (
    config.engine_version !== ENGINE_VERSION ||
    config.policy_version !== POLICY_VERSION ||
    JSON.stringify(config.symbols) !== JSON.stringify([SYMBOL])
  ) throw Error("WAIT_V8_CLEAN_RESTART");
  if (
    config.shadow_only !== true || config.live_execution !== false ||
    config.browser_execution !== false || config.server_authoritative !== true
  ) throw Error("INVALID_SHADOW_CONTRACT");
  if (
    config.allow_shadow_short !== false ||
    n(config.max_shadow_leverage, 1) !== 1
  ) throw Error("SPOT_LONG_ONLY");
  if (!["OBSERVE", "SHADOW_PAPER"].includes(String(config.execution_mode))) {
    throw Error("INVALID_EXECUTION_MODE");
  }
}
export function parseBars(
  raw: unknown,
  tf: string,
  nowMs: number,
  minBars = 50,
): Bar[] {
  if (!Array.isArray(raw) || !TF_MS[tf]) throw Error("NO_DATA:" + tf);
  const duration = TF_MS[tf];
  const rows: Bar[] = raw.map((x) => {
    if (!Array.isArray(x) || x.length < 7) throw Error("INVALID_CANDLE:" + tf);
    return {
      t: Number(x[0]),
      ct: Number(x[6]),
      o: Number(x[1]),
      h: Number(x[2]),
      l: Number(x[3]),
      c: Number(x[4]),
      v: Number(x[5]),
    };
  });
  for (let i = 0; i < rows.length; i++) {
    const b = rows[i];
    if (
      ![b.t, b.ct, b.o, b.h, b.l, b.c, b.v].every(Number.isFinite) ||
      b.t % duration || b.ct !== b.t + duration - 1 || b.l <= 0 || b.v < 0 ||
      b.h < Math.max(b.o, b.c) || b.l > Math.min(b.o, b.c) || b.t > nowMs
    ) throw Error("INVALID_CANDLE:" + tf);
    if (i && b.t !== rows[i - 1].t + duration) {
      throw Error("CANDLE_GAP_OR_ORDER:" + tf);
    }
  }
  const closed = rows.filter((b) => b.ct < nowMs);
  if (closed.length < minBars) throw Error("WARMUP_INSUFFICIENT:" + tf);
  if (nowMs - (closed.at(-1)!.ct + 1) > duration + 15_000) {
    throw Error("STALE_DATA:" + tf);
  }
  return closed;
}
export function readBook(raw: unknown, receivedAt: number): Book {
  const r = raw as { bids?: unknown[][]; asks?: unknown[][] };
  if (
    !Array.isArray(r?.bids) || !Array.isArray(r?.asks) || !r.bids.length ||
    !r.asks.length
  ) throw Error("NO_BOOK");
  const bids = r.bids.map((x) => [Number(x[0]), Number(x[1])]),
    asks = r.asks.map((x) => [Number(x[0]), Number(x[1])]);
  for (const side of [bids, asks]) {
    for (const [p, q] of side) {
      if (!(p > 0 && q > 0) || !Number.isFinite(p * q)) {
        throw Error("INVALID_BOOK");
      }
    }
  }
  if (
    bids.some((x, i) => i > 0 && x[0] >= bids[i - 1][0]) ||
    asks.some((x, i) => i > 0 && x[0] <= asks[i - 1][0])
  ) throw Error("BOOK_ORDER");
  const bid = bids[0][0], ask = asks[0][0], mid = (bid + ask) / 2;
  if (bid >= ask) throw Error("CROSSED_BOOK");
  const bn = bids.slice(0, 5).reduce((s, x) => s + x[0] * x[1], 0),
    an = asks.slice(0, 5).reduce((s, x) => s + x[0] * x[1], 0);
  return {
    bid,
    ask,
    mid,
    spreadBps: (ask - bid) / mid * 10000,
    pressure: bn / an,
    receivedAt,
  };
}
export function readFlow(raw: unknown, nowMs: number): Flow {
  if (!Array.isArray(raw) || raw.length < 10) throw Error("NO_TRADES");
  let buy = 0, sell = 0, priorId = -1, priorAt = 0;
  for (const x of raw) {
    const r = x as J,
      at = Number(r.T),
      id = Number(r.a),
      value = Number(r.p) * Number(r.q);
    if (
      !(value > 0) || !Number.isFinite(value) || !Number.isSafeInteger(id) ||
      id <= priorId || at < priorAt || at > nowMs + 2000 ||
      typeof r.m !== "boolean"
    ) throw Error("INVALID_TRADE_ORDER");
    r.m ? sell += value : buy += value;
    priorId = id;
    priorAt = at;
  }
  if (nowMs - priorAt > 15_000) throw Error("STALE_TRADES");
  return {
    ofi: (buy - sell) / (buy + sell),
    ratio: sell ? buy / sell : 9,
    prints: raw.length,
    firstAt: Number(raw[0].T),
    lastAt: priorAt,
  };
}
export function calibrate(
  rows: { hit: boolean | null }[],
  unavailable = false,
): Cal {
  const clean = rows.filter((x) => x.hit !== null),
    samples = clean.length,
    hits = clean.filter((x) => x.hit).length;
  const p = samples ? hits / samples : 0,
    z = 1.96,
    d = 1 + z * z / Math.max(samples, 1);
  const centre = (p + z * z / (2 * Math.max(samples, 1))) / d;
  const width = z *
    Math.sqrt(
      (p * (1 - p) + z * z / (4 * Math.max(samples, 1))) / Math.max(samples, 1),
    ) / d;
  return {
    samples,
    hits,
    p: samples >= MIN_CAL_SAMPLES ? p : null,
    lower: samples ? Math.max(0, centre - width) : 0,
    upper: samples ? Math.min(1, centre + width) : 1,
    ambiguous: rows.length - clean.length,
    unavailable,
  };
}
export function sizeLong(
  cash: number,
  entry: number,
  stop: number,
  cal: Cal,
  feeBps: number,
  slippageBps: number,
  rules: Rules,
) {
  if (!(cash > 0 && entry > stop && stop > 0) || cal.unavailable) return null;
  const cap = cal.p === null ? MAX_POSITION_FRACTION_CALIBRATING : MAX_HEAT;
  const lossPerUnit = entry - stop +
    entry * (feeBps * 2 + slippageBps * 2) / 10000;
  const budgetQty = Math.min(
    cash * cap / entry,
    cash * 0.005 / lossPerUnit,
    cash / (entry * (1 + feeBps / 10000)),
    rules.maxQty,
  );
  const qty = Math.floor((budgetQty + Number.EPSILON) / rules.stepSize) *
    rules.stepSize;
  const notional = qty * entry;
  if (
    !Number.isFinite(qty) || qty < rules.minQty ||
    notional < rules.minNotional || notional > cash * cap + 1e-8
  ) return null;
  return {
    qty,
    notional,
    actual_fraction: notional / cash,
    fees_open: notional * feeBps / 10000,
  };
}
export type Segment = {
  start: number;
  end: number;
  o: number;
  h: number;
  l: number;
  c: number;
  kind: "BAR" | "TRADES";
  points?: { t: number; id: number; p: number }[];
};
export type Resolution = {
  reason:
    | "TARGET_FIRST"
    | "INVALIDATION_FIRST"
    | "AMBIGUOUS"
    | "EXPIRED_NO_BARRIER"
    | "PENDING"
    | "INDETERMINATE";
  hit: boolean | null;
  price: number;
  eventAt: number | null;
  eventRangeStart: number | null;
  checkedUntil: number;
  detail?: string;
};
// Windows are [start, end). Sealed evidence only. Boundary segments are timestamped trades.
export function evaluatePath(
  input: {
    direction: "UP" | "DOWN";
    target: number;
    stop: number;
    start: number;
    due: number;
    now: number;
    end: number;
    entry: number;
    segments: Segment[];
  },
): Resolution {
  const { direction, target, stop, start, due, now, end, entry, segments } =
    input;
  const result = (
    reason: Resolution["reason"],
    hit: boolean | null,
    price: number,
    eventAt: number | null,
    checkedUntil: number,
    eventRangeStart: number | null = eventAt,
  ): Resolution => ({
    reason,
    hit,
    price,
    eventAt,
    eventRangeStart,
    checkedUntil,
  });
  if (
    !(target > 0 && stop > 0 && start < due && end <= now && end <= due &&
      end >= start)
  ) return result("INDETERMINATE", null, entry, null, start);
  let cursor = start, last = entry;
  const hit = (p: number) =>
    direction === "UP"
      ? { win: p >= target, loss: p <= stop }
      : { win: p <= target, loss: p >= stop };
  for (const s of segments) {
    if (
      s.start !== cursor || s.end <= s.start || s.end > end ||
      ![s.o, s.h, s.l, s.c].every(Number.isFinite) || s.l <= 0 || s.h < s.l
    ) return result("INDETERMINATE", null, last, null, cursor);
    if (s.kind === "TRADES") {
      let previousAt = s.start, previousId = -1;
      for (const p of s.points || []) {
        if (
          p.t < previousAt || p.t < s.start || p.t >= s.end ||
          p.id <= previousId || !(p.p > 0)
        ) return result("INDETERMINATE", null, last, null, cursor);
        const h = hit(p.p);
        previousAt = p.t;
        previousId = p.id;
        last = p.p;
        if (h.loss) return result("INVALIDATION_FIRST", false, p.p, p.t, s.end);
        if (h.win) return result("TARGET_FIRST", true, target, p.t, s.end);
      }
    } else {
      const open = hit(s.o);
      if (open.loss) {
        return result("INVALIDATION_FIRST", false, s.o, s.start, s.end);
      }
      if (open.win) return result("TARGET_FIRST", true, target, s.start, s.end);
      const win = direction === "UP" ? s.h >= target : s.l <= target,
        loss = direction === "UP" ? s.l <= stop : s.h >= stop;
      if (win && loss) {
        return result("AMBIGUOUS", null, stop, s.end - 1, s.end, s.start);
      }
      if (loss) {
        return result(
          "INVALIDATION_FIRST",
          false,
          stop,
          s.end - 1,
          s.end,
          s.start,
        );
      }
      if (win) {
        return result("TARGET_FIRST", true, target, s.end - 1, s.end, s.start);
      }
      last = s.c;
    }
    cursor = s.end;
  }
  if (cursor !== end) return result("INDETERMINATE", null, last, null, cursor);
  return end === due
    ? result("EXPIRED_NO_BARRIER", false, last, due, end)
    : result("PENDING", null, last, null, cursor);
}
export function closeLong(
  rt: Runtime,
  resolution: Resolution,
  fingerprint: string,
  last5m: number,
  recordedAt: number,
): J | null {
  const p = rt.pos;
  if (!p || ["PENDING", "INDETERMINATE"].includes(resolution.reason)) {
    return null;
  }
  // Historical barrier fill model: frozen half-spread + slippage, each charged once.
  const exit = resolution.price *
    (1 - (p.spread_bps / 2 + p.slippage_bps) / 10000);
  const feeClose = exit * p.qty * p.fee_bps / 10000;
  const net = (exit - p.entry) * p.qty - p.fees_open - feeClose;
  rt.cash += exit * p.qty - feeClose;
  rt.realized += net;
  rt.trades++;
  if (net > 0) rt.wins++;
  else if (net < 0) rt.losses++;
  rt.lastLock = {
    episode_id: p.episode_id,
    fingerprint,
    last5m_t: last5m,
    reason: resolution.reason,
  };
  rt.lastClosedAt = recordedAt;
  rt.pos = null;
  return {
    event_kind: "SELL",
    position_id: p.position_id,
    occurrence_id: p.thesis_id,
    episode_id: p.episode_id,
    price: exit,
    entry_price: p.entry,
    exit_price: exit,
    quantity: p.qty,
    notional: p.notional,
    fees: p.fees_open + feeClose,
    realized_pnl: net,
    cash_after: rt.cash,
    equity_after: rt.cash,
    metadata: {
      server_v8: true,
      exit_reason: resolution.reason === "AMBIGUOUS"
        ? "AMBIGUOUS_CONSERVATIVE_STOP"
        : resolution.reason,
      thesis_id: p.thesis_id,
      setup: p.setup,
      venue: p.venue,
      policy_version: p.policy_version,
      resolution,
      execution_model: "barrier_taker_frozen_spread_v1",
      fee_bps: p.fee_bps,
      slippage_bps: p.slippage_bps,
    },
  };
}
