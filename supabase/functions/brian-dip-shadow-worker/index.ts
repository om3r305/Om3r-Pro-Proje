import { createClient } from "npm:@supabase/supabase-js@2";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { withCollectorLease } from "../_shared/collector_lease.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });

const COLLECTOR_ID = "brian-dip-shadow-worker-v7";
const WORKER_VERSION = "brian-dip-server-v7";
const EVIDENCE = "AGGRESSIVE_DIP_SHADOW";
const LEASE_SECONDS = 55;
const MIN_INTERVAL_SECONDS = 45;
const MAX_OPEN = 2;
const MAX_PORTFOLIO_HEAT = 0.80;
const MAX_POSITION_FRACTION = 0.72;
const MIN_POSITION_FRACTION = 0.035;
const MAX_ENTRY_SPREAD_BPS = 8;
const COST_EDGE_MULT = 2.5;
const MIN_RR = 1.35;
const LONG_HOLD_MS = 18 * 60_000;
const SHORT_HOLD_MS = 22 * 60_000;
const PAIR_LOCK_MS = 30 * 60_000;
const GLOBAL_LOCK_MS = 10 * 60_000;
const DD_LOCK_MS = 30 * 60_000;
const BRAIN_MIN_CONF = 0.52;
const BRAIN_MIN_AGREEMENT = 0.54;
const BRAIN_MIN_EDGE = 0.18;
const BRAIN_MIN_QUALITY = 0.58;
const BRAIN_MISS_HORIZON_MS = 5 * 60_000;
const BROWSER_HANDOFF_GRACE_MS = 90_000;
const FUTURES_FEE_BPS = 5;
const SPOT_HOSTS = ["https://api.binance.com", "https://api1.binance.com", "https://api3.binance.com"];
const FUTURES_HOST = "https://fapi.binance.com";

const BASE_WEIGHTS: Record<string, number> = {
  structure: 0.32, trend: 0.24, momentum: 0.18, volume: 0.12, mean_reversion: 0.14, microstructure: 0.20,
};

type Json = Record<string, unknown>;
type Bar = { t: number; o: number; h: number; l: number; c: number; v: number };
type Feature = {
  last: number; open: number; high: number; low: number; atr: number; atrPct: number;
  ema9: number; ema21: number; ema50: number; rsi: number; z: number; volRel: number;
  swingLow: number | null; swingHigh: number | null; trend: number;
};
type Book = { bid: number; ask: number; mid: number; spreadBps: number; pressure: number; bidN: number; askN: number; bids: [number, number][]; asks: [number, number][] };
type Flow = { buy: number; sell: number; ofi: number; ratio: number; prints: number; retPct: number };
type Market = { symbol: string; bars1: Bar[]; bars5: Bar[]; bars15: Bar[]; bars1h: Bar[]; f1: Feature; f5: Feature; f15: Feature; f1h: Feature; book: Book; flow: Flow; price: number };
type Vote = { name: string; bias: number; confidence: number; weight: number };
type Decision = {
  action: "BUY" | "SELL" | "WAIT"; edge: number; confidence: number; agreement: number; uncertainty: number;
  setup: string; regime: string; grossEdgeBps: number; costBps: number; netEdgeBps: number; quality: number;
  desiredFraction: number; riskBudgetPct: number; vetoReasons: string[]; votes: Vote[]; hardDrift: boolean;
};

type Session = {
  session_id: string; started_at: string; starting_equity: number; trade_notional: number; config: Json;
};

type Runtime = {
  start: number; cash: number; realized: number; trades: number; wins: number; losses: number; config: Json;
  symbols: Record<string, Json>; pairGuard: Record<string, number>; closedOutcomes: Json[]; sessionLossLockUntil: number;
  brain: Json; serverWasAuthoritative: boolean;
};

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}
function n(v: unknown, d = 0) { const x = Number(v); return Number.isFinite(x) ? x : d; }
function clip(v: unknown, lo: number, hi: number) { return Math.max(lo, Math.min(hi, n(v))); }
function pct(a: number, b: number) { return b ? (a / b - 1) * 100 : 0; }
function sum(xs: number[]) { return xs.reduce((a, b) => a + n(b), 0); }
function mean(xs: number[]) { return xs.length ? sum(xs) / xs.length : 0; }
function sign(x: number) { return x > 0 ? 1 : x < 0 ? -1 : 0; }
function errorText(e: unknown) { return e instanceof Error ? `${e.name}: ${e.message}` : String(e); }
function nowIso() { return new Date().toISOString(); }
async function sha(s: string) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)));
  return [...d].map((x) => x.toString(16).padStart(2, "0")).join("");
}
function ema(xs: number[], period: number) {
  if (!xs.length) return 0; const k = 2 / (period + 1); let x = xs[0];
  for (let i = 1; i < xs.length; i++) x = xs[i] * k + x * (1 - k); return x;
}
function rsi(xs: number[], period = 14) {
  if (xs.length < period + 1) return 50; let g = 0, l = 0;
  for (let i = xs.length - period; i < xs.length; i++) { const d = xs[i] - xs[i - 1]; if (d >= 0) g += d; else l -= d; }
  if (!l) return 100; const rs = (g / period) / (l / period); return 100 - 100 / (1 + rs);
}
function atr(rows: Bar[], period = 14) {
  if (rows.length < 2) return 0; const xs: number[] = [];
  for (let i = Math.max(1, rows.length - period); i < rows.length; i++) {
    const r = rows[i], pc = rows[i - 1].c; xs.push(Math.max(r.h - r.l, Math.abs(r.h - pc), Math.abs(r.l - pc)));
  }
  return mean(xs);
}
function feature(rows: Bar[]): Feature | null {
  if (rows.length < 30) return null;
  const closes = rows.map((x) => x.c), last = rows[rows.length - 1], a = atr(rows, 14), atrPct = last.c ? a / last.c * 100 : 0;
  const ema9 = ema(closes.slice(-40), 9), ema21 = ema(closes.slice(-70), 21), ema50 = ema(closes.slice(-100), 50);
  const m = ema(closes.slice(-40), 20), z = a ? (last.c - m) / a : 0, rr = rsi(closes, 14);
  const priorVol = rows.slice(-21, -1).map((x) => x.v), volRel = mean(priorVol) > 0 ? last.v / mean(priorVol) : 1;
  const swingRows = rows.slice(-18), swingLow = swingRows.length ? Math.min(...swingRows.map((x) => x.l)) : null;
  const swingHigh = swingRows.length ? Math.max(...swingRows.map((x) => x.h)) : null;
  const slope21 = ema21 ? (last.c / ema21 - 1) * 100 : 0; let trend = 0;
  trend += last.c > ema9 ? .22 : -.22; trend += ema9 > ema21 ? .28 : -.28; trend += ema21 > ema50 ? .30 : -.30; trend += clip(slope21 * 1.5, -.20, .20);
  return { last: last.c, open: last.o, high: last.h, low: last.l, atr: a, atrPct, ema9, ema21, ema50, rsi: rr, z, volRel, swingLow, swingHigh, trend: clip(trend, -1, 1) };
}

async function fetchJson(path: string, futures = false): Promise<unknown> {
  const hosts = futures ? [FUTURES_HOST] : SPOT_HOSTS;
  let last: unknown = null;
  for (const host of hosts) {
    try {
      const r = await fetch(`${host}${path}`, { headers: { accept: "application/json", "user-agent": "Brian-Dip-V7-Shadow/1.0" }, signal: AbortSignal.timeout(8_000) });
      if (!r.ok) throw new Error(`HTTP_${r.status}`); return await r.json();
    } catch (e) { last = e; }
  }
  throw new Error(`BINANCE_FETCH_FAILED:${path}:${errorText(last)}`);
}
function parseKlines(raw: unknown): Bar[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((x) => Array.isArray(x) ? ({ t: n(x[0]), o: n(x[1]), h: n(x[2]), l: n(x[3]), c: n(x[4]), v: n(x[5]) }) : null)
    .filter((x): x is Bar => Boolean(x && x.t > 0 && x.c > 0));
}
function parseBook(raw: unknown): Book {
  const r = raw && typeof raw === "object" ? raw as Json : {};
  const bids = Array.isArray(r.bids) ? r.bids.slice(0, 20).map((z) => Array.isArray(z) ? [n(z[0]), n(z[1])] as [number, number] : [0, 0] as [number, number]).filter((z) => z[0] > 0 && z[1] >= 0) : [];
  const asks = Array.isArray(r.asks) ? r.asks.slice(0, 20).map((z) => Array.isArray(z) ? [n(z[0]), n(z[1])] as [number, number] : [0, 0] as [number, number]).filter((z) => z[0] > 0 && z[1] >= 0) : [];
  const bid = bids[0]?.[0] ?? 0, ask = asks[0]?.[0] ?? 0, mid = bid > 0 && ask >= bid ? (bid + ask) / 2 : 0;
  const spreadBps = mid > 0 ? (ask - bid) / mid * 10000 : 999;
  const bidN = sum(bids.slice(0, 5).map(([p, q]) => p * q)), askN = sum(asks.slice(0, 5).map(([p, q]) => p * q));
  return { bid, ask, mid, spreadBps, pressure: askN > 0 ? bidN / askN : 1, bidN, askN, bids, asks };
}
function parseFlow(raw: unknown): Flow {
  if (!Array.isArray(raw) || !raw.length) return { buy: 0, sell: 0, ofi: 0, ratio: 1, prints: 0, retPct: 0 };
  let buy = 0, sell = 0; const prices: number[] = [];
  for (const x of raw) {
    if (!x || typeof x !== "object") continue; const r = x as Json, p = n(r.p), q = n(r.q); if (!(p > 0) || !(q > 0)) continue;
    const notional = p * q; if (r.m === true) sell += notional; else buy += notional; prices.push(p);
  }
  const total = buy + sell, first = prices[0] ?? 0, last = prices[prices.length - 1] ?? 0;
  return { buy, sell, ofi: total ? (buy - sell) / total : 0, ratio: sell > 0 ? buy / sell : buy > 0 ? 9 : 1, prints: prices.length, retPct: first > 0 ? pct(last, first) : 0 };
}
async function fetchMarket(symbol: string): Promise<Market> {
  const q = encodeURIComponent(symbol);
  const [a1, a5, a15, a1h, dep, trades] = await Promise.all([
    fetchJson(`/api/v3/klines?symbol=${q}&interval=1m&limit=100`),
    fetchJson(`/api/v3/klines?symbol=${q}&interval=5m&limit=100`),
    fetchJson(`/api/v3/klines?symbol=${q}&interval=15m&limit=100`),
    fetchJson(`/api/v3/klines?symbol=${q}&interval=1h&limit=100`),
    fetchJson(`/api/v3/depth?symbol=${q}&limit=20`),
    fetchJson(`/api/v3/aggTrades?symbol=${q}&limit=100`),
  ]);
  const bars1 = parseKlines(a1), bars5 = parseKlines(a5), bars15 = parseKlines(a15), bars1h = parseKlines(a1h);
  const f1 = feature(bars1), f5 = feature(bars5), f15 = feature(bars15), f1h = feature(bars1h), book = parseBook(dep), flow = parseFlow(trades);
  if (!f1 || !f5 || !f15 || !f1h || !(book.mid > 0)) throw new Error(`MARKET_CONTEXT_INCOMPLETE:${symbol}`);
  return { symbol, bars1, bars5, bars15, bars1h, f1, f5, f15, f1h, book, flow, price: book.mid };
}
async function loadMarkets(symbols: string[]): Promise<{ markets: Map<string, Market>; degraded: string[] }> {
  const markets = new Map<string, Market>(), degraded: string[] = [];
  for (let i = 0; i < symbols.length; i += 4) {
    const batch = symbols.slice(i, i + 4);
    const rows = await Promise.all(batch.map(async (s) => { try { return { s, m: await fetchMarket(s), e: null }; } catch (e) { return { s, m: null, e }; } }));
    for (const row of rows) { if (row.m) markets.set(row.s, row.m); else degraded.push(`${row.s}:${errorText(row.e)}`); }
  }
  return { markets, degraded };
}
async function futuresSymbols(): Promise<Set<string>> {
  const raw = await fetchJson("/fapi/v1/exchangeInfo", true); const out = new Set<string>();
  const rows = raw && typeof raw === "object" && Array.isArray((raw as Json).symbols) ? (raw as Json).symbols as unknown[] : [];
  for (const x of rows) if (x && typeof x === "object") { const r = x as Json, s = String(r.symbol ?? ""); if (r.status === "TRADING" && /^[A-Z0-9]+USDT$/.test(s)) out.add(s); }
  return out;
}
async function perpBook(symbol: string): Promise<Book> { return parseBook(await fetchJson(`/fapi/v1/depth?symbol=${encodeURIComponent(symbol)}&limit=20`, true)); }

function impactBps(book: Book, notional: number, side: "BUY" | "SELL") {
  const levels = side === "BUY" ? book.asks : book.bids; if (!(book.mid > 0) || !levels.length) return 8;
  let rem = Math.max(1, notional), spent = 0, qty = 0;
  for (const [p, q] of levels) { const cap = p * q, take = Math.min(rem, cap); spent += take; qty += take / p; rem -= take; if (rem <= 0) break; }
  if (rem > 0) return Math.min(50, 8 + rem / Math.max(1, notional) * 25);
  const vwap = qty ? spent / qty : book.mid;
  return side === "BUY" ? Math.max(0, (vwap - book.mid) / book.mid * 10000) : Math.max(0, (book.mid - vwap) / book.mid * 10000);
}
function fillPrice(book: Book, side: "LONG" | "SHORT", notional: number, isExit: boolean, configuredSlipBps: number) {
  const buy = (!isExit && side === "LONG") || (isExit && side === "SHORT");
  const impact = impactBps(book, notional, buy ? "BUY" : "SELL"), slipBps = Math.max(configuredSlipBps, impact);
  const base = buy ? (book.ask || book.mid) : (book.bid || book.mid);
  const px = buy ? base * (1 + slipBps / 10000) : base * (1 - slipBps / 10000);
  return { px, slipBps, spreadBps: book.spreadBps, impactBps: impact };
}

function betaReliability(row: unknown) {
  const r = row && typeof row === "object" ? row as Json : {}, count = n(r.n), wins = n(r.wins), wr = (wins + 3) / (count + 6);
  return clip(.72 + (wr - .5) * 1.25, .65, 1.35);
}
function setupReliability(brain: Json, setup: string) {
  const setups = brain.setups && typeof brain.setups === "object" ? brain.setups as Json : {}, row = setups[setup];
  return betaReliability(row);
}
function regimeWeight(name: string, regime: string) {
  const m: Record<string, number> = { structure: 1, trend: 1, momentum: 1, volume: 1, mean_reversion: 1, microstructure: 1 };
  if (regime === "TREND_UP" || regime === "TREND_DOWN") { m.trend = 1.35; m.momentum = 1.20; m.structure = 1.15; m.mean_reversion = .72; }
  else if (regime === "RANGE") { m.mean_reversion = 1.38; m.structure = 1.18; m.trend = .82; }
  else if (regime === "VOLATILE_RANGE") { m.microstructure = 1.30; m.structure = 1.15; m.mean_reversion = 1.12; m.momentum = .90; }
  else if (regime === "STRESS_DOWN") { m.microstructure = 1.32; m.structure = 1.25; m.trend = 1.18; m.mean_reversion = .62; }
  return m[name] ?? 1;
}
function brainFresh(): Json {
  return {
    schemaVersion: "brian.dip-intelligence-state.v2", intelligenceVersion: WORKER_VERSION, cashboxId: "DIP_SHADOW_CASHBOX_V7", mainRuntimeMutation: false,
    createdAt: Date.now(), lastLearningAt: Date.now(), experts: {}, setups: {}, vetoes: {}, pending: [], audits: [], outcomes: [],
    ab: { champion: "BRIAN_SERVER_V7", challenger: "COUNTERFACTUAL", policy: "BRAIN_AUTHORITATIVE_RISK_GATED", attempted: 0, accepted: 0, brainVetoed: 0, brainOnlyCandidates: 0, savedLosses: 0, missedWins: 0, correctAbstentions: 0, promotedInsideDip: true },
  };
}
function ensureBrain(x: unknown, preserve = true): Json {
  if (!preserve || !x || typeof x !== "object") return brainFresh();
  const old = x as Json, fresh = brainFresh();
  return { ...fresh, ...old, intelligenceVersion: WORKER_VERSION, cashboxId: "DIP_SHADOW_CASHBOX_V7", mainRuntimeMutation: false, experts: old.experts && typeof old.experts === "object" ? old.experts : {}, setups: old.setups && typeof old.setups === "object" ? old.setups : {}, vetoes: old.vetoes && typeof old.vetoes === "object" ? old.vetoes : {}, pending: Array.isArray(old.pending) ? old.pending : [], audits: Array.isArray(old.audits) ? old.audits : [], outcomes: Array.isArray(old.outcomes) ? old.outcomes : [], ab: { ...(fresh.ab as Json), ...(old.ab && typeof old.ab === "object" ? old.ab as Json : {}) } };
}
function brainRow(brain: Json, bucket: "experts" | "setups" | "vetoes", key: string): Json {
  const b = brain[bucket] && typeof brain[bucket] === "object" ? brain[bucket] as Json : {}; brain[bucket] = b;
  if (!b[key] || typeof b[key] !== "object") b[key] = bucket === "vetoes" ? { n: 0, savedLoss: 0, missedWin: 0, neutral: 0 } : { n: 0, wins: 0, pnl: 0, mfe: 0, mae: 0 };
  return b[key] as Json;
}
function regime(m: Market, btc: Market) {
  const h = .58 * m.f15.trend + .42 * m.f1h.trend, btcRisk = .55 * btc.f15.trend + .45 * btc.f1h.trend;
  if (btcRisk < -.55 && h < -.12) return "STRESS_DOWN"; if (h > .34) return "TREND_UP"; if (h < -.34) return "TREND_DOWN"; if (m.f5.atrPct > 1.6) return "VOLATILE_RANGE"; return "RANGE";
}
function sweep(m: Market) {
  const rows = m.bars1; if (rows.length < 8) return { bull: false, bear: false, lowerWickRatio: 0, upperWickRatio: 0 };
  const current = rows[rows.length - 1], prior = rows.slice(-8, -1), priorLow = Math.min(...prior.map((x) => x.l)), priorHigh = Math.max(...prior.map((x) => x.h));
  const body = Math.max(Math.abs(current.c - current.o), current.c * .00002), lower = Math.min(current.o, current.c) - current.l, upper = current.h - Math.max(current.o, current.c);
  return { bull: current.l < priorLow && m.price > priorLow, bear: current.h > priorHigh && m.price < priorHigh, lowerWickRatio: lower / body, upperWickRatio: upper / body };
}
function setupFor(m: Market, dir: number, sw: ReturnType<typeof sweep>, htfLong: number) {
  if (dir > 0 && sw.bull && m.flow.ofi > .05 && m.book.pressure > 1.01) return "LIQUIDITY_SWEEP_REVERSAL";
  if (dir < 0 && sw.bear && m.flow.ofi < -.05 && m.book.pressure < .99) return "FAILED_BREAK_REVERSAL";
  if (dir > 0 && m.flow.retPct >= Math.max(.04, m.f1.atrPct * .18) && m.f1.trend > .35) return "BREAKOUT_RETEST_CONTINUATION";
  if (dir > 0 && htfLong > .08 && m.f5.trend > .24 && m.price >= m.f1.ema9 * .998) return "PULLBACK_CONTINUATION";
  if (Math.abs(htfLong) < .22 && dir > 0 && m.f1.z < -1.0) return "RANGE_REJECTION";
  if (dir < 0 && -htfLong > .20) return "TREND_EXHAUSTION";
  return dir > 0 ? "DIP_RECLAIM" : dir < 0 ? "DOWNTREND_BREAK" : "NO_CLEAR_SETUP";
}
function setupBaseQuality(setup: string) {
  return ({ LIQUIDITY_SWEEP_REVERSAL: 1, FAILED_BREAK_REVERSAL: .96, BREAKOUT_RETEST_CONTINUATION: .92, PULLBACK_CONTINUATION: .82, RANGE_REJECTION: .80, TREND_EXHAUSTION: .84, DIP_RECLAIM: .72, DOWNTREND_BREAK: .76 } as Record<string, number>)[setup] ?? .60;
}
function desiredFraction(q: number) {
  if (q < .60) return .08;
  if (q < .68) return .10 + (q - .60) / .08 * .10;
  if (q < .76) return .20 + (q - .68) / .08 * .14;
  if (q < .84) return .34 + (q - .76) / .08 * .12;
  if (q < .91) return .46 + (q - .84) / .07 * .14;
  return Math.min(MAX_POSITION_FRACTION, .60 + (q - .91) / .08 * .12);
}
function decision(m: Market, btc: Market, futures: Set<string>, brain: Json, cfg: Json): Decision {
  const rg = regime(m, btc), sw = sweep(m), pressure = clip((m.book.pressure - 1) / .50, -1, 1), htfLong = .58 * m.f15.trend + .42 * m.f1h.trend;
  const structure0 = .46 * m.f5.trend + .30 * m.f15.trend + .24 * m.f1h.trend + (sw.bull ? .28 : 0) - (sw.bear ? .28 : 0);
  const trend0 = .25 * m.f5.trend + .30 * m.f15.trend + .45 * m.f1h.trend;
  const rsiBias = clip((m.f1.rsi - 50) / 50, -1, 1), pulseBias = clip(m.flow.retPct / Math.max(.08, m.f1.atrPct * .45), -1, 1);
  const momentum0 = .34 * m.f1.trend + .28 * rsiBias + .38 * pulseBias;
  const volume0 = clip(.50 * clip((m.f1.volRel - 1) / 1.5, -1, 1) * sign(m.flow.retPct || m.flow.ofi) + .50 * m.flow.ofi, -1, 1);
  let meanRev0 = clip(-m.f1.z * .24, -.55, .55); if (Math.abs(trend0) > .35) meanRev0 *= .62;
  const micro0 = .62 * m.flow.ofi + .38 * pressure;
  const biases: Record<string, number> = { structure: structure0, trend: trend0, momentum: momentum0, volume: volume0, mean_reversion: meanRev0, microstructure: micro0 };
  const experts = brain.experts && typeof brain.experts === "object" ? brain.experts as Json : {};
  const votes: Vote[] = Object.entries(biases).map(([name, bias]) => {
    const confBase = ({ structure: .58, trend: .55, momentum: .50, volume: .48, mean_reversion: .48, microstructure: .54 } as Record<string, number>)[name] ?? .5;
    const confSlope = ({ structure: .30, trend: .34, momentum: .34, volume: .35, mean_reversion: .32, microstructure: .36 } as Record<string, number>)[name] ?? .3;
    const confidence = clip(confBase + confSlope * Math.abs(bias), 0, 1);
    return { name, bias: clip(bias, -1, 1), confidence, weight: (BASE_WEIGHTS[name] ?? .15) * regimeWeight(name, rg) * betaReliability(experts[name]) };
  });
  const total = sum(votes.map((v) => v.weight * Math.max(.08, v.confidence)));
  const raw = total ? sum(votes.map((v) => v.bias * v.weight * Math.max(.08, v.confidence))) / total : 0, dir = sign(raw);
  const same = dir ? sum(votes.filter((v) => sign(v.bias) === dir).map((v) => v.weight * Math.max(.08, v.confidence))) : 0;
  const agreement = total ? same / total : 0, uncertainty = clip(1 - agreement, 0, 1);
  let confidence = clip(.45 + .36 * Math.abs(raw) + .28 * Math.max(0, agreement - .45), 0, .99);
  const feeSideBps = dir < 0 ? FUTURES_FEE_BPS : n(cfg.fee_bps, 10), baseline = Math.max(50, n(cfg.starting_equity, 1000) * .25);
  const impact = impactBps(m.book, baseline, dir < 0 ? "SELL" : "BUY"), costBps = 2 * feeSideBps + Math.max(0, m.book.spreadBps) + 2 * impact;
  const h15s = Math.max(0, Math.abs(m.flow.retPct) * 100 + 10 * Math.max(0, m.flow.ofi * (dir || 1)));
  const h1 = Math.max(0, m.f1.atrPct * 100 * (.55 + .45 * Math.max(0, m.f1.trend * (dir || 1))));
  const expected5 = Math.max(m.f5.atrPct * 100 * 1.55, m.f15.atrPct * 100 * .55);
  const h5 = Math.max(0, expected5 * (.55 + .45 * Math.max(0, m.f5.trend * (dir || 1))));
  const h15m = Math.max(0, m.f15.atrPct * 100 * (.48 + .52 * Math.max(0, m.f15.trend * (dir || 1))));
  const grossEdgeBps = Math.max(h15s, h1, h5, h15m), netEdgeBps = grossEdgeBps - costBps;
  const set = setupFor(m, dir, sw, htfLong), setRel = setupReliability(brain, set); confidence = clip(confidence * (.86 + .14 * setRel), 0, .99);
  const edgeRatio = costBps > 0 ? grossEdgeBps / costBps : 0, btcRisk = .55 * btc.f15.trend + .45 * btc.f1h.trend, vetoReasons: string[] = [];
  if (m.book.spreadBps > MAX_ENTRY_SPREAD_BPS) vetoReasons.push("SPREAD_TOO_WIDE");
  if (edgeRatio < COST_EDGE_MULT) vetoReasons.push("EDGE_BELOW_COST");
  if (dir > 0 && btcRisk < -.48) vetoReasons.push("BTC_STRESS_LONG");
  if (dir < 0 && !futures.has(m.symbol)) vetoReasons.push("NO_PERP_MARKET");
  if (agreement < BRAIN_MIN_AGREEMENT) vetoReasons.push("EXPERT_DISAGREEMENT");
  if (confidence < BRAIN_MIN_CONF) vetoReasons.push("LOW_CONFIDENCE");
  if (Math.abs(raw) < BRAIN_MIN_EDGE) vetoReasons.push("EDGE_TOO_SMALL");
  if (netEdgeBps <= Math.max(2, costBps * .35)) vetoReasons.push("NET_EDGE_TOO_SMALL");
  const setupQ = setupBaseQuality(set), edgeQ = clip(netEdgeBps / Math.max(20, costBps * 5), 0, 1);
  const quality = clip(.34 * confidence + .26 * agreement + .22 * edgeQ + .18 * setupQ, 0, 1);
  if (quality < BRAIN_MIN_QUALITY) vetoReasons.push("BRAIN_QUALITY_TOO_LOW");
  const action: "BUY" | "SELL" | "WAIT" = vetoReasons.length || !dir ? "WAIT" : dir > 0 ? "BUY" : "SELL";
  const df = desiredFraction(quality), riskBudgetPct = clip(.0035 + Math.max(0, quality - .58) / .42 * .009, .0035, .0125);
  return { action, edge: clip(raw, -1, 1), confidence, agreement, uncertainty, setup: set, regime: rg, grossEdgeBps, costBps, netEdgeBps, quality, desiredFraction: df, riskBudgetPct, vetoReasons, votes, hardDrift: false };
}
function slim(d: Decision): Json {
  return { ts: Date.now(), action: d.action, edge: d.edge, confidence: d.confidence, agreement: d.agreement, uncertainty: d.uncertainty, setup: d.setup, regime: d.regime, grossEdgeBps: d.grossEdgeBps, costBps: d.costBps, netEdgeBps: d.netEdgeBps, brainQuality: d.quality, desiredFraction: d.desiredFraction, riskBudgetPct: d.riskBudgetPct, vetoReasons: d.vetoReasons, votes: d.votes };
}

async function latestSession(): Promise<Session | null> {
  const q = await db.from("brian_dip_session_events").select("session_id,event_kind,requested_at,starting_equity,trade_notional,config").order("requested_at", { ascending: false }).order("event_id", { ascending: false }).limit(1).maybeSingle();
  if (q.error) throw q.error; if (!q.data || q.data.event_kind !== "START") return null;
  const sessionId = String(q.data.session_id);
  const s = await db.from("brian_dip_session_events").select("session_id,requested_at,starting_equity,trade_notional,config").eq("session_id", sessionId).eq("event_kind", "START").order("requested_at", { ascending: true }).limit(1).maybeSingle();
  if (s.error || !s.data) throw s.error ?? new Error("DIP_START_MISSING");
  return { session_id: sessionId, started_at: String(s.data.requested_at), starting_equity: n(s.data.starting_equity), trade_notional: n(s.data.trade_notional), config: s.data.config && typeof s.data.config === "object" ? s.data.config as Json : {} };
}
async function latestSnapshot(sessionId: string) {
  const q = await db.from("brian_dip_snapshots").select("snapshot_id,observed_at,cash,equity,realized_pnl,unrealized_pnl,trade_count,win_count,loss_count,state").eq("session_id", sessionId).order("observed_at", { ascending: false }).limit(1).maybeSingle();
  if (q.error) throw q.error; return q.data;
}
async function browserHeartbeatFresh(sessionId: string) {
  const q = await db.from("brian_dip_engine_leases").select("heartbeat_at").eq("session_id", sessionId).maybeSingle();
  if (q.error || !q.data?.heartbeat_at) return false; const t = Date.parse(String(q.data.heartbeat_at)); return Number.isFinite(t) && Date.now() - t < BROWSER_HANDOFF_GRACE_MS;
}
function freshSymbol(sym: string): Json {
  return { symbol: sym, pos: null, last: 0, lastAction: "WATCH", realized: 0, dip: null, armed: false, v4: { phase: "WATCH", armSide: null, armLow: null, armHigh: null, armAt: 0, lastVeto: "WAIT_DATA", lastDecisionAt: 0, lastSignal: null, brainV5: null } };
}
function loadRuntime(session: Session, snap: Awaited<ReturnType<typeof latestSnapshot>>): Runtime {
  const state = snap?.state && typeof snap.state === "object" ? snap.state as Json : {}, serverRuntime = state.serverRuntime && typeof state.serverRuntime === "object" ? state.serverRuntime as Json : {};
  const serverWasAuthoritative = serverRuntime.authoritative === true;
  const symbols = state.symbols && typeof state.symbols === "object" ? structuredClone(state.symbols as Record<string, Json>) : {};
  const config = { ...session.config, server_authoritative: true, sizing_policy: "BRAIN_CONFIDENCE_V7", max_shadow_leverage: 1, max_open_positions: MAX_OPEN, max_portfolio_heat: MAX_PORTFOLIO_HEAT };
  const brain = ensureBrain(state.v5Brain, serverWasAuthoritative);
  if (!serverWasAuthoritative) {
    brain.pending = []; brain.audits = []; brain.outcomes = [];
    brain.legacyBrowserLearningQuarantinedAt = nowIso(); brain.legacyBrowserReason = "browser suspension contaminated hold/outcome timing; V7 server learning starts fresh";
  }
  return {
    start: session.starting_equity, cash: snap ? n(snap.cash, session.starting_equity) : session.starting_equity, realized: snap ? n(snap.realized_pnl) : 0,
    trades: snap ? Math.max(0, Math.trunc(n(snap.trade_count))) : 0, wins: snap ? Math.max(0, Math.trunc(n(snap.win_count))) : 0, losses: snap ? Math.max(0, Math.trunc(n(snap.loss_count))) : 0,
    config, symbols, pairGuard: state.pairGuard && typeof state.pairGuard === "object" ? structuredClone(state.pairGuard as Record<string, number>) : {},
    closedOutcomes: Array.isArray(state.closedOutcomes) ? structuredClone(state.closedOutcomes as Json[]).slice(-30) : [], sessionLossLockUntil: n(state.sessionLossLockUntil), brain, serverWasAuthoritative,
  };
}
function ensureSymbol(rt: Runtime, sym: string) {
  if (!rt.symbols[sym] || typeof rt.symbols[sym] !== "object") rt.symbols[sym] = freshSymbol(sym);
  const st = rt.symbols[sym]; if (!st.v4 || typeof st.v4 !== "object") st.v4 = (freshSymbol(sym).v4 as Json); return st;
}
function positionOf(st: Json) { return st.pos && typeof st.pos === "object" ? st.pos as Json : null; }
function openStates(rt: Runtime) { return Object.values(rt.symbols).filter((st) => positionOf(st)); }
function openNotional(rt: Runtime) { return sum(openStates(rt).map((st) => { const p = positionOf(st)!; return n(p.margin, n(p.notional)); })); }
function markEquity(rt: Runtime, markets: Map<string, Market>) {
  let eq = rt.cash;
  for (const st of openStates(rt)) {
    const p = positionOf(st)!, sym = String(st.symbol ?? ""), m = markets.get(sym), entry = n(p.entry), qty = n(p.qty), margin = n(p.margin, n(p.notional)); if (!m || !(entry > 0) || !(qty > 0)) { eq += margin; continue; }
    const side = String(p.side ?? "LONG"), px = m.price, raw = side === "LONG" ? qty * (px - entry) : qty * (entry - px), feeBps = String(p.venue) === "USDM_PERP" ? FUTURES_FEE_BPS : n(rt.config.fee_bps, 10), exitFee = Math.max(0, qty * px) * feeBps / 10000;
    eq += Math.max(0, margin + raw - exitFee);
  }
  return Math.max(0.01, eq);
}
function addEvent(events: Json[], sessionId: string, kind: string, sym: string | null, price: number | null, st: Json | null, extra: Json = {}) {
  events.push({ event_id: `dip-server-v7-${crypto.randomUUID()}`, session_id: sessionId, observed_at: nowIso(), event_kind: kind, symbol: sym, price, dip_low: st?.dip ?? null, entry_price: extra.entry_price ?? positionOf(st ?? {})?.entry ?? null, exit_price: extra.exit_price ?? null, quantity: extra.quantity ?? positionOf(st ?? {})?.qty ?? null, notional: extra.notional ?? positionOf(st ?? {})?.notional ?? null, fees: extra.fees ?? null, realized_pnl: extra.realized_pnl ?? null, cash_after: extra.cash_after ?? null, equity_after: extra.equity_after ?? null, metadata: { server_v7: true, authoritative: true, worker_version: WORKER_VERSION, ...(extra.metadata && typeof extra.metadata === "object" ? extra.metadata as Json : {}) }, evidence_class: EVIDENCE, shadow_only: true, live_execution: false });
}
function recordOutcome(rt: Runtime, sym: string, pnl: number, reason: string) {
  const o = { t: Date.now(), sym, pnl, reason }; rt.closedOutcomes.push(o); rt.closedOutcomes = rt.closedOutcomes.slice(-30);
  const pair = rt.closedOutcomes.filter((x) => String(x.sym) === sym).slice(-3), losses = pair.filter((x) => n(x.pnl) <= 0).length;
  if (pair.length >= 3 && losses >= 2) rt.pairGuard[sym] = Date.now() + PAIR_LOCK_MS;
  const recent = rt.closedOutcomes.slice(-10), bad = recent.filter((x) => n(x.pnl) < 0 && ["HARD_5M_STRUCTURE_STOP", "FLOW_INVALIDATION", "TIME_EXIT", "HTF_INVALIDATION"].includes(String(x.reason))).length;
  if (recent.length >= 5 && bad >= 3) rt.sessionLossLockUntil = Math.max(rt.sessionLossLockUntil, Date.now() + GLOBAL_LOCK_MS);
}
function learnClosed(rt: Runtime, q: Json, pnl: number, reason: string) {
  const brain = rt.brain, d = q.brainV5 && typeof q.brainV5 === "object" ? q.brainV5 as Json : null; if (!d) return;
  const won = pnl > 0, setup = String(d.setup ?? q.mode ?? "UNKNOWN"), sr = brainRow(brain, "setups", setup); sr.n = n(sr.n) + 1; sr.wins = n(sr.wins) + (won ? 1 : 0); sr.pnl = n(sr.pnl) + pnl; sr.mfe = n(sr.mfe) + n(q.mfePct); sr.mae = n(sr.mae) + n(q.maePct);
  const votes = Array.isArray(d.votes) ? d.votes : [];
  for (const raw of votes) if (raw && typeof raw === "object") { const v = raw as Json, er = brainRow(brain, "experts", String(v.name ?? "unknown")); er.n = n(er.n) + 1; const correct = (won && sign(n(v.bias)) === sign(n(d.edge))) || (!won && sign(n(v.bias)) !== sign(n(d.edge))); er.wins = n(er.wins) + (correct ? 1 : 0); er.pnl = n(er.pnl) + pnl * Math.abs(n(v.bias)); }
  const outcomes = Array.isArray(brain.outcomes) ? brain.outcomes as unknown[] : []; outcomes.push({ t: Date.now(), sym: q.symbol ?? null, setup, pnl, reason, mfePct: n(q.mfePct), maePct: n(q.maePct), server_v7: true }); brain.outcomes = outcomes.slice(-80); brain.lastLearningAt = Date.now();
}
async function closePosition(rt: Runtime, sessionId: string, st: Json, m: Market, reason: string, events: Json[]) {
  const q = positionOf(st); if (!q) return; const sym = String(st.symbol), side = String(q.side ?? "LONG") as "LONG" | "SHORT", venue = String(q.venue ?? "SPOT");
  const book = venue === "USDM_PERP" ? await perpBook(sym) : m.book, fill = fillPrice(book, side, n(q.notional, n(q.margin)), true, n(rt.config.slippage_bps, 1)), exit = fill.px;
  const entry = n(q.entry), qty = n(q.qty), raw = side === "LONG" ? qty * (exit - entry) : qty * (entry - exit), exitNotional = Math.max(0, qty * exit), feeSideBps = venue === "USDM_PERP" ? FUTURES_FEE_BPS : n(rt.config.fee_bps, 10), exitFee = exitNotional * feeSideBps / 10000, pnl = raw - n(q.entryFee) - exitFee, margin = n(q.margin, n(q.notional));
  rt.cash += Math.max(0, margin + raw - exitFee); rt.realized += pnl; rt.trades++; pnl > 0 ? rt.wins++ : rt.losses++; st.realized = n(st.realized) + pnl; st.pos = null; st.armed = false; st.dip = null; st.lastAction = "COOLDOWN";
  const v4 = st.v4 as Json; v4.phase = "WATCH"; v4.armSide = null; v4.armAt = 0; v4.lastVeto = reason;
  recordOutcome(rt, sym, pnl, reason); learnClosed(rt, { ...q, symbol: sym }, pnl, reason);
  addEvent(events, sessionId, side === "LONG" ? "SELL" : "SHORT_CLOSE", sym, m.price, st, { entry_price: entry, exit_price: exit, quantity: qty, notional: n(q.notional), fees: n(q.entryFee) + exitFee, realized_pnl: pnl, cash_after: rt.cash, metadata: { side, venue, mode: q.mode, exit_reason: reason, hold_min: (Date.now() - Date.parse(String(q.at))) / 60000, stop_pct: q.stopPct, target_pct: q.targetPct, brain_quality: (q.brainSizing as Json | undefined)?.quality, slippage_bps: fill.slipBps, spread_bps: fill.spreadBps } });
}
async function managePosition(rt: Runtime, sessionId: string, st: Json, m: Market, events: Json[]) {
  const q = positionOf(st); if (!q) return; const side = String(q.side ?? "LONG") as "LONG" | "SHORT", entry = n(q.entry), p = m.price, move = side === "LONG" ? pct(p, entry) : pct(entry, p), hold = Date.now() - Date.parse(String(q.at));
  q.bestPrice = side === "LONG" ? Math.max(n(q.bestPrice, p), p) : Math.min(n(q.bestPrice, p), p); q.mfePct = Math.max(n(q.mfePct), move); q.maePct = Math.max(n(q.maePct), Math.max(0, -move));
  if ((side === "LONG" && p <= n(q.stop)) || (side === "SHORT" && p >= n(q.stop))) return await closePosition(rt, sessionId, st, m, "HARD_5M_STRUCTURE_STOP", events);
  const entrySwingLow = n(q.entrySwingLow), entrySwingHigh = n(q.entrySwingHigh), last5 = m.bars5[m.bars5.length - 1];
  if (side === "LONG" && entrySwingLow > 0 && last5.c < entrySwingLow && move < n(q.costPct) * .5) return await closePosition(rt, sessionId, st, m, "HTF_INVALIDATION", events);
  if (side === "SHORT" && entrySwingHigh > 0 && last5.c > entrySwingHigh && move < n(q.costPct) * .5) return await closePosition(rt, sessionId, st, m, "HTF_INVALIDATION", events);
  const flowBad = side === "LONG" ? (m.flow.ofi < -.28 && m.book.pressure < .88) : (m.flow.ofi > .28 && m.book.pressure > 1.14);
  if (move < -Math.max(n(q.costPct) * .55, m.f5.atrPct * .30) && flowBad) return await closePosition(rt, sessionId, st, m, "FLOW_INVALIDATION", events);
  if (move >= n(q.trailArmPct, 999)) {
    const floorPct = n(q.costPct) * 1.12, gap = n(q.trailGapPct, .2);
    if (side === "LONG") q.trailPrice = Math.max(n(q.trailPrice), entry * (1 + floorPct / 100), n(q.bestPrice) * (1 - gap / 100));
    else { const ceil = entry * (1 - floorPct / 100), trail = n(q.bestPrice) * (1 + gap / 100); q.trailPrice = q.trailPrice == null ? Math.min(ceil, trail) : Math.min(n(q.trailPrice), ceil, trail); }
  }
  if (q.trailPrice != null && ((side === "LONG" && p <= n(q.trailPrice)) || (side === "SHORT" && p >= n(q.trailPrice)))) return await closePosition(rt, sessionId, st, m, "PROFIT_TRAIL_V7", events);
  if ((side === "LONG" && p >= n(q.target)) || (side === "SHORT" && p <= n(q.target))) return await closePosition(rt, sessionId, st, m, "EXPERT_TARGET_V7", events);
  if (hold >= n(q.maxHoldMs, side === "LONG" ? LONG_HOLD_MS : SHORT_HOLD_MS)) return await closePosition(rt, sessionId, st, m, "TIME_EXIT", events);
}
function updateDipVisual(st: Json, m: Market, events: Json[], sessionId: string) {
  if (positionOf(st)) return; const v4 = st.v4 as Json, p = m.price, swingHigh = m.f5.swingHigh ?? p, drop = Math.max(0, -pct(p, swingHigh)), trigger = clip(m.f5.atrPct * .55, .35, 1.8), idioDip = m.f1.z <= -1.05;
  const wasArmed = st.armed === true;
  if (!wasArmed && (drop >= trigger || idioDip)) {
    st.armed = true; st.dip = p; st.lastAction = "DIP_HUNT"; v4.phase = "ARM_LONG"; v4.armSide = "LONG"; v4.armLow = p; v4.armHigh = swingHigh; v4.armAt = Date.now();
    addEvent(events, sessionId, "DIP_ARMED", String(st.symbol), p, st, { metadata: { trigger_pct: trigger, drop_pct: drop, z: m.f1.z, atr5_pct: m.f5.atrPct } });
  } else if (wasArmed) {
    if (Date.now() - n(v4.armAt) > 12 * 60_000) { st.armed = false; st.dip = null; v4.phase = "WATCH"; v4.lastVeto = "STALE_DIP_ARM"; }
    else if (p < n(v4.armLow, p)) { v4.armLow = p; st.dip = p; addEvent(events, sessionId, "DIP_NEW_LOW", String(st.symbol), p, st, { metadata: { atr5_pct: m.f5.atrPct } }); }
  }
}
function sizePlan(rt: Runtime, d: Decision, m: Market, eq: number, side: "LONG" | "SHORT") {
  const costPct = d.costBps / 100, structure = side === "LONG" && m.f5.swingLow ? Math.max(0, pct(m.book.mid, m.f5.swingLow)) : side === "SHORT" && m.f5.swingHigh ? Math.max(0, pct(m.f5.swingHigh, m.book.mid)) : 0;
  const stopPct = clip(Math.max(m.f5.atrPct * 1.05, costPct * 2.5, structure * .55), .35, 1.60);
  let targetPct = Math.max(costPct * 3.0, m.f5.atrPct * 1.45, stopPct * MIN_RR); targetPct = clip(targetPct, .55, 2.75);
  if (targetPct < stopPct * MIN_RR) return null;
  const riskDollars = eq * d.riskBudgetPct, riskCap = riskDollars / Math.max(.0035, stopPct / 100), desired = eq * d.desiredFraction;
  const heatAvailable = Math.max(0, eq * MAX_PORTFOLIO_HEAT - openNotional(rt));
  const notional = Math.min(riskCap, desired, eq * MAX_POSITION_FRACTION, heatAvailable, rt.cash * .95);
  if (notional < Math.max(12, eq * MIN_POSITION_FRACTION)) return null;
  return { stopPct, targetPct, costPct, notional, heatAvailable, riskDollars, desired, fraction: notional / eq, trailArmPct: Math.max(costPct * 1.65, targetPct * .62), trailGapPct: clip(Math.max(m.f5.atrPct * .38, costPct * .45), .10, .55), maxHoldMs: side === "SHORT" ? SHORT_HOLD_MS : LONG_HOLD_MS };
}
function queueCounterfactual(rt: Runtime, sym: string, side: "LONG" | "SHORT", p: number, d: Decision, reason: string) {
  const brain = rt.brain, pending = Array.isArray(brain.pending) ? brain.pending as unknown[] : []; const last = pending.findLast?.((x) => x && typeof x === "object" && String((x as Json).sym) === sym) as Json | undefined;
  if (last && Date.now() - n(last.createdAt) < 60_000) return;
  pending.push({ id: `v7-miss-${crypto.randomUUID()}`, sym, side, entry: p, createdAt: Date.now(), dueAt: Date.now() + BRAIN_MISS_HORIZON_MS, bestPx: p, worstPx: p, reason, decision: slim(d), server_v7: true }); brain.pending = pending.slice(-48);
  const ab = brain.ab as Json; ab.brainOnlyCandidates = n(ab.brainOnlyCandidates) + 1;
}
function resolveCounterfactuals(rt: Runtime, markets: Map<string, Market>) {
  const brain = rt.brain, pending = Array.isArray(brain.pending) ? brain.pending as unknown[] : [], audits = Array.isArray(brain.audits) ? brain.audits as unknown[] : [], keep: unknown[] = [];
  for (const raw of pending) {
    if (!raw || typeof raw !== "object") continue; const x = raw as Json, m = markets.get(String(x.sym)); if (!m) { keep.push(x); continue; }
    const p = m.price; x.bestPx = Math.max(n(x.bestPx, p), p); x.worstPx = Math.min(n(x.worstPx, p), p); if (Date.now() < n(x.dueAt)) { keep.push(x); continue; }
    const long = String(x.side) === "LONG", mfe = long ? pct(n(x.bestPx), n(x.entry)) : pct(n(x.entry), n(x.worstPx)), mae = long ? Math.max(0, -pct(n(x.worstPx), n(x.entry))) : Math.max(0, -pct(n(x.entry), n(x.bestPx))), costPct = n((x.decision as Json | undefined)?.costBps) / 100;
    const missed = mfe > Math.max(costPct * 1.35, .18), saved = mae > Math.max(costPct * 1.5, .25) && !missed, result = missed ? "MISSED_PROFIT" : saved ? "SAVED_LOSS" : "CORRECT_ABSTENTION";
    const vr = brainRow(brain, "vetoes", String(x.reason ?? "WAIT")); vr.n = n(vr.n) + 1; const ab = brain.ab as Json;
    if (missed) { vr.missedWin = n(vr.missedWin) + 1; ab.missedWins = n(ab.missedWins) + 1; } else if (saved) { vr.savedLoss = n(vr.savedLoss) + 1; ab.savedLosses = n(ab.savedLosses) + 1; } else { vr.neutral = n(vr.neutral) + 1; ab.correctAbstentions = n(ab.correctAbstentions) + 1; }
    audits.push({ id: x.id, sym: x.sym, side: x.side, reason: x.reason, result, mfePct: mfe, maePct: mae, at: Date.now(), decision: x.decision, server_v7: true });
  }
  brain.pending = keep.slice(-48); brain.audits = audits.slice(-80); brain.lastLearningAt = Date.now();
}
async function openPosition(rt: Runtime, sessionId: string, st: Json, m: Market, d: Decision, futures: Set<string>, events: Json[]) {
  const side: "LONG" | "SHORT" = d.action === "SELL" ? "SHORT" : "LONG", sym = String(st.symbol), eq = markEquity(rt, new Map([[sym, m]]));
  if (Date.now() < n(rt.pairGuard[sym])) { (st.v4 as Json).lastVeto = "PAIR_COOLDOWN"; queueCounterfactual(rt, sym, side, m.price, d, "PAIR_COOLDOWN"); return; }
  if (Date.now() < rt.sessionLossLockUntil) { (st.v4 as Json).lastVeto = "GLOBAL_STOP_GUARD"; queueCounterfactual(rt, sym, side, m.price, d, "GLOBAL_STOP_GUARD"); return; }
  if (eq <= rt.start * .96) { rt.sessionLossLockUntil = Date.now() + DD_LOCK_MS; (st.v4 as Json).lastVeto = "SESSION_DD_GUARD"; return; }
  if (openStates(rt).length >= MAX_OPEN) { (st.v4 as Json).lastVeto = "MAX_OPEN"; queueCounterfactual(rt, sym, side, m.price, d, "MAX_OPEN"); return; }
  if (side === "SHORT" && !futures.has(sym)) { (st.v4 as Json).lastVeto = "NO_PERP_MARKET"; return; }
  const plan = sizePlan(rt, d, m, eq, side); if (!plan) { (st.v4 as Json).lastVeto = "RISK_SIZE_TOO_SMALL_OR_HEAT"; queueCounterfactual(rt, sym, side, m.price, d, "PORTFOLIO_HEAT"); return; }
  if (openNotional(rt) + plan.notional > eq * MAX_PORTFOLIO_HEAT + 1e-6) { (st.v4 as Json).lastVeto = "PORTFOLIO_HEAT"; queueCounterfactual(rt, sym, side, m.price, d, "PORTFOLIO_HEAT"); return; }
  const venue = side === "SHORT" ? "USDM_PERP" : "SPOT", book = side === "SHORT" ? await perpBook(sym) : m.book;
  if (!(book.mid > 0) || book.spreadBps > MAX_ENTRY_SPREAD_BPS) { (st.v4 as Json).lastVeto = "SPREAD_TOO_WIDE"; return; }
  const fill = fillPrice(book, side, plan.notional, false, n(rt.config.slippage_bps, 1)), entry = fill.px, feeSideBps = side === "SHORT" ? FUTURES_FEE_BPS : n(rt.config.fee_bps, 10), entryFee = plan.notional * feeSideBps / 10000, required = plan.notional + entryFee;
  if (rt.cash < required) { (st.v4 as Json).lastVeto = "NO_CASH"; addEvent(events, sessionId, "NO_CASH", sym, m.price, st, { metadata: { available: rt.cash, required, brain_quality: d.quality } }); return; }
  const qty = plan.notional / entry, stop = side === "LONG" ? entry * (1 - plan.stopPct / 100) : entry * (1 + plan.stopPct / 100), target = side === "LONG" ? entry * (1 + plan.targetPct / 100) : entry * (1 - plan.targetPct / 100);
  rt.cash -= required; st.pos = { v4: true, serverV7: true, symbol: sym, side, venue, entry, qty, margin: plan.notional, notional: plan.notional, exposure: plan.notional, leverage: 1, entryFee, at: nowIso(), mode: `V7_${d.setup}`, stop, stopPct: plan.stopPct, target, targetPct: plan.targetPct, costPct: plan.costPct, trailArmPct: plan.trailArmPct, trailGapPct: plan.trailGapPct, trailPrice: null, bestPrice: entry, maxHoldMs: plan.maxHoldMs, entrySwingLow: m.f5.swingLow, entrySwingHigh: m.f5.swingHigh, mfePct: 0, maePct: 0, brainV5: slim(d), brainSizing: { policy: "BRAIN_CONFIDENCE_V7", quality: d.quality, confidence: d.confidence, agreement: d.agreement, net_edge_bps: d.netEdgeBps, desired_fraction: d.desiredFraction, actual_fraction: plan.fraction, risk_budget_pct: d.riskBudgetPct, risk_dollars: plan.riskDollars, heat_before: openNotional(rt) / Math.max(eq, 1), heat_cap: MAX_PORTFOLIO_HEAT } };
  st.armed = false; st.dip = null; st.lastAction = side; const v4 = st.v4 as Json; v4.phase = "POSITION"; v4.armSide = null; v4.lastVeto = null; v4.brainV5 = slim(d);
  const ab = rt.brain.ab as Json; ab.attempted = n(ab.attempted) + 1; ab.accepted = n(ab.accepted) + 1;
  addEvent(events, sessionId, side === "LONG" ? "BUY" : "SHORT_OPEN", sym, m.price, st, { entry_price: entry, quantity: qty, notional: plan.notional, fees: entryFee, cash_after: rt.cash, metadata: { side, venue, mode: `V7_${d.setup}`, stop_price: stop, stop_pct: plan.stopPct, target_price: target, target_pct: plan.targetPct, cost_pct: plan.costPct, spread_bps: book.spreadBps, impact_bps: fill.impactBps, ofi: m.flow.ofi, book_pressure: m.book.pressure, brain_quality: d.quality, brain_confidence: d.confidence, brain_agreement: d.agreement, brain_net_edge_bps: d.netEdgeBps, desired_fraction: d.desiredFraction, actual_fraction: plan.fraction, risk_budget_pct: d.riskBudgetPct, sizing_policy: "BRAIN_CONFIDENCE_V7" } });
}
async function recordRun(startedAt: string, status: string, observed: number, stored: number, degraded: string[], metadata: Json = {}, error?: unknown) {
  const finishedAt = nowIso(), runId = await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q = await db.from("brian_collector_runs").insert({ run_id: runId, collector_id: COLLECTOR_ID, started_at: startedAt, finished_at: finishedAt, status, observed_records: observed, stored_records: stored, degraded_sources: degraded, error_class: error ? "DIP_SERVER_V7_ERROR" : null, error_message: error ? errorText(error).slice(0, 1500) : null, evidence_class: EVIDENCE, shadow_only: true, live_execution: false, metadata: { worker_version: WORKER_VERSION, server_authoritative: true, sizing_policy: "BRAIN_CONFIDENCE_V7", ...metadata } });
  if (q.error) console.error("run log failed", q.error.message);
}

async function runWorker() {
  const startedAt = nowIso(), session = await latestSession();
  if (!session) { await recordRun(startedAt, "SUCCESS", 0, 0, [], { status: "NO_ACTIVE_SESSION" }); return { status: "NO_ACTIVE_SESSION", shadow_only: true, live_execution: false }; }
  const previous = await latestSnapshot(session.session_id), oldState = previous?.state && typeof previous.state === "object" ? previous.state as Json : {}, oldServer = oldState.serverRuntime && typeof oldState.serverRuntime === "object" ? oldState.serverRuntime as Json : {};
  if (oldServer.authoritative !== true && await browserHeartbeatFresh(session.session_id)) {
    await recordRun(startedAt, "SUCCESS", 0, 0, [], { status: "WAIT_BROWSER_HANDOFF", session_id: session.session_id });
    return { status: "WAIT_BROWSER_HANDOFF", session_id: session.session_id, shadow_only: true, live_execution: false };
  }
  const rt = loadRuntime(session, previous), symbols = Array.isArray(session.config.symbols) ? [...new Set((session.config.symbols as unknown[]).map((x) => String(x).toUpperCase()).filter((x) => /^[A-Z0-9]+USDT$/.test(x)))].slice(0, 12) : [];
  if (!symbols.length) throw new Error("NO_DIP_SYMBOLS");
  for (const sym of symbols) ensureSymbol(rt, sym);
  const wanted = [...new Set([...symbols, "BTCUSDT"])], { markets, degraded } = await loadMarkets(wanted), btc = markets.get("BTCUSDT"); if (!btc) throw new Error("BTC_CONTEXT_UNAVAILABLE");
  const futures = await futuresSymbols(); const events: Json[] = [];
  if (!rt.serverWasAuthoritative) addEvent(events, session.session_id, "INFO", null, null, null, { metadata: { info: "SERVER_AUTHORITATIVE_TAKEOVER_V7", browser_learning_quarantined: true, cadence_seconds: 60, max_position_fraction: MAX_POSITION_FRACTION, max_portfolio_heat: MAX_PORTFOLIO_HEAT } });
  resolveCounterfactuals(rt, markets);
  for (const sym of symbols) {
    const st = ensureSymbol(rt, sym), m = markets.get(sym); if (!m) continue; st.last = m.price;
    if (positionOf(st)) await managePosition(rt, session.session_id, st, m, events);
  }
  let eq = markEquity(rt, markets);
  for (const sym of symbols) {
    const st = ensureSymbol(rt, sym), m = markets.get(sym); if (!m || positionOf(st)) continue; updateDipVisual(st, m, events, session.session_id);
    const d = decision(m, btc, futures, rt.brain, { ...rt.config, starting_equity: eq }); (st.v4 as Json).brainV5 = slim(d); (st.v4 as Json).lastDecisionAt = Date.now();
    if (d.action === "WAIT") { (st.v4 as Json).lastVeto = d.vetoReasons[0] ?? "WAIT"; continue; }
    await openPosition(rt, session.session_id, st, m, d, futures, events); eq = markEquity(rt, markets);
  }
  eq = markEquity(rt, markets); const unrealized = eq - rt.cash - sum(openStates(rt).map((st) => n(positionOf(st)?.margin))) + sum(openStates(rt).map((st) => n(positionOf(st)?.margin)));
  const state: Json = {
    v4: true, startingEquity: rt.start, cash: rt.cash, realizedPnl: rt.realized, tradeCount: rt.trades, winCount: rt.wins, lossCount: rt.losses, config: rt.config, v4Universe: symbols,
    symbols: rt.symbols, pairGuard: rt.pairGuard, closedOutcomes: rt.closedOutcomes.slice(-20), sessionLossLockUntil: rt.sessionLossLockUntil, v5Brain: rt.brain,
    isolation: { cashbox_id: "DIP_SHADOW_CASHBOX_V7", intelligence_version: WORKER_VERSION, main_runtime_mutation: false, main_phase37_mutation: false },
    serverRuntime: { authoritative: true, worker_version: WORKER_VERSION, generated_at: nowIso(), cadence_seconds: 60, market_source: "BINANCE_PUBLIC_REST", decision_owner: "BRIAN_V7", sizing_policy: "BRAIN_CONFIDENCE_V7", max_open: MAX_OPEN, max_position_fraction: MAX_POSITION_FRACTION, max_portfolio_heat: MAX_PORTFOLIO_HEAT, browser_executor_disabled: true, shadow_only: true, live_execution: false, degraded_markets: degraded },
  };
  const snapId = `dip-server-v7-snap-${crypto.randomUUID()}`;
  const snap = await db.from("brian_dip_snapshots").insert({ snapshot_id: snapId, session_id: session.session_id, observed_at: nowIso(), cash: rt.cash, equity: eq, realized_pnl: rt.realized, unrealized_pnl: eq - rt.start - rt.realized, trade_count: rt.trades, win_count: rt.wins, loss_count: rt.losses, state, evidence_class: EVIDENCE, shadow_only: true, live_execution: false });
  if (snap.error) throw snap.error;
  if (events.length) {
    const withEq = events.map((e) => ({ ...e, cash_after: e.cash_after ?? rt.cash, equity_after: e.equity_after ?? eq }));
    const ins = await db.from("brian_dip_events").insert(withEq); if (ins.error) throw ins.error;
  }
  await recordRun(startedAt, degraded.length ? "DEGRADED" : "SUCCESS", symbols.length, 1 + events.length, degraded, { session_id: session.session_id, equity: eq, realized_pnl: rt.realized, trade_count: rt.trades, open_positions: openStates(rt).length, events: events.length });
  return { status: degraded.length ? "DEGRADED" : "CAPTURED", session_id: session.session_id, equity: eq, realized_pnl: rt.realized, trades: rt.trades, open: openStates(rt).length, events: events.length, degraded_markets: degraded, server_authoritative: true, sizing_policy: "BRAIN_CONFIDENCE_V7", shadow_only: true, live_execution: false };
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return json({ error: "POST required" }, 405);
  try { await requireCronAuth(req, db); }
  catch (e) { const message = errorText(e), unauthorized = message.includes("UNAUTHORIZED_CRON"); return json({ status: unauthorized ? "UNAUTHORIZED" : "FAILED_CLOSED", error: message, shadow_only: true, live_execution: false }, unauthorized ? 401 : 503); }
  const startedAt = nowIso();
  try {
    const last = await db.from("brian_collector_runs").select("started_at").eq("collector_id", COLLECTOR_ID).in("status", ["SUCCESS", "DEGRADED"]).order("started_at", { ascending: false }).limit(1).maybeSingle();
    if (last.error) throw last.error; if (last.data?.started_at) { const age = (Date.now() - Date.parse(String(last.data.started_at))) / 1000; if (Number.isFinite(age) && age < MIN_INTERVAL_SECONDS) return json({ status: "SKIPPED_RATE_GUARD", age_seconds: age, shadow_only: true, live_execution: false }); }
    const lease = await withCollectorLease(db, COLLECTOR_ID, LEASE_SECONDS, async () => await runWorker());
    if (lease.contended) return json({ status: "SKIPPED_LEASE_CONTENDED", shadow_only: true, live_execution: false }); return json(lease.value!);
  } catch (e) {
    console.error("brian-dip-shadow-worker-v7 failed", errorText(e)); try { await recordRun(startedAt, "FAILED", 0, 0, [], {}, e); } catch { /* primary error wins */ }
    return json({ status: "FAILED_CLOSED", error: errorText(e), server_authoritative: true, shadow_only: true, live_execution: false }, 500);
  }
});
