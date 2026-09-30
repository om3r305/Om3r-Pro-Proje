// Daily paper step for the BTC/ETH trend ensemble (see _shared/trend_shadow.ts).
// Runs after the UTC daily close, appends one ledger row per closed day and
// backfills up to MAX_BACKFILL missed days so the NAV path stays continuous.
// Paper only: it reads public Binance candles and writes the shadow ledger.
import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { MIN_HISTORY, step, TREND_ASSETS, TREND_POLICY_VERSION, type TrendAsset, type Weights } from "../_shared/trend_shadow.ts";

const db = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!, {
  auth: { persistSession: false, autoRefreshToken: false },
});
const HOSTS = ["https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com", "https://api3.binance.com"];
const MAX_BACKFILL = 14;
const out = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
const errText = (e: unknown) => (e instanceof Error ? `${e.name}: ${e.message}` : (() => { try { return JSON.stringify(e); } catch { return String(e); } })());

/** Closed daily candles only, keyed by UTC open date (YYYY-MM-DD). */
async function dailyCloses(symbol: TrendAsset): Promise<Map<string, number>> {
  let last: unknown;
  for (const host of HOSTS) {
    try {
      const r = await fetch(`${host}/api/v3/klines?symbol=${symbol}&interval=1d&limit=${MIN_HISTORY + MAX_BACKFILL + 10}`, {
        signal: AbortSignal.timeout(10_000),
      });
      if (!r.ok) throw new Error(`${host} ${r.status}`);
      const rows = await r.json() as unknown[][];
      const now = Date.now();
      return new Map(rows.filter((k) => Number(k[6]) < now).map((k) => [new Date(Number(k[0])).toISOString().slice(0, 10), Number(k[4])]));
    } catch (e) {
      last = e;
    }
  }
  throw new Error(`binance klines unavailable for ${symbol}: ${errText(last)}`);
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ error: "POST required" }, 405);
  try {
    await requireCronAuth(req, db);
  } catch (e) {
    return out({ status: "UNAUTHORIZED", error: errText(e), shadow_only: true, live_execution: false }, 401);
  }
  try {
    const series = Object.fromEntries(await Promise.all(TREND_ASSETS.map(async (a) => [a, await dailyCloses(a)]))) as Record<TrendAsset, Map<string, number>>;
    const days = [...series.BTCUSDT.keys()].filter((d) => series.ETHUSDT.has(d)).sort();
    if (days.length < MIN_HISTORY + 1) throw new Error(`only ${days.length} aligned closed days`);

    const q = await db.from("brian_trend_shadow_ledger").select("day,target_weights,nav")
      .eq("policy_version", TREND_POLICY_VERSION).order("day", { ascending: false }).limit(1).maybeSingle();
    if (q.error) throw q.error;
    let prev: { held: Weights; nav: number } | null = q.data ? { held: q.data.target_weights as Weights, nav: Number(q.data.nav) } : null;
    const lastDay: string | null = q.data?.day ?? null;

    const pending = days.filter((d) => lastDay === null ? d === days[days.length - 1] : d > lastDay);
    if (lastDay !== null && pending.length > MAX_BACKFILL) {
      throw new Error(`ledger gap of ${pending.length} days exceeds backfill limit ${MAX_BACKFILL}; manual review required`);
    }
    if (lastDay !== null && pending.length && days.indexOf(pending[0]) - 1 !== days.indexOf(lastDay)) {
      throw new Error(`candle history no longer contains ${lastDay}; cannot chain NAV`);
    }

    const written: unknown[] = [];
    for (const day of pending) {
      const i = days.indexOf(day);
      const window = days.slice(0, i + 1);
      const closes = Object.fromEntries(TREND_ASSETS.map((a) => [a, window.map((d) => series[a].get(d)!)])) as Record<TrendAsset, number[]>;
      const s = step(day, prev, closes);
      const row = {
        day, policy_version: TREND_POLICY_VERSION,
        closes: Object.fromEntries(TREND_ASSETS.map((a) => [a, s.signals[a].close])),
        signals: s.signals, held_weights: s.held, target_weights: s.target, asset_returns: s.asset_returns,
        day_return: s.day_return, turnover: s.turnover, cost: s.cost, nav: s.nav, rebalanced: s.rebalanced,
        shadow_only: true, live_execution: false,
      };
      const ins = await db.from("brian_trend_shadow_ledger").insert(row);
      if (ins.error) throw ins.error;
      written.push({ day, nav: s.nav, rebalanced: s.rebalanced, target: s.target });
      prev = { held: s.target, nav: s.nav };
    }
    return out({ status: "SUCCESS", policy_version: TREND_POLICY_VERSION, latest_closed_day: days[days.length - 1], written, shadow_only: true, live_execution: false });
  } catch (e) {
    return out({ status: "FAILED_CLOSED", error: errText(e), shadow_only: true, live_execution: false }, 500);
  }
});
