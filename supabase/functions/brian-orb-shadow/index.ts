import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { ORB_RULES_VERSION, orbResults, orbRulesFingerprint, parseYahooBars } from "../_shared/orb_shadow.ts";

// Opening-range breakout shadow study orb-shadow-v1 (research/research_lab/studies/orb_shadow/PREREGISTRATION.md).
// Once per US weekday after the close: fetch the last month of public 5-minute bars for the frozen
// universe, apply the frozen rules and append one row per symbol, session and variant. Rows are never
// rewritten (first write wins); sessions before the registration date are labelled RETRO and do not
// count for the decision. Read-only public data; hypothetical trades only; nothing can place an order.

const db = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!, { auth: { persistSession: false, autoRefreshToken: false } });
const ID = "brian-orb-shadow-v1";
const SAFE = { shadow_only: true, live_execution: false, trade_signal: false };
const PROSPECTIVE_FROM = "2026-10-26";
const UNIVERSE: { assetId: string; assetClass: string; symbol: string }[] = [
  ...["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "CVX", "XOM", "JPM"].map((s) => ({ assetId: `equity:${s}`, assetClass: "equity", symbol: s })),
  ...["SPY", "QQQ", "IWM", "SMH", "XLE", "XLF", "USO", "GLD", "TLT", "EWG"].map((s) => ({ assetId: `etf:${s}`, assetClass: "etf", symbol: s })),
];
const out = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });
const errorText = (e: unknown) => e instanceof Error ? `${e.name}: ${e.message}` : String(e);

async function enabled(): Promise<boolean> {
  const q = await db.from("brian_evolution_runtime_config").select("config_value").eq("config_key", "brian_system_enabled").abortSignal(AbortSignal.timeout(4000)).single();
  if (q.error) throw Error("OPERATOR_CONTROL_UNAVAILABLE");
  return q.data?.config_value === "true";
}

async function fetchBars(symbol: string) {
  const r = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=5m&range=1mo`, {
    headers: { accept: "application/json", "user-agent": "Mozilla/5.0 Brian-Shadow-ORB/1.0" }, signal: AbortSignal.timeout(9000),
  });
  if (!r.ok) throw Error(`${symbol}:HTTP_${r.status}`);
  return parseYahooBars(await r.json());
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ error: "POST required" }, 405);
  try { await requireCronAuth(req, db); } catch (e) { return out({ status: "UNAUTHORIZED", error: errorText(e), ...SAFE }, 401); }
  const startedAt = new Date().toISOString();
  try {
    if (!await enabled()) return out({ status: "OPERATOR_STOPPED", ...SAFE });
    const fingerprint = await orbRulesFingerprint();
    const rows: Record<string, unknown>[] = [];
    const degraded: string[] = [];
    for (let i = 0; i < UNIVERSE.length; i += 4) {
      const batch = UNIVERSE.slice(i, i + 4);
      const settled = await Promise.allSettled(batch.map((u) => fetchBars(u.symbol)));
      settled.forEach((s, j) => {
        const u = batch[j];
        if (s.status === "rejected") { degraded.push(errorText(s.reason).slice(0, 80)); return; }
        for (const r of orbResults(s.value.bars, u.assetClass, s.value.tz)) {
          rows.push({
            asset_id: u.assetId, asset_class: u.assetClass, symbol: u.symbol, session_date: r.sessionDate, variant: r.variant,
            period: r.sessionDate >= PROSPECTIVE_FROM ? "PROSPECTIVE" : "RETRO", status: r.status, direction: r.direction,
            or_high: r.orHigh, or_low: r.orLow, or_volume: r.orVolume, rel_volume: r.relVolume, entry_at: r.entryAt, entry_price: r.entryPrice,
            stop_price: r.stopPrice, target_price: r.targetPrice, exit_at: r.exitAt, exit_price: r.exitPrice, exit_reason: r.exitReason,
            gross_bps: r.grossBps, cost_bps: r.costBps, net_bps: r.netBps, rules_version: ORB_RULES_VERSION, rules_fingerprint: fingerprint,
          });
        }
      });
    }
    // A session without its opening range or 15:55 bar (unfinished or gappy) is not stored, so a later
    // complete fetch can still write it; first write wins after that.
    const complete = rows.filter((r) => r.or_high != null);
    let stored = 0;
    for (let i = 0; i < complete.length; i += 500) {
      const q = await db.from("brian_orb_shadow_results").upsert(complete.slice(i, i + 500), { onConflict: "asset_id,session_date,variant", ignoreDuplicates: true }).select("asset_id");
      if (q.error) throw Error(`persist:${q.error.message}`);
      stored += q.data?.length ?? 0;
    }
    const status = degraded.length ? "DEGRADED" : "SUCCESS";
    await db.from("brian_collector_runs").insert({
      run_id: crypto.randomUUID(), collector_id: ID, started_at: startedAt, finished_at: new Date().toISOString(), status,
      observed_records: rows.length, stored_records: stored, degraded_sources: degraded,
      metadata: { rules_version: ORB_RULES_VERSION, rules_fingerprint: fingerprint, symbols: UNIVERSE.length, ...SAFE },
      evidence_class: "PROSPECTIVE_DEVELOPMENT_SHADOW", shadow_only: true, live_execution: false,
    });
    return out({ status, observed: rows.length, stored, degraded, rules_fingerprint: fingerprint, ...SAFE });
  } catch (e) {
    await db.from("brian_collector_runs").insert({
      run_id: crypto.randomUUID(), collector_id: ID, started_at: startedAt, finished_at: new Date().toISOString(), status: "FAILED",
      observed_records: 0, stored_records: 0, degraded_sources: [], error_class: "ORB_SHADOW_ERROR", error_message: errorText(e).slice(0, 1200),
      metadata: { rules_version: ORB_RULES_VERSION, ...SAFE }, evidence_class: "PROSPECTIVE_DEVELOPMENT_SHADOW", shadow_only: true, live_execution: false,
    });
    return out({ status: "FAILED_CLOSED", error: errorText(e), ...SAFE }, 500);
  }
});
