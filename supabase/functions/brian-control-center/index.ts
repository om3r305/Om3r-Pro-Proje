import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const CORE = `${URL}/functions/v1/brian-control-center-core`;

function finite(v: unknown): number | null {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
function ageSeconds(v: string | null): number | null {
  if (!v) return null;
  const t = Date.parse(v);
  return Number.isFinite(t) ? Math.max(0, Math.round((Date.now() - t) / 1000)) : null;
}

async function v8DipSummary() {
  const latestEventQ = await db.from("brian_dip_session_events")
    .select("event_id,session_id,event_kind,requested_at")
    .order("requested_at", { ascending: false }).order("event_id", { ascending: false })
    .limit(1).maybeSingle();
  if (latestEventQ.error) throw latestEventQ.error;
  if (!latestEventQ.data) return null;
  const sid = String(latestEventQ.data.session_id);

  const [runtimeQ, startQ, lastQ] = await Promise.all([
    db.from("brian_dip_v8_runtime")
      .select("session_id,state_version,updated_at,runtime,snapshot,shadow_only,live_execution")
      .eq("session_id", sid).maybeSingle(),
    db.from("brian_dip_session_events")
      .select("event_id,requested_at,starting_equity,trade_notional,config")
      .eq("session_id", sid).eq("event_kind", "START")
      .order("requested_at", { ascending: true }).order("event_id", { ascending: true })
      .limit(1).maybeSingle(),
    db.from("brian_dip_session_events")
      .select("event_id,event_kind,requested_at")
      .eq("session_id", sid)
      .order("requested_at", { ascending: false }).order("event_id", { ascending: false })
      .limit(1).maybeSingle(),
  ]);
  if (runtimeQ.error) throw runtimeQ.error;
  if (startQ.error) throw startQ.error;
  if (lastQ.error) throw lastQ.error;
  if (!runtimeQ.data || !startQ.data) return null;

  const runtime = (runtimeQ.data.runtime ?? {}) as Record<string, unknown>;
  const storedSnapshot = runtimeQ.data.snapshot as Record<string, unknown> | null;
  const updatedAt = String(runtimeQ.data.updated_at);
  const heartbeatAge = ageSeconds(updatedAt);
  const active = String(lastQ.data?.event_kind ?? "PAUSE") === "START";
  const config = (startQ.data.config ?? {}) as Record<string, unknown>;
  const cash = finite(runtime.cash) ?? finite(startQ.data.starting_equity) ?? 0;
  const position = runtime.pos as Record<string, unknown> | null;
  const mark = finite(position?.market_price) ?? finite(position?.entry);
  const qty = finite(position?.qty) ?? 0;
  const syntheticEquity = position && mark != null ? cash + qty * mark : cash;
  const snapshot = storedSnapshot ?? {
    session_id: sid,
    observed_at: updatedAt,
    cash,
    equity: syntheticEquity,
    realized_pnl: finite(runtime.realized) ?? 0,
    unrealized_pnl: syntheticEquity - (finite(runtime.start) ?? cash) - (finite(runtime.realized) ?? 0),
    trade_count: finite(runtime.trades) ?? 0,
    win_count: finite(runtime.wins) ?? 0,
    loss_count: finite(runtime.losses) ?? 0,
  };

  return {
    status: !active ? "PAUSED" : heartbeatAge != null && heartbeatAge <= 420 ? "BROWSER_ACTIVE" : "BROWSER_STOPPED",
    session_id: sid,
    started_at: startQ.data.requested_at ?? null,
    starting_equity: finite(startQ.data.starting_equity),
    engine_version: String(config.engine_version ?? "brian-dip-chart-reader-v8"),
    latest_snapshot_at: String((snapshot as Record<string, unknown>)?.observed_at ?? updatedAt),
    heartbeat_at: updatedAt,
    heartbeat_age_seconds: heartbeatAge,
    snapshot,
    state_version: Number(runtimeQ.data.state_version ?? 0),
    server_authoritative: config.server_authoritative === true,
    browser_engine_required: false,
    cloud_runner_enabled: true,
    execution_mode: config.execution_mode ?? "SHADOW_PAPER",
    policy_version: config.policy_version ?? null,
    symbol: "ETHUSDT",
    shadow_only: runtimeQ.data.shadow_only !== false,
    live_execution: runtimeQ.data.live_execution === true,
  };
}

type RadarCandidate = {
  symbol: string;
  base_asset: string;
  radar_score: number;
  price_change_pct: number;
  range_pct: number;
  spread_bps: number | null;
  quote_volume: number | null;
  trades_24h: number | null;
  momentum_score: number | null;
  volatility_score: number | null;
  reasons: string[];
};

async function marketRadarSummary() {
  const result = await db.from("brian_universe_snapshots")
    .select("snapshot_id,provider,observed_at,eligible_count,candidates")
    .order("observed_at", { ascending: false })
    .limit(1).maybeSingle();
  if (result.error) throw result.error;
  if (!result.data) {
    return { status: "EMPTY", observed_at: null, age_seconds: null, candidates: [], hot: [] };
  }

  const payload = (result.data.candidates ?? {}) as Record<string, unknown>;
  const rawRows = Array.isArray(payload.candidates) ? payload.candidates : [];
  const candidates: RadarCandidate[] = rawRows.map((value) => {
    const row = (value ?? {}) as Record<string, unknown>;
    const symbol = String(row.symbol ?? "").toUpperCase();
    const baseAsset = String(row.base_asset ?? symbol.replace(/USDT$/i, "")).toUpperCase();
    return {
      symbol,
      base_asset: baseAsset,
      radar_score: finite(row.radar_score) ?? 0,
      price_change_pct: finite(row.price_change_pct) ?? 0,
      range_pct: finite(row.range_pct) ?? 0,
      spread_bps: finite(row.spread_bps),
      quote_volume: finite(row.quote_volume),
      trades_24h: finite(row.trades_24h),
      momentum_score: finite(row.momentum_score),
      volatility_score: finite(row.volatility_score),
      reasons: Array.isArray(row.reasons) ? row.reasons.map((x) => String(x)).slice(0, 6) : [],
    };
  }).filter((row) => /^[A-Z0-9]{2,20}USDT$/.test(row.symbol) && /^[A-Z0-9]{2,16}$/.test(row.base_asset));

  const hot = candidates
    .filter((row) => Math.abs(row.price_change_pct) >= 5 || row.range_pct >= 15 || (row.momentum_score ?? 0) >= 0.8 || (row.volatility_score ?? 0) >= 0.8)
    .sort((a, b) => b.radar_score - a.radar_score)
    .slice(0, 10);
  const observedAt = String(result.data.observed_at ?? "");
  const age = ageSeconds(observedAt || null);

  return {
    status: age != null && age <= 420 ? "ONLINE" : "STALE",
    provider: result.data.provider ?? null,
    snapshot_id: result.data.snapshot_id ?? null,
    observed_at: observedAt || null,
    age_seconds: age,
    eligible_count: Number(result.data.eligible_count ?? 0),
    collector_version: String(payload.collector_version ?? ""),
    candidates: candidates.slice(0, 16),
    hot,
    shadow_only: true,
    live_execution: false,
  };
}

Deno.serve(async (req: Request) => {
  const bodyText = req.method === "POST" ? await req.text() : "";
  let action = "";
  try {
    action = String((JSON.parse(bodyText || "{}") as Record<string, unknown>).action ?? "status").toLowerCase();
  } catch {
    action = "";
  }

  const headers = new Headers(req.headers);
  headers.delete("host");
  headers.delete("content-length");
  const core = await fetch(CORE, {
    method: req.method,
    headers,
    body: req.method === "POST" ? bodyText : undefined,
  });
  const coreText = await core.text();

  if (!core.ok || req.method !== "POST" || action !== "status") {
    const outHeaders = new Headers(core.headers);
    outHeaders.delete("content-length");
    return new Response(coreText, { status: core.status, headers: outHeaders });
  }

  let data: Record<string, unknown>;
  try {
    data = JSON.parse(coreText) as Record<string, unknown>;
  } catch {
    const outHeaders = new Headers(core.headers);
    outHeaders.delete("content-length");
    return new Response(coreText, { status: core.status, headers: outHeaders });
  }

  try {
    const [dip, radar] = await Promise.all([v8DipSummary(), marketRadarSummary()]);
    if (dip) {
      const system = (data.system ?? {}) as Record<string, unknown>;
      system.dip = dip;
      system.architecture = {
        ...((system.architecture ?? {}) as Record<string, unknown>),
        dip_browser_independent: true,
        dip_server_authoritative: true,
      };
      data.system = system;
    }
    const alpha = (data.alpha_v2 ?? {}) as Record<string, unknown>;
    alpha.market_radar = radar;
    data.alpha_v2 = alpha;
    data.market_radar = radar;
  } catch (e) {
    console.error("control-center-v8-radar-overlay", e instanceof Error ? e.message : String(e));
  }

  const outHeaders = new Headers(core.headers);
  outHeaders.delete("content-length");
  outHeaders.set("content-type", "application/json; charset=utf-8");
  outHeaders.set("cache-control", "no-store");
  outHeaders.set("x-brian-dip-overlay", "v8-server-authoritative+market-radar");
  return new Response(JSON.stringify(data), { status: core.status, headers: outHeaders });
});
