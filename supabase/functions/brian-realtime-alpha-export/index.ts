import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { requireBridgeAuth } from "./bridge_auth.ts";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const VERSION = "brian.realtime-alpha-export.v3-audit-price-path";

function out(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}
function finite(v: unknown, d: number) {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ error: "POST required" }, 405);
  try {
    await requireBridgeAuth(req);
  } catch {
    return out({ status: "UNAUTHORIZED" }, 401);
  }

  const body = await req.json().catch(() => ({})) as Record<string, unknown>;
  const sinceMinutes = Math.max(1, Math.min(60, Math.trunc(finite(body.since_minutes, 15))));
  const limit = Math.max(1, Math.min(500, Math.trunc(finite(body.limit, 250))));
  const since = new Date(Date.now() - sinceMinutes * 60_000).toISOString();

  const q = await db.from("brian_alpha_decisions")
    .select("decision_id,compiler_version,observed_at,asset_id,observed_reference_price,action,direction,evidence_score,independent_group_count,support_groups,conflict_groups,source_observation_ids,source_intrabar_event_ids,requested_virtual_notional_usd,gross_edge_bps,estimated_round_trip_cost_bps,net_edge_bps,veto_reason,reason,metadata,evidence_class,shadow_only,live_execution,created_at")
    .gte("observed_at", since)
    .order("observed_at", { ascending: true })
    .limit(limit);

  if (q.error) {
    return out({ status: "FAILED_CLOSED", version: VERSION, error: q.error.message, rows: [] }, 500);
  }

  let pricePath: Record<string, unknown>[] = [];
  if (body.include_price_path === true) {
    const path = await db.from("brian_intrabar_reaction_events")
      .select("event_id,asset_id,observed_at,observed_mid_price,estimated_round_trip_cost_bps,shadow_only,live_execution")
      .gte("observed_at", since).gt("observed_mid_price", 0)
      .eq("shadow_only", true).eq("live_execution", false)
      .order("observed_at", { ascending: false }).limit(500);
    if (path.error) return out({status:"FAILED_CLOSED",version:VERSION,error:path.error.message,rows:[]},500);
    pricePath = path.data ?? [];
  }
  return out({
    status: "SUCCESS",
    version: VERSION,
    since,
    price_path: pricePath,
    price_path_count: pricePath.length,
    rows: q.data ?? [],
    count: q.data?.length ?? 0,
    shadow_only: true,
    live_execution: false,
  });
});
