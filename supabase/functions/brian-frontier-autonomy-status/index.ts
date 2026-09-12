import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const AUTH_ID = "control-v3";
const ALLOWED_ORIGIN = /^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const ALLOWED_EXACT = new Set([
  "https://monster-coins-pro-seven.vercel.app",
  "https://monster-coins-pro-oemer-yildirim.vercel.app",
  "https://monster-coins-pro-git-brian-2026-oemer-yildirim.vercel.app",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);

function cors(origin?: string | null): Record<string, string> {
  const allowed = origin && (ALLOWED_EXACT.has(origin) || ALLOWED_ORIGIN.test(origin))
    ? origin
    : "https://monster-coins-pro-oemer-yildirim.vercel.app";
  return {
    "access-control-allow-origin": allowed,
    "access-control-allow-headers": "content-type,x-brian-dashboard-key",
    "access-control-allow-methods": "POST,OPTIONS",
    vary: "Origin",
  };
}
function out(body: unknown, status = 200, origin?: string | null) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...cors(origin) },
  });
}
async function sha256Hex(value: string) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
function constantTimeEqual(left: string, right: string) {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let index = 0; index < left.length; index++) diff |= left.charCodeAt(index) ^ right.charCodeAt(index);
  return diff === 0;
}
async function auth(req: Request) {
  const supplied = (req.headers.get("x-brian-dashboard-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_DASHBOARD");
  const q = await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id", AUTH_ID).single();
  if (q.error || !q.data) throw new Error("DASHBOARD_AUTH_UNAVAILABLE");
  if (!constantTimeEqual(await sha256Hex(supplied), String(q.data.dashboard_key_sha256 ?? ""))) {
    throw new Error("UNAUTHORIZED_DASHBOARD");
  }
}

type QueryResult = { data: unknown[] | null; error: { message?: string } | null; count?: number | null };
function requireOk(name: string, result: QueryResult) {
  if (result.error) throw new Error(`${name}: ${result.error.message ?? "query failed"}`);
  return result.data ?? [];
}

Deno.serve(async (req: Request) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
  if (req.method !== "POST") return out({ error: "POST required" }, 405, origin);
  try { await auth(req); } catch (error) { return out({ error: String(error) }, 401, origin); }

  try {
    const [
      codegenQ,
      candidatesQ,
      artifactsQ,
      hypothesesQ,
      sourcesQ,
      assessmentsQ,
      eventsQ,
      treasuryQ,
      actionsQ,
      codegenCountQ,
      candidatesCountQ,
      artifactsCountQ,
      sourcesCountQ,
    ] = await Promise.all([
      db.from("brian_evolution_codegen_requests")
        .select("request_id,candidate_id,hypothesis_id,requested_at,parent_commit,branch_name,changed_paths,objective,constraints,success_criteria,required_human_review,external_generator_required,autonomous_apply_allowed,metadata")
        .order("requested_at", { ascending: false }).limit(8),
      db.from("brian_evolution_code_candidates")
        .select("candidate_id,hypothesis_id,proposed_at,parent_commit,changed_paths,test_plan,test_results,replay_results,stress_results,prospective_results,contamination_declaration,stage,autonomous_apply_allowed,metadata")
        .order("proposed_at", { ascending: false }).limit(8),
      db.from("brian_evolution_code_artifact_receipts")
        .select("receipt_id,candidate_id,hypothesis_id,evidence_kind,observed_at,passed,artifact_sha256,parent_commit,branch_name,changed_paths,patch_bytes,generated_by,generator_run_id,provenance_complete,protected_scope_clear,leakage_detected,payload")
        .order("observed_at", { ascending: false }).limit(12),
      db.from("brian_evolution_hypothesis_snapshots")
        .select("hypothesis_id,observed_at,title,problem_statement,proposed_mechanism,target_capabilities,measurable_success_criteria,stage,uncertainty")
        .order("observed_at", { ascending: false }).limit(8),
      db.from("brian_world_source_candidates")
        .select("source_id,discovered_at,canonical_uri,provider,source_kind,authority_class,access_mode,stage,freshness_seconds,manipulation_risk,rationale")
        .order("discovered_at", { ascending: false }).limit(10),
      db.from("brian_world_source_assessments")
        .select("source_id,assessed_at,authority_score,freshness_score,manipulation_penalty,corroboration_penalty,access_penalty,trust_score,eligible_for_research,eligible_for_decision_evidence,reasons")
        .order("assessed_at", { ascending: false }).limit(10),
      db.from("brian_evolution_events")
        .select("event_id,entity_type,entity_id,event_type,occurred_at,stage,title,summary,evidence_refs")
        .order("occurred_at", { ascending: false }).limit(12),
      db.from("brian_treasury_shadow_snapshots")
        .select("snapshot_id,cycle_id,observed_at,starting_equity_usd,cash_usd,equity_usd,realized_pnl_usd,cumulative_costs_usd,deployment_usd,deployment_pct,cash_reserve_pct,positions,action_count,promotion_gate_open,promotion_gate_ref,promotion_gate_reason,blocked_reasons,treasury_version,gate_version")
        .order("observed_at", { ascending: false }).limit(1),
      db.from("brian_treasury_shadow_actions")
        .select("action_id,cycle_id,observed_at,kind,asset_id,direction,capital_usd,reference_price,cost_usd,expected_net_edge_bps,source_decision_id,reason,position_id")
        .order("observed_at", { ascending: false }).limit(20),
      db.from("brian_evolution_codegen_requests").select("request_id", { count: "exact", head: true }),
      db.from("brian_evolution_code_candidates").select("candidate_id", { count: "exact", head: true }),
      db.from("brian_evolution_code_artifact_receipts").select("receipt_id", { count: "exact", head: true }),
      db.from("brian_world_source_candidates").select("source_id", { count: "exact", head: true }),
    ]);

    const codegen = requireOk("codegen", codegenQ as QueryResult);
    const candidates = requireOk("candidates", candidatesQ as QueryResult);
    const artifacts = requireOk("artifacts", artifactsQ as QueryResult);
    const hypotheses = requireOk("hypotheses", hypothesesQ as QueryResult);
    const sources = requireOk("sources", sourcesQ as QueryResult);
    const assessments = requireOk("assessments", assessmentsQ as QueryResult);
    const events = requireOk("events", eventsQ as QueryResult);
    const treasuryRows = requireOk("treasury", treasuryQ as QueryResult);
    const actions = requireOk("treasury_actions", actionsQ as QueryResult);

    const assessmentBySource = new Map(
      assessments.map((row: any) => [String(row.source_id), row]),
    );
    const sourceLibrary = sources.map((row: any) => ({
      ...row,
      assessment: assessmentBySource.get(String(row.source_id)) ?? null,
    }));

    return out({
      status: "ONLINE",
      observed_at: new Date().toISOString(),
      governance: {
        autonomous_research: true,
        autonomous_candidate_codegen: true,
        autonomous_test_evidence: true,
        autonomous_canonical_apply: false,
        human_review_required: true,
        protected_scopes_remain_locked: true,
        dip_isolated: true,
      },
      summary: {
        codegen_requests: (codegenCountQ as any).count ?? codegen.length,
        code_candidates: (candidatesCountQ as any).count ?? candidates.length,
        artifact_receipts: (artifactsCountQ as any).count ?? artifacts.length,
        discovered_world_sources: (sourcesCountQ as any).count ?? sources.length,
        hypotheses_visible: hypotheses.length,
        treasury_actions_visible: actions.length,
      },
      codegen_requests: codegen,
      code_candidates: candidates,
      artifact_receipts: artifacts,
      hypotheses,
      source_library: sourceLibrary,
      evolution_events: events,
      treasury: treasuryRows[0] ?? null,
      treasury_actions: actions,
      shadow_only: true,
      live_execution: false,
    }, 200, origin);
  } catch (error) {
    return out({ status: "DEGRADED", error: String(error), shadow_only: true, live_execution: false }, 500, origin);
  }
});
