import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, {
  auth: { persistSession: false, autoRefreshToken: false },
});

const AUTH_ID = "control-v3";
const ALLOWED_ORIGIN = /^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const ALLOWED_EXACT = new Set([
  "https://monster-coins-pro-seven.vercel.app",
  "https://monster-coins-pro-oemer-yildirim.vercel.app",
  "https://monster-coins-pro-git-brian-2026-oemer-yildirim.vercel.app",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);

const PIPELINE = [
  "CLAIMED",
  "UNDERSTAND",
  "PLAN",
  "CODE",
  "COMPILE",
  "TEST",
  "REPLAY",
  "STRESS",
  "REVIEW",
  "PR",
  "PREVIEW",
  "MEASURE",
  "HUMAN_APPROVAL",
  "DEPLOY",
  "MONITOR",
  "COMPLETE",
];

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
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...cors(origin),
    },
  });
}

async function sha256Hex(value: string) {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)),
  );
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function constantTimeEqual(left: string, right: string) {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let index = 0; index < left.length; index++) {
    diff |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return diff === 0;
}

async function auth(req: Request) {
  const supplied = (req.headers.get("x-brian-dashboard-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_DASHBOARD");
  const q = await db.from("brian_dashboard_auth")
    .select("dashboard_key_sha256")
    .eq("auth_id", AUTH_ID)
    .single();
  if (q.error || !q.data) throw new Error("DASHBOARD_AUTH_UNAVAILABLE");
  if (
    !constantTimeEqual(
      await sha256Hex(supplied),
      String(q.data.dashboard_key_sha256 ?? ""),
    )
  ) throw new Error("UNAUTHORIZED_DASHBOARD");
}

function priority(row: any) {
  const value = Number(row?.metadata?.priority ?? 0);
  return Number.isFinite(value) ? value : 0;
}

function timeMs(value: unknown) {
  const ms = Date.parse(String(value ?? ""));
  return Number.isFinite(ms) ? ms : 0;
}

function compareNewest(a: any, b: any) {
  return timeMs(b.requested_at) - timeMs(a.requested_at) ||
    timeMs(b.created_at) - timeMs(a.created_at) ||
    String(b.request_id ?? "").localeCompare(String(a.request_id ?? ""));
}

Deno.serve(async (req: Request) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: cors(origin) });
  }
  if (req.method !== "POST") return out({ error: "POST required" }, 405, origin);
  try {
    await auth(req);
  } catch (error) {
    return out({ error: String(error) }, 401, origin);
  }

  try {
    const [controlQ, runsQ, eventsQ, requestsQ] = await Promise.all([
      db.from("brian_evolution_engineering_control")
        .select("control_id,autonomous_claim_enabled,base_branch,max_concurrent_runs,require_human_approval,monitor_minutes,updated_at,metadata")
        .eq("control_id", "default")
        .single(),
      db.from("brian_evolution_engineering_runs")
        .select("run_id,request_id,candidate_id,hypothesis_id,worker_id,phase,status,base_branch,base_sha,source_parent_sha,branch_name,commit_sha,pr_number,pr_url,preview_url,previous_good_sha,deployed_sha,rollback_sha,compile_passed,tests_passed,replay_passed,stress_passed,review_passed,preview_passed,measurement_passed,human_approval_status,human_approved_by,human_approved_at,monitor_status,failure_reason,shadow_only,live_execution,autonomous_apply_allowed,claimed_at,created_at,updated_at")
        .order("created_at", { ascending: false })
        .limit(200),
      db.from("brian_evolution_engineering_events")
        .select("event_id,run_id,observed_at,event_kind,phase,passed,commit_sha,evidence_class,shadow_only,live_execution,payload")
        .order("observed_at", { ascending: false })
        .limit(500),
      db.from("brian_evolution_codegen_requests")
        .select("request_id,candidate_id,hypothesis_id,requested_at,created_at,parent_commit,branch_name,changed_paths,objective,constraints,success_criteria,required_human_review,external_generator_required,shadow_only,live_execution,autonomous_apply_allowed,metadata")
        .order("requested_at", { ascending: false })
        .limit(500),
    ]);

    if (controlQ.error || !controlQ.data) {
      throw new Error(`control: ${controlQ.error?.message ?? "missing"}`);
    }
    if (runsQ.error) throw new Error(`runs: ${runsQ.error.message}`);
    if (eventsQ.error) throw new Error(`events: ${eventsQ.error.message}`);
    if (requestsQ.error) throw new Error(`requests: ${requestsQ.error.message}`);

    const runs = runsQ.data ?? [];
    const events = eventsQ.data ?? [];
    const requests = requestsQ.data ?? [];
    const requestById = new Map(requests.map((row: any) => [String(row.request_id), row]));
    const runRequestIds = new Set(runs.map((row: any) => String(row.request_id)));

    const newestByHypothesis = new Map<string, any>();
    for (const row of [...requests].sort(compareNewest)) {
      const key = String(row.hypothesis_id ?? row.request_id ?? "");
      if (!newestByHypothesis.has(key)) newestByHypothesis.set(key, row);
    }

    const eligibleQueue = [...newestByHypothesis.values()]
      .filter((row: any) =>
        !runRequestIds.has(String(row.request_id)) &&
        row.required_human_review === true &&
        row.shadow_only === true &&
        row.live_execution === false &&
        row.autonomous_apply_allowed === false
      )
      .sort((a: any, b: any) =>
        priority(b) - priority(a) || compareNewest(a, b)
      );

    const eventsByRun = new Map<string, any[]>();
    for (const event of events) {
      const key = String(event.run_id);
      const list = eventsByRun.get(key) ?? [];
      if (list.length < 24) list.push(event);
      eventsByRun.set(key, list);
    }

    const enrichedRuns = runs.slice(0, 30).map((run: any) => {
      const request = requestById.get(String(run.request_id));
      return {
        ...run,
        objective: request?.objective ?? null,
        requested_changed_paths: request?.changed_paths ?? [],
        request_priority: priority(request),
        recent_events: eventsByRun.get(String(run.run_id)) ?? [],
      };
    });

    const summary = {
      runs_total: runs.length,
      blocked_runs: runs.filter((r: any) => r.status === "BLOCKED").length,
      active_runs: runs.filter((r: any) =>
        ["RUNNING", "WAITING"].includes(String(r.status)) &&
        !["HUMAN_APPROVAL", "COMPLETE", "BLOCKED", "ROLLBACK"].includes(String(r.phase))
      ).length,
      review_passed_runs: runs.filter((r: any) => r.review_passed === true).length,
      waiting_human_approval: runs.filter((r: any) =>
        r.phase === "HUMAN_APPROVAL" && r.status === "WAITING"
      ).length,
      completed_runs: runs.filter((r: any) => r.phase === "COMPLETE" && r.status === "COMPLETE").length,
      eligible_queue: eligibleQueue.length,
    };

    return out({
      status: "ONLINE",
      observed_at: new Date().toISOString(),
      control: controlQ.data,
      summary,
      pipeline: PIPELINE,
      current_run: enrichedRuns.find((r: any) =>
        ["RUNNING", "WAITING"].includes(String(r.status)) && r.phase !== "HUMAN_APPROVAL"
      ) ?? null,
      approval_run: enrichedRuns.find((r: any) =>
        r.phase === "HUMAN_APPROVAL" && r.status === "WAITING"
      ) ?? null,
      recent_runs: enrichedRuns,
      eligible_queue: eligibleQueue.slice(0, 12),
      shadow_only: true,
      live_execution: false,
      dip_isolated: true,
    }, 200, origin);
  } catch (error) {
    return out({
      status: "DEGRADED",
      error: String(error),
      shadow_only: true,
      live_execution: false,
      dip_isolated: true,
    }, 500, origin);
  }
});
