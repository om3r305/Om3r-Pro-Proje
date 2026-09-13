import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const ISSUER = "https://token.actions.githubusercontent.com";
const AUDIENCE = "brian-evolution-engineering-v1";
const REPOSITORY = "om3r305/Om3r-Pro-Proje";
const BASE_REF = "refs/heads/brian-2026";
const ENGINEER_WORKFLOW = ".github/workflows/brian-engineer.yml";
const RELEASE_WORKFLOW = ".github/workflows/brian-engineer-release.yml";
const ALLOWED_WORKFLOWS = new Set([ENGINEER_WORKFLOW, RELEASE_WORKFLOW]);

type Claims = Record<string, unknown>;
let jwksCache: { expires: number; keys: JsonWebKey[] } | null = null;

function out(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}
function b64url(value: string): Uint8Array {
  const normalized = value.replaceAll("-", "+").replaceAll("_", "/") + "=".repeat((4 - value.length % 4) % 4);
  return Uint8Array.from(atob(normalized), (c) => c.charCodeAt(0));
}
function jsonPart(value: string): Record<string, unknown> {
  return JSON.parse(new TextDecoder().decode(b64url(value))) as Record<string, unknown>;
}
function audienceOk(value: unknown): boolean {
  return typeof value === "string" ? value === AUDIENCE : Array.isArray(value) && value.map(String).includes(AUDIENCE);
}
async function jwks(): Promise<JsonWebKey[]> {
  if (jwksCache && jwksCache.expires > Date.now()) return jwksCache.keys;
  const response = await fetch(`${ISSUER}/.well-known/jwks`, { headers: { accept: "application/json" } });
  if (!response.ok) throw new Error(`OIDC_JWKS_HTTP_${response.status}`);
  const body = await response.json() as { keys?: JsonWebKey[] };
  if (!Array.isArray(body.keys) || !body.keys.length) throw new Error("OIDC_JWKS_EMPTY");
  jwksCache = { expires: Date.now() + 60 * 60_000, keys: body.keys };
  return body.keys;
}
async function verifyGithubOidc(req: Request): Promise<Claims> {
  const auth = req.headers.get("authorization") ?? "";
  if (!auth.startsWith("Bearer ")) throw new Error("OIDC_BEARER_REQUIRED");
  const token = auth.slice(7).trim();
  const parts = token.split(".");
  if (parts.length !== 3) throw new Error("OIDC_FORMAT_INVALID");
  const header = jsonPart(parts[0]);
  const claims = jsonPart(parts[1]);
  if (header.alg !== "RS256" || typeof header.kid !== "string") throw new Error("OIDC_HEADER_INVALID");
  const key = (await jwks()).find((candidate) => candidate.kid === header.kid);
  if (!key) throw new Error("OIDC_KID_UNKNOWN");
  const cryptoKey = await crypto.subtle.importKey("jwk", key, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]);
  const signed = new TextEncoder().encode(`${parts[0]}.${parts[1]}`);
  const verified = await crypto.subtle.verify("RSASSA-PKCS1-v1_5", cryptoKey, b64url(parts[2]), signed);
  if (!verified) throw new Error("OIDC_SIGNATURE_INVALID");

  const now = Math.floor(Date.now() / 1000);
  if (claims.iss !== ISSUER || !audienceOk(claims.aud)) throw new Error("OIDC_ISSUER_OR_AUDIENCE_INVALID");
  if (Number(claims.exp ?? 0) < now - 15 || Number(claims.nbf ?? 0) > now + 15) throw new Error("OIDC_TIME_INVALID");
  if (claims.repository !== REPOSITORY) throw new Error("OIDC_REPOSITORY_DENIED");
  const workflowRef = String(claims.job_workflow_ref ?? "");
  const workflowPath = workflowRef.startsWith(`${REPOSITORY}/`) ? workflowRef.slice(REPOSITORY.length + 1).split("@")[0] : "";
  if (!ALLOWED_WORKFLOWS.has(workflowPath)) throw new Error("OIDC_WORKFLOW_DENIED");
  return { ...claims, _workflow_path: workflowPath };
}
function cleanPayload(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const text = JSON.stringify(value);
  if (text.length > 60_000) throw new Error("PAYLOAD_TOO_LARGE");
  return value as Record<string, unknown>;
}
function requireEngineerControl(claims: Claims) {
  if (claims._workflow_path !== ENGINEER_WORKFLOW) throw new Error("ENGINEER_WORKFLOW_REQUIRED");
  if (!new Set(["schedule", "workflow_dispatch"]).has(String(claims.event_name ?? ""))) throw new Error("ENGINEER_EVENT_DENIED");
  if (claims.ref !== BASE_REF) throw new Error("ENGINEER_REF_DENIED");
}
function requireReleaseControl(claims: Claims) {
  if (claims._workflow_path !== RELEASE_WORKFLOW) throw new Error("RELEASE_WORKFLOW_REQUIRED");
  if (claims.event_name !== "pull_request_review") throw new Error("RELEASE_EVENT_DENIED");
}
async function record(runId: string, eventKind: string, phase: string, passed: boolean | null, commitSha: string | null, payload: Record<string, unknown>) {
  const q = await db.rpc("record_engineering_event", {
    p_run_id: runId,
    p_event_kind: eventKind,
    p_phase: phase,
    p_passed: passed,
    p_commit_sha: commitSha,
    p_payload: payload,
  });
  if (q.error) throw new Error(`EVENT:${q.error.message}`);
}
async function refreshMeasurements() {
  const runs = await db.from("brian_evolution_engineering_runs")
    .select("run_id,candidate_id,branch_name,commit_sha,claimed_at")
    .eq("phase", "MEASURE").eq("status", "RUNNING").order("claimed_at", { ascending: true }).limit(20);
  if (runs.error) throw new Error(`MEASURE_RUNS:${runs.error.message}`);
  const promoted: string[] = [];
  for (const run of runs.data ?? []) {
    const evidence = await db.from("brian_evolution_code_artifact_receipts")
      .select("receipt_id,observed_at,evidence_kind,passed,branch_name,artifact_sha256,provenance_complete,protected_scope_clear,payload")
      .eq("candidate_id", run.candidate_id).eq("evidence_kind", "PROSPECTIVE").eq("passed", true)
      .eq("branch_name", run.branch_name).gte("observed_at", run.claimed_at)
      .order("observed_at", { ascending: false }).limit(1).maybeSingle();
    if (evidence.error) throw new Error(`MEASURE_EVIDENCE:${evidence.error.message}`);
    if (!evidence.data || evidence.data.provenance_complete !== true || evidence.data.protected_scope_clear !== true) continue;
    await record(String(run.run_id), "PROSPECTIVE", "MEASURE", true, run.commit_sha ? String(run.commit_sha) : null, { receipt: evidence.data });
    const update = await db.from("brian_evolution_engineering_runs").update({
      phase: "HUMAN_APPROVAL",
      status: "WAITING",
      measurement_passed: true,
      measurement_result: evidence.data,
      updated_at: new Date().toISOString(),
    }).eq("run_id", run.run_id).eq("phase", "MEASURE");
    if (update.error) throw new Error(`MEASURE_PROMOTE:${update.error.message}`);
    await db.from("brian_evolution_engineering_events").insert({
      run_id: run.run_id, event_kind: "HUMAN_APPROVAL_REQUIRED", phase: "HUMAN_APPROVAL", passed: null,
      commit_sha: run.commit_sha, payload: { prospective_receipt_id: evidence.data.receipt_id }, shadow_only: true, live_execution: false,
    });
    promoted.push(String(run.run_id));
  }
  return promoted;
}
async function runByBranch(branchName: string) {
  if (!branchName.startsWith("brian-engineer/")) throw new Error("ENGINEERING_BRANCH_INVALID");
  const q = await db.from("brian_evolution_engineering_runs").select("*").eq("branch_name", branchName).maybeSingle();
  if (q.error || !q.data) throw new Error(`ENGINEERING_RUN_NOT_FOUND:${q.error?.message ?? branchName}`);
  return q.data as Record<string, unknown>;
}
function releaseReady(run: Record<string, unknown>) {
  return run.phase === "HUMAN_APPROVAL" && run.status === "WAITING" &&
    run.compile_passed === true && run.tests_passed === true && run.replay_passed === true && run.stress_passed === true &&
    run.review_passed === true && run.preview_passed === true && run.measurement_passed === true;
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ status: "METHOD_NOT_ALLOWED" }, 405);
  try {
    const claims = await verifyGithubOidc(req);
    const body = cleanPayload(await req.json().catch(() => ({})));
    const action = String(body.action ?? "");

    if (action === "measure_pending") {
      requireEngineerControl(claims);
      const promoted = await refreshMeasurements();
      return out({ status: "OK", promoted_runs: promoted });
    }

    if (action === "claim") {
      requireEngineerControl(claims);
      const baseSha = String(body.base_sha ?? "");
      const requestId = body.request_id == null || String(body.request_id).trim() === "" ? null : String(body.request_id);
      const workerId = `github-actions:${String(claims.run_id ?? "unknown")}:${String(claims.run_attempt ?? "1")}`;
      const q = await db.rpc("claim_engineering_task", { p_worker_id: workerId, p_base_sha: baseSha, p_request_id: requestId });
      if (q.error) throw new Error(`CLAIM:${q.error.message}`);
      return out({ status: q.data ? "CLAIMED" : "NO_TASK", claim: q.data, identity: { run_id: claims.run_id, event_name: claims.event_name } });
    }

    if (action === "event") {
      const runId = String(body.run_id ?? "");
      const phase = String(body.phase ?? "");
      const eventKind = String(body.event_kind ?? "");
      const commitSha = body.commit_sha == null ? null : String(body.commit_sha);
      const passed = body.passed == null ? null : Boolean(body.passed);
      const payload = cleanPayload(body.payload);
      if (!/^[0-9a-f-]{36}$/i.test(runId)) throw new Error("RUN_ID_INVALID");
      if (!eventKind || eventKind.length > 120) throw new Error("EVENT_KIND_INVALID");
      await record(runId, eventKind, phase, passed, commitSha, payload);
      return out({ status: "RECORDED", run_id: runId, phase, event_kind: eventKind });
    }

    if (action === "human_approve") {
      requireReleaseControl(claims);
      const branchName = String(body.branch_name ?? "");
      const headSha = String(body.head_sha ?? "");
      const run = await runByBranch(branchName);
      if (!releaseReady(run)) return out({ status: "BLOCKED", reason: "engineering evidence gates are incomplete", run_id: run.run_id }, 409);
      if (run.commit_sha !== headSha) return out({ status: "BLOCKED", reason: "approved head SHA does not match measured commit", run_id: run.run_id }, 409);
      const actor = String(claims.actor ?? "");
      if (!actor || /\[bot\]$/i.test(actor)) throw new Error("HUMAN_ACTOR_REQUIRED");
      const q = await db.from("brian_evolution_engineering_runs").update({
        human_approval_status: "APPROVED",
        human_approved_by: actor,
        human_approved_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }).eq("run_id", run.run_id).eq("human_approval_status", "PENDING");
      if (q.error) throw new Error(`HUMAN_APPROVAL:${q.error.message}`);
      await db.from("brian_evolution_engineering_events").insert({
        run_id: run.run_id, event_kind: "HUMAN_APPROVED", phase: "HUMAN_APPROVAL", passed: true,
        commit_sha: headSha, payload: { actor }, shadow_only: true, live_execution: false,
      });
      return out({ status: "APPROVED", run_id: run.run_id, actor });
    }

    if (action === "release_gate") {
      requireReleaseControl(claims);
      const run = await runByBranch(String(body.branch_name ?? ""));
      const ready = releaseReady(run) && run.human_approval_status === "APPROVED";
      return out({ status: ready ? "READY" : "BLOCKED", ready, run });
    }

    return out({ status: "UNKNOWN_ACTION" }, 400);
  } catch (error) {
    console.error("brian-evolution-engineering-gateway", String(error));
    return out({ status: "FAILED_CLOSED", error: String(error) }, 401);
  }
});
