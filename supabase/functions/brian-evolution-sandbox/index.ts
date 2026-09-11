import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { EVOLUTION_EVIDENCE_CLASS } from "../_shared/evolution_contract.ts";
import type { HypothesisCandidate } from "../_shared/evolution_research.ts";
import {
  buildSandboxGenerationBrief,
  evaluateSandboxReview,
  EVOLUTION_SANDBOX_VERSION,
  type SandboxArtifactSubmission,
  type SandboxEvidenceKind,
  type SandboxReviewState,
  validateSandboxArtifact,
} from "../_shared/evolution_sandbox.ts";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const COLLECTOR_ID = "brian-evolution-sandbox-v1";
const LEASE_SECONDS = 180;
const DEFAULT_PARENT_COMMIT = "ba330f4ec4bd3d76b1fe5564d6b4e95cd41ef4f0";

function out(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

async function sha(value: string): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function metadata(row: Record<string, unknown>): Record<string, unknown> {
  return (row.metadata ?? {}) as Record<string, unknown>;
}

function toHypothesis(row: Record<string, unknown>): HypothesisCandidate {
  const meta = metadata(row);
  return {
    hypothesisId: String(row.hypothesis_id),
    observedAt: String(row.observed_at),
    problemStatement: String(row.problem_statement),
    proposedMechanism: String(row.proposed_mechanism),
    targetCapabilities: Array.isArray(row.target_capabilities) ? row.target_capabilities.map(String) : [],
    evidenceRefs: Array.isArray(row.evidence_refs) ? row.evidence_refs.map(String) : [],
    counterEvidenceRefs: Array.isArray(row.counter_evidence_refs) ? row.counter_evidence_refs.map(String) : [],
    measurableSuccessCriteria: Array.isArray(row.measurable_success_criteria) ? row.measurable_success_criteria.map(String) : [],
    stage: row.stage as HypothesisCandidate["stage"],
    uncertainty: Number(row.uncertainty ?? 0.5),
    priority: Number(meta.priority ?? 0.5),
    hypothesisKind: String(meta.hypothesis_kind ?? "CAPABILITY_GAP") as HypothesisCandidate["hypothesisKind"],
    metadata: meta,
  };
}

function parentCommit(): string {
  const value = (Deno.env.get("BRIAN_EVOLUTION_PARENT_COMMIT") ?? DEFAULT_PARENT_COMMIT).trim();
  return /^[0-9a-f]{7,64}$/i.test(value) ? value : DEFAULT_PARENT_COMMIT;
}

async function latestHypotheses(): Promise<HypothesisCandidate[]> {
  const q = await db.from("brian_evolution_hypothesis_snapshots")
    .select("hypothesis_id,observed_at,problem_statement,proposed_mechanism,target_capabilities,evidence_refs,counter_evidence_refs,measurable_success_criteria,stage,uncertainty,metadata,created_at")
    .in("stage", ["RESEARCHING", "EXPERIMENTAL"])
    .order("observed_at", { ascending: false })
    .limit(300);
  if (q.error) throw new Error(`hypotheses:${q.error.message}`);
  const seen = new Set<string>();
  const rows: HypothesisCandidate[] = [];
  for (const raw of q.data ?? []) {
    const h = toHypothesis(raw as Record<string, unknown>);
    if (!h.hypothesisId || seen.has(h.hypothesisId)) continue;
    seen.add(h.hypothesisId);
    rows.push(h);
  }
  return rows.sort((a, b) => b.priority - a.priority);
}

async function planCandidates(): Promise<{ planned: number; skippedExisting: number; candidateIds: string[] }> {
  const hypotheses = await latestHypotheses();
  if (!hypotheses.length) return { planned: 0, skippedExisting: 0, candidateIds: [] };
  const existingQ = await db.from("brian_evolution_codegen_requests").select("hypothesis_id,candidate_id").limit(2000);
  if (existingQ.error) throw new Error(`existing_codegen:${existingQ.error.message}`);
  const existingHypotheses = new Set((existingQ.data ?? []).map((row) => String(row.hypothesis_id)));
  const candidateRows: Record<string, unknown>[] = [];
  const requestRows: Record<string, unknown>[] = [];
  const candidateIds: string[] = [];
  let skippedExisting = 0;

  for (const h of hypotheses.slice(0, 20)) {
    if (existingHypotheses.has(h.hypothesisId)) {
      skippedExisting++;
      continue;
    }
    const brief = buildSandboxGenerationBrief(h, parentCommit());
    candidateIds.push(brief.candidateId);
    candidateRows.push({
      candidate_id: brief.candidateId,
      hypothesis_id: brief.hypothesisId,
      proposed_at: brief.proposedAt,
      parent_commit: brief.parentCommit,
      changed_paths: brief.changedPaths,
      test_plan: brief.testPlan,
      test_results: {},
      replay_results: {},
      stress_results: {},
      prospective_results: {},
      contamination_declaration: brief.contaminationDeclaration,
      stage: brief.stage,
      autonomous_apply_allowed: false,
      metadata: {
        ...brief.metadata,
        branch_name: brief.branchName,
        objective: brief.objective,
        constraints: brief.constraints,
        success_criteria: brief.successCriteria,
        required_human_review: true,
        artifact_state: "PLANNED",
      },
      evidence_class: EVOLUTION_EVIDENCE_CLASS,
      shadow_only: true,
      live_execution: false,
    });
    requestRows.push({
      request_id: await sha(`codegen-request|${brief.candidateId}|${brief.parentCommit}`),
      candidate_id: brief.candidateId,
      hypothesis_id: brief.hypothesisId,
      requested_at: brief.proposedAt,
      parent_commit: brief.parentCommit,
      branch_name: brief.branchName,
      changed_paths: brief.changedPaths,
      objective: brief.objective,
      constraints: brief.constraints,
      success_criteria: brief.successCriteria,
      evidence_refs: brief.evidenceRefs,
      contamination_declaration: brief.contaminationDeclaration,
      external_generator_required: true,
      required_human_review: true,
      metadata: {
        sandbox_version: EVOLUTION_SANDBOX_VERSION,
        test_plan: brief.testPlan,
        hypothesis_kind: h.hypothesisKind,
        priority: h.priority,
      },
      evidence_class: EVOLUTION_EVIDENCE_CLASS,
      shadow_only: true,
      live_execution: false,
      autonomous_apply_allowed: false,
    });
  }

  if (candidateRows.length) {
    const q = await db.from("brian_evolution_code_candidates")
      .upsert(candidateRows, { onConflict: "candidate_id", ignoreDuplicates: true });
    if (q.error) throw new Error(`code_candidates:${q.error.message}`);
  }
  if (requestRows.length) {
    const q = await db.from("brian_evolution_codegen_requests")
      .upsert(requestRows, { onConflict: "request_id", ignoreDuplicates: true });
    if (q.error) throw new Error(`codegen_requests:${q.error.message}`);
  }
  return { planned: requestRows.length, skippedExisting, candidateIds };
}

async function loadRequest(candidateId: string) {
  const q = await db.from("brian_evolution_codegen_requests")
    .select("candidate_id,hypothesis_id,parent_commit,branch_name,changed_paths,requested_at")
    .eq("candidate_id", candidateId)
    .order("requested_at", { ascending: false })
    .limit(1)
    .maybeSingle();
  if (q.error) throw new Error(`codegen_request:${q.error.message}`);
  if (!q.data) throw new Error("UNKNOWN_CANDIDATE");
  return q.data;
}

async function submitGeneratedArtifact(input: SandboxArtifactSubmission) {
  const request = await loadRequest(input.candidateId);
  if (String(request.hypothesis_id) !== input.hypothesisId) throw new Error("HYPOTHESIS_MISMATCH");
  if (String(request.parent_commit) !== input.parentCommit) throw new Error("PARENT_COMMIT_MISMATCH");
  if (String(request.branch_name) !== input.branchName) throw new Error("BRANCH_MISMATCH");
  const validation = validateSandboxArtifact(input);
  if (!validation.valid) return { status: "REJECTED", validation, canonical_mutation: false };
  const expectedPaths = new Set(Array.isArray(request.changed_paths) ? request.changed_paths.map(String) : []);
  const unexpected = validation.normalizedPaths.filter((path) => !expectedPaths.has(path));
  if (unexpected.length) {
    return { status: "REJECTED", validation: { ...validation, valid: false, reasons: [`unexpected path(s):${unexpected.join(",")}`] }, canonical_mutation: false };
  }
  const receiptId = await sha(`artifact|${input.candidateId}|${input.generatorRunId}|${input.patchSha256}`);
  const row = {
    receipt_id: receiptId,
    candidate_id: input.candidateId,
    hypothesis_id: input.hypothesisId,
    evidence_kind: "GENERATED",
    observed_at: input.generatedAt,
    passed: true,
    artifact_sha256: input.patchSha256,
    parent_commit: input.parentCommit,
    branch_name: input.branchName,
    changed_paths: validation.normalizedPaths,
    patch_bytes: input.patchBytes,
    generated_by: input.generatedBy,
    generator_run_id: input.generatorRunId,
    provenance_complete: true,
    protected_scope_clear: true,
    leakage_detected: false,
    evidence_refs: [],
    payload: { ...input.metadata, sandbox_validation: validation.reasons, sandbox_version: EVOLUTION_SANDBOX_VERSION },
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
    autonomous_apply_allowed: false,
  };
  const q = await db.from("brian_evolution_code_artifact_receipts")
    .upsert(row, { onConflict: "receipt_id", ignoreDuplicates: true });
  if (q.error) throw new Error(`artifact_receipt:${q.error.message}`);
  return { status: "ACCEPTED_FOR_TESTING", receipt_id: receiptId, validation, autonomous_apply_allowed: false, required_human_review: true };
}

async function recordEvidence(body: Record<string, unknown>) {
  const candidateId = String(body.candidate_id ?? "");
  const kind = String(body.evidence_kind ?? "") as SandboxEvidenceKind;
  if (!candidateId) throw new Error("candidate_id required");
  if (!["TYPECHECK", "UNIT", "REPLAY", "STRESS", "PROSPECTIVE"].includes(kind)) throw new Error("unsupported evidence_kind");
  const request = await loadRequest(candidateId);
  const observedAt = String(body.observed_at ?? new Date().toISOString());
  if (!Number.isFinite(Date.parse(observedAt))) throw new Error("invalid observed_at");
  const passed = body.passed === true;
  const leakageDetected = body.leakage_detected === true;
  const receiptId = await sha(`artifact-evidence|${candidateId}|${kind}|${observedAt}|${String(body.run_id ?? "")}|${passed}|${leakageDetected}`);
  const row = {
    receipt_id: receiptId,
    candidate_id: candidateId,
    hypothesis_id: String(request.hypothesis_id),
    evidence_kind: kind,
    observed_at: observedAt,
    passed: leakageDetected ? false : passed,
    artifact_sha256: body.artifact_sha256 == null ? null : String(body.artifact_sha256),
    parent_commit: String(request.parent_commit),
    branch_name: String(request.branch_name),
    changed_paths: Array.isArray(request.changed_paths) ? request.changed_paths : [],
    patch_bytes: null,
    generated_by: null,
    generator_run_id: body.run_id == null ? null : String(body.run_id),
    provenance_complete: body.provenance_complete !== false,
    protected_scope_clear: body.protected_scope_clear !== false,
    leakage_detected: leakageDetected,
    evidence_refs: Array.isArray(body.evidence_refs) ? body.evidence_refs.map(String) : [],
    payload: (body.payload ?? {}) as Record<string, unknown>,
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
    autonomous_apply_allowed: false,
  };
  const q = await db.from("brian_evolution_code_artifact_receipts")
    .upsert(row, { onConflict: "receipt_id", ignoreDuplicates: true });
  if (q.error) throw new Error(`evidence_receipt:${q.error.message}`);
  return { status: "RECORDED", receipt_id: receiptId, evidence_kind: kind, passed: row.passed, autonomous_apply_allowed: false };
}

async function reviewCandidate(candidateId: string) {
  const request = await loadRequest(candidateId);
  const q = await db.from("brian_evolution_latest_code_artifact_receipts")
    .select("receipt_id,evidence_kind,observed_at,passed,artifact_sha256,provenance_complete,protected_scope_clear,leakage_detected")
    .eq("candidate_id", candidateId);
  if (q.error) throw new Error(`artifact_evidence:${q.error.message}`);
  const byKind = new Map((q.data ?? []).map((row) => [String(row.evidence_kind), row]));
  const generated = byKind.get("GENERATED");
  const pass = (kind: string) => byKind.get(kind)?.passed === true;
  const state: SandboxReviewState = {
    protectedScopeClear: generated?.protected_scope_clear === true && [...byKind.values()].every((row) => row.protected_scope_clear !== false),
    artifactHashVerified: typeof generated?.artifact_sha256 === "string" && /^[0-9a-f]{64}$/i.test(String(generated.artifact_sha256)),
    provenanceComplete: generated?.provenance_complete === true && [...byKind.values()].every((row) => row.provenance_complete !== false),
    typecheckGreen: pass("TYPECHECK"),
    unitGreen: pass("UNIT"),
    replayGreen: pass("REPLAY"),
    stressGreen: pass("STRESS"),
    prospectiveGreen: pass("PROSPECTIVE"),
    leakageClear: [...byKind.values()].every((row) => row.leakage_detected !== true),
  };
  const decision = evaluateSandboxReview(state);
  const reviewedAt = new Date().toISOString();
  const receiptId = await sha(`code-review|${candidateId}|${reviewedAt}|${decision.verdict}|${decision.reasons.join("|")}`);
  const receipt = {
    receipt_id: receiptId,
    candidate_id: candidateId,
    reviewed_at: reviewedAt,
    protected_scope_clear: state.protectedScopeClear,
    tests_green: state.typecheckGreen && state.unitGreen,
    replay_green: state.replayGreen,
    stress_green: state.stressGreen,
    prospective_green: state.prospectiveGreen,
    leakage_clear: state.leakageClear,
    reviewer_kind: "AUTOMATED_GUARD",
    verdict: decision.verdict,
    reasons: decision.reasons,
    metadata: {
      sandbox_version: EVOLUTION_SANDBOX_VERSION,
      artifact_hash_verified: state.artifactHashVerified,
      provenance_complete: state.provenanceComplete,
      evidence_receipt_ids: [...byKind.values()].map((row) => row.receipt_id),
      required_human_review: true,
      branch_name: request.branch_name,
    },
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
    autonomous_apply_allowed: false,
  };
  const ins = await db.from("brian_evolution_code_review_receipts").insert(receipt);
  if (ins.error) throw new Error(`review_receipt:${ins.error.message}`);
  return { status: "REVIEWED", candidate_id: candidateId, ...decision, state, receipt_id: receiptId };
}

async function recordCollectorRun(startedAt: string, status: "SUCCESS" | "FAILED" | "SKIPPED", observed: number, stored: number, error?: unknown) {
  const finishedAt = new Date().toISOString();
  const runId = await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q = await db.from("brian_collector_runs").insert({
    run_id: runId,
    collector_id: COLLECTOR_ID,
    started_at: startedAt,
    finished_at: finishedAt,
    status,
    observed_records: observed,
    stored_records: stored,
    degraded_sources: [],
    error_class: error ? "EVOLUTION_SANDBOX_ERROR" : null,
    error_message: error ? String(error).slice(0, 1200) : null,
    metadata: { sandbox_version: EVOLUTION_SANDBOX_VERSION, canonical_mutation: false, autonomous_apply_allowed: false },
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
  });
  if (q.error) console.error("sandbox collector receipt", q.error.message);
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ error: "POST required" }, 405);
  const startedAt = new Date().toISOString();
  try {
    await requireCronAuth(req, db);
  } catch (error) {
    return out({ status: "UNAUTHORIZED", error: String(error), shadow_only: true, live_execution: false }, 401);
  }

  let body: Record<string, unknown> = {};
  try {
    body = await req.json() as Record<string, unknown>;
  } catch {
    body = {};
  }
  const action = String(body.action ?? "plan").toLowerCase();

  try {
    if (action === "submit_artifact") {
      const result = await submitGeneratedArtifact((body.artifact ?? {}) as SandboxArtifactSubmission);
      return out({ ...result, shadow_only: true, live_execution: false });
    }
    if (action === "record_evidence") {
      const result = await recordEvidence(body);
      return out({ ...result, shadow_only: true, live_execution: false });
    }
    if (action === "review") {
      const candidateId = String(body.candidate_id ?? "");
      if (!candidateId) return out({ error: "candidate_id required" }, 400);
      const result = await reviewCandidate(candidateId);
      return out({ ...result, shadow_only: true, live_execution: false });
    }
    if (action !== "plan") return out({ error: `unsupported action:${action}` }, 400);

    const lease = await withCollectorLease(db, COLLECTOR_ID, LEASE_SECONDS, async () => {
      const result = await planCandidates();
      await recordCollectorRun(startedAt, "SUCCESS", result.planned + result.skippedExisting, result.planned * 2);
      return {
        status: "SUCCESS",
        collector_id: COLLECTOR_ID,
        sandbox_version: EVOLUTION_SANDBOX_VERSION,
        ...result,
        external_generator_required: true,
        required_human_review: true,
        canonical_mutation: false,
        autonomous_apply_allowed: false,
        cloud_independent: true,
        shadow_only: true,
        live_execution: false,
      };
    });
    if (lease.contended) {
      await recordCollectorRun(startedAt, "SKIPPED", 0, 0);
      return out({ status: "SKIPPED_LEASE_CONTENDED", shadow_only: true, live_execution: false });
    }
    return out(lease.value);
  } catch (error) {
    if (action === "plan") await recordCollectorRun(startedAt, "FAILED", 0, 0, error);
    return out({ status: "FAILED", error: String(error), canonical_mutation: false, autonomous_apply_allowed: false, shadow_only: true, live_execution: false }, 500);
  }
});
