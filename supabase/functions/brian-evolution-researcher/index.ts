import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { ALPHA_COMPILER_VERSION } from "../_shared/alpha_decision.ts";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { EVOLUTION_ALPHA_INTELLIGENCE_VERSION } from "../_shared/evolution_alpha_intelligence.ts";
import { EVOLUTION_EVIDENCE_CLASS } from "../_shared/evolution_contract.ts";
import {
  buildExperimentPlan,
  EVOLUTION_RESEARCH_VERSION,
  generateResearchHypotheses,
  type ChallengerSignal,
  type GapSignal,
  type HypothesisCandidate,
  type OutcomeSignal,
  type ReliabilitySignal,
  type ResearchInputs,
} from "../_shared/evolution_research.ts";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const COLLECTOR_ID = "brian-evolution-researcher-v1";
const LEASE_SECONDS = 240;
const ACTION_GATE_CHALLENGER_VERSION = "brian-alpha-calibration-challenger-v1";

function out(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}
async function sha(value: string): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}
const finite = (v: unknown, fallback = 0) => Number.isFinite(Number(v)) ? Number(v) : fallback;
const nullableFinite = (v: unknown): number | null => {
  if (v == null || v === "") return null;
  const parsed = Number(v);
  return Number.isFinite(parsed) ? parsed : null;
};
const avg = (values: number[]): number | null => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
const challengerVersionFor = (kind: HypothesisCandidate["hypothesisKind"]): string => {
  if (kind === "ACTION_GATE") return ACTION_GATE_CHALLENGER_VERSION;
  if (kind === "EXPECTED_EDGE" || kind === "RELIABILITY_FEEDBACK" || kind === "COST_CONTROL") return EVOLUTION_ALPHA_INTELLIGENCE_VERSION;
  return `${EVOLUTION_RESEARCH_VERSION}:${kind.toLowerCase()}`;
};

async function loadGaps(): Promise<GapSignal[]> {
  const q = await db.from("brian_evolution_latest_gaps")
    .select("gap_id,capability_id,severity,reason,suggested_action,evidence_refs")
    .in("severity", ["CRITICAL", "HIGH"])
    .limit(100);
  if (q.error) throw new Error(`gaps:${q.error.message}`);
  return (q.data ?? []).map((row) => ({
    gapId: String(row.gap_id), capabilityId: String(row.capability_id), severity: row.severity as GapSignal["severity"],
    reason: String(row.reason), suggestedAction: String(row.suggested_action),
    evidenceRefs: Array.isArray(row.evidence_refs) ? row.evidence_refs.map(String) : [],
  }));
}

async function loadChallenger(observedAt: string): Promise<ChallengerSignal | null> {
  const since = new Date(Date.parse(observedAt) - 24 * 3600_000).toISOString();
  const q = await db.from("brian_alpha_calibration_challenger")
    .select("challenger_id,challenger_action,avg_support_cost_adjusted_bps,evaluated_at")
    .gte("evaluated_at", since).order("evaluated_at", { ascending: false }).limit(10000);
  if (q.error) throw new Error(`challenger:${q.error.message}`);
  const rows = q.data ?? [];
  if (!rows.length) return null;
  const count = (name: string) => rows.filter((r) => String(r.challenger_action) === name).length;
  const allowCosts = rows.filter((r) => String(r.challenger_action) === "ALLOW_ACTION")
    .map((r) => nullableFinite(r.avg_support_cost_adjusted_bps)).filter((n): n is number => n != null);
  const downCosts = rows.filter((r) => String(r.challenger_action) === "DOWNGRADE_TO_WAIT")
    .map((r) => nullableFinite(r.avg_support_cost_adjusted_bps)).filter((n): n is number => n != null);
  return {
    allowAction: count("ALLOW_ACTION"), downgradeToWait: count("DOWNGRADE_TO_WAIT"), keepWait: count("KEEP_WAIT"),
    allowAvgCostAdjustedBps: avg(allowCosts), downgradeAvgCostAdjustedBps: avg(downCosts), observedAt,
    evidenceRefs: [`brian_alpha_calibration_challenger:24h:${rows.length}`],
  };
}

async function loadOutcomes(observedAt: string): Promise<OutcomeSignal[]> {
  const since = new Date(Date.parse(observedAt) - 7 * 24 * 3600_000).toISOString();
  const q = await db.from("brian_alpha_decision_outcomes")
    .select("outcome_id,horizon_seconds,direction_adjusted_return,classification,resolved_at,metadata")
    .gte("resolved_at", since).order("resolved_at", { ascending: false }).limit(12000);
  if (q.error) throw new Error(`outcomes:${q.error.message}`);
  const openRows = (q.data ?? []).filter((row) => {
    const action = String((row.metadata as Record<string, unknown> | null)?.original_action ?? "");
    return action === "OPEN_LONG" || action === "OPEN_SHORT";
  });
  const horizons = [...new Set(openRows.map((r) => Math.trunc(finite(r.horizon_seconds))).filter((n) => n > 0))].sort((a, b) => a - b);
  return horizons.map((horizon) => {
    const rows = openRows.filter((r) => Math.trunc(finite(r.horizon_seconds)) === horizon);
    const valid = rows.flatMap((row) => {
      const directionAdjustedReturn = nullableFinite(row.direction_adjusted_return);
      const metadata = (row.metadata ?? {}) as Record<string, unknown>;
      const cost = nullableFinite(metadata.estimated_round_trip_cost_bps);
      if (directionAdjustedReturn == null || cost == null || cost < 0) return [];
      return [{
        grossBps: directionAdjustedReturn * 10_000,
        afterCostBps: directionAdjustedReturn * 10_000 - cost,
        favorable: String(row.classification) === "ACTION_FAVORABLE_AFTER_COST",
      }];
    });
    const directional = valid.map((r) => r.grossBps);
    const afterCost = valid.map((r) => r.afterCostBps);
    const positive = directional.filter((n) => n > 0).length;
    const favorable = valid.filter((r) => r.favorable).length;
    return {
      horizonSeconds: horizon, samples: valid.length,
      grossPositiveRate: valid.length ? positive / valid.length : null,
      avgDirectionBps: avg(directional), avgAfterCostBps: avg(afterCost),
      favorableAfterCostRate: valid.length ? favorable / valid.length : null,
      observedAt, evidenceRefs: [`brian_alpha_decision_outcomes:${horizon}s:cost-complete:${valid.length}`],
    };
  });
}

async function loadReliability(observedAt: string): Promise<ReliabilitySignal[]> {
  const latestQ = await db.from("brian_sensor_reliability_shadow_snapshots")
    .select("window_end").order("window_end", { ascending: false }).limit(1).maybeSingle();
  if (latestQ.error) throw new Error(`reliability_window:${latestQ.error.message}`);
  const windowEnd = latestQ.data?.window_end ? String(latestQ.data.window_end) : null;
  if (!windowEnd) return [];
  const [measuredQ, canonicalQ] = await Promise.all([
    db.from("brian_sensor_reliability_shadow_snapshots")
      .select("independent_group,sample_count,bayesian_hit_rate_beta10_10,avg_cost_adjusted_signed_bps,outcome_horizon_seconds")
      .eq("window_end", windowEnd).order("sample_count", { ascending: false }).limit(1000),
    db.from("brian_sensor_observations")
      .select("independent_group,reliability,observed_at")
      .gte("observed_at", new Date(Date.parse(observedAt) - 6 * 3600_000).toISOString())
      .order("observed_at", { ascending: false }).limit(10000),
  ]);
  if (measuredQ.error) throw new Error(`measured_reliability:${measuredQ.error.message}`);
  if (canonicalQ.error) throw new Error(`canonical_reliability:${canonicalQ.error.message}`);
  const canonical = new Map<string, number[]>();
  for (const row of canonicalQ.data ?? []) {
    const group = String(row.independent_group); const bucket = canonical.get(group) ?? [];
    const reliability = nullableFinite(row.reliability);
    if (reliability != null && reliability >= 0 && reliability <= 1) bucket.push(reliability);
    canonical.set(group, bucket);
  }
  const best = new Map<string, Record<string, unknown>>();
  for (const row of measuredQ.data ?? []) {
    const group = String(row.independent_group); const current = best.get(group);
    if (!current || finite(row.sample_count) > finite(current.sample_count)) best.set(group, row as Record<string, unknown>);
  }
  return [...best.entries()].map(([group, row]) => ({
    sensorFamily: group, samples: Math.trunc(finite(row.sample_count)),
    canonicalReliability: avg(canonical.get(group) ?? []), measuredScore: nullableFinite(row.bayesian_hit_rate_beta10_10),
    avgCostAdjustedBps: nullableFinite(row.avg_cost_adjusted_signed_bps), observedAt,
    evidenceRefs: [`brian_sensor_reliability_shadow_snapshots:${group}:${windowEnd}`],
  }));
}

async function persistHypotheses(inputs: ResearchInputs): Promise<{ hypotheses: number; experiments: number }> {
  const hypotheses = generateResearchHypotheses(inputs).slice(0, 30);
  const hypothesisRows: Record<string, unknown>[] = [];
  const experimentRows: Record<string, unknown>[] = [];
  for (const h of hypotheses) {
    const snapshotId = await sha(`${h.hypothesisId}|${h.observedAt}|${EVOLUTION_RESEARCH_VERSION}`);
    hypothesisRows.push({
      snapshot_id: snapshotId, hypothesis_id: h.hypothesisId, created_at_source: h.observedAt, observed_at: h.observedAt,
      title: `${h.hypothesisKind}: ${h.targetCapabilities.join(", ")}`.slice(0, 240), problem_statement: h.problemStatement,
      proposed_mechanism: h.proposedMechanism, target_capabilities: h.targetCapabilities, evidence_refs: h.evidenceRefs,
      counter_evidence_refs: h.counterEvidenceRefs, measurable_success_criteria: h.measurableSuccessCriteria,
      stage: h.stage, uncertainty: h.uncertainty, metadata: { ...h.metadata, priority: h.priority, hypothesis_kind: h.hypothesisKind, research_version: EVOLUTION_RESEARCH_VERSION },
      evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true, live_execution: false,
    });
    const challengerVersion = challengerVersionFor(h.hypothesisKind);
    const plan = buildExperimentPlan(h, ALPHA_COMPILER_VERSION, challengerVersion);
    experimentRows.push({
      experiment_id: plan.experimentId, hypothesis_id: plan.hypothesisId, created_at_source: plan.createdAt,
      control_version: plan.controlVersion, challenger_version: plan.challengerVersion, mode: plan.mode,
      minimum_samples: plan.minimumSamples, minimum_regimes: plan.minimumRegimes, success_metrics: plan.successMetrics,
      hard_fail_conditions: plan.hardFailConditions, contamination_rules: plan.contaminationRules, stage: plan.stage,
      metadata: { research_version: EVOLUTION_RESEARCH_VERSION, auto_generated_plan: true, stable_identity: true }, evidence_class: EVOLUTION_EVIDENCE_CLASS,
      shadow_only: true, live_execution: false, autonomous_apply_allowed: false,
    });
  }
  if (hypothesisRows.length) {
    const q = await db.from("brian_evolution_hypothesis_snapshots").upsert(hypothesisRows, { onConflict: "snapshot_id", ignoreDuplicates: true });
    if (q.error) throw new Error(`persist_hypotheses:${q.error.message}`);
  }
  if (experimentRows.length) {
    const q = await db.from("brian_evolution_experiments").upsert(experimentRows, { onConflict: "experiment_id", ignoreDuplicates: true });
    if (q.error) throw new Error(`persist_experiments:${q.error.message}`);
  }
  return { hypotheses: hypothesisRows.length, experiments: experimentRows.length };
}

async function receipt(startedAt:string,status:"SUCCESS"|"FAILED"|"SKIPPED",observed:number,stored:number,error?:unknown):Promise<void>{
  const finishedAt=new Date().toISOString();const runId=await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q=await db.from("brian_collector_runs").insert({run_id:runId,collector_id:COLLECTOR_ID,started_at:startedAt,finished_at:finishedAt,status,observed_records:observed,stored_records:stored,degraded_sources:[],error_class:error?"EVOLUTION_RESEARCHER_ERROR":null,error_message:error?String(error).slice(0,1200):null,evidence_class:EVOLUTION_EVIDENCE_CLASS,shadow_only:true,live_execution:false,metadata:{research_version:EVOLUTION_RESEARCH_VERSION,canonical_mutation:false,autonomous_apply_allowed:false,stable_experiment_identity:true,control_version:ALPHA_COMPILER_VERSION}});if(q.error)console.error("research receipt",q.error.message);
}

Deno.serve(async(req:Request)=>{
  if(req.method!=="POST")return out({error:"POST required"},405);const startedAt=new Date().toISOString();
  try{await requireCronAuth(req,db);}catch(error){return out({error:String(error),shadow_only:true,live_execution:false},401);}
  try{
    const lease=await withCollectorLease(db,COLLECTOR_ID,LEASE_SECONDS,async()=>{
      const observedAt=new Date().toISOString();
      const [gaps,challenger,outcomes,reliability]=await Promise.all([loadGaps(),loadChallenger(observedAt),loadOutcomes(observedAt),loadReliability(observedAt)]);
      const inputs:ResearchInputs={gaps,challenger,outcomes,reliability,observedAt};
      const persisted=await persistHypotheses(inputs);const observed=gaps.length+(challenger?1:0)+outcomes.length+reliability.length;
      await receipt(startedAt,"SUCCESS",observed,persisted.hypotheses+persisted.experiments);
      return{status:"SUCCESS",collector_id:COLLECTOR_ID,observed_at:observedAt,inputs:{gaps:gaps.length,challenger:Boolean(challenger),outcome_horizons:outcomes.length,reliability_groups:reliability.length},...persisted,stable_experiment_identity:true,control_version:ALPHA_COMPILER_VERSION,canonical_mutation:false,autonomous_apply_allowed:false,cloud_independent:true,shadow_only:true,live_execution:false};
    });
    if(lease.contended){await receipt(startedAt,"SKIPPED",0,0);return out({status:"SKIPPED_LEASE_CONTENDED",shadow_only:true,live_execution:false});}
    return out(lease.value);
  }catch(error){await receipt(startedAt,"FAILED",0,0,error);return out({status:"FAILED",error:String(error),shadow_only:true,live_execution:false},500);}
});
