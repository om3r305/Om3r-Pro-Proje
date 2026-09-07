import { createClient } from "npm:@supabase/supabase-js@2";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { withCollectorLease } from "../_shared/collector_lease.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });

const COLLECTOR_ID = "brian-alpha-calibration-challenger-v1";
const VERSION = "brian-alpha-calibration-challenger-v1";
const EVIDENCE = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const LEASE_SECONDS = 55;
const MIN_INTERVAL_SECONDS = 45;
const MIN_MATURE_SAMPLES = 100;
const PROMOTE_MIN_SCORE = 0.62;
const ALLOW_ACTION_MIN_SCORE = 0.56;

type Json = Record<string, unknown>;
type ReliabilityRow = {
  independent_group: string;
  sensor_horizon: string;
  outcome_horizon_seconds: number;
  sample_count: number;
  bayesian_hit_rate_beta10_10: number;
  avg_cost_adjusted_signed_bps: number;
  avg_signed_bps: number;
};

type GroupScore = {
  group: string;
  mature: boolean;
  score: number;
  sample_count: number;
  horizon_seconds: number;
  bayesian_hit_rate: number;
  avg_cost_adjusted_bps: number;
  avg_signed_bps: number;
};

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}
function finite(v: unknown, d = 0) { const n = Number(v); return Number.isFinite(n) ? n : d; }
function clip(v: number, lo = 0, hi = 1) { return Math.max(lo, Math.min(hi, v)); }
function errorText(e: unknown) { return e instanceof Error ? `${e.name}: ${e.message}` : String(e); }
async function sha(s: string) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)));
  return [...d].map((x) => x.toString(16).padStart(2, "0")).join("");
}
function horizonWeight(seconds: number) { return seconds >= 3600 ? 1.00 : seconds >= 900 ? .82 : .58; }
function rowScore(r: ReliabilityRow) {
  // This is a challenger score, never the canonical ALPHA evidence score.
  const hitEdge = clip((r.bayesian_hit_rate_beta10_10 - .5) / .15, -1, 1);
  const costEdge = clip(r.avg_cost_adjusted_signed_bps / 25, -1, 1);
  const sample = clip(Math.log1p(Math.max(0, r.sample_count)) / Math.log1p(1500), 0, 1);
  const signed = clip(r.avg_signed_bps / 35, -1, 1);
  const raw = .42 * hitEdge + .38 * costEdge + .12 * signed + .08 * (sample * 2 - 1);
  return clip(.5 + .5 * raw * horizonWeight(r.outcome_horizon_seconds), 0, 1);
}
function bestGroupScore(group: string, rows: ReliabilityRow[]): GroupScore | null {
  const candidates = rows.filter((r) => r.independent_group === group);
  if (!candidates.length) return null;
  const ranked = candidates.map((r) => ({ r, score: rowScore(r) })).sort((a, b) => {
    const am = a.r.sample_count >= MIN_MATURE_SAMPLES ? 1 : 0, bm = b.r.sample_count >= MIN_MATURE_SAMPLES ? 1 : 0;
    if (am !== bm) return bm - am;
    const ah = horizonWeight(a.r.outcome_horizon_seconds), bh = horizonWeight(b.r.outcome_horizon_seconds);
    if (ah !== bh) return bh - ah;
    return b.score - a.score;
  });
  const x = ranked[0];
  return {
    group,
    mature: x.r.sample_count >= MIN_MATURE_SAMPLES,
    score: x.score,
    sample_count: x.r.sample_count,
    horizon_seconds: x.r.outcome_horizon_seconds,
    bayesian_hit_rate: x.r.bayesian_hit_rate_beta10_10,
    avg_cost_adjusted_bps: x.r.avg_cost_adjusted_signed_bps,
    avg_signed_bps: x.r.avg_signed_bps,
  };
}
function combine(scores: GroupScore[]) {
  const mature = scores.filter((x) => x.mature);
  if (!mature.length) return { score: .5, matureCount: 0, avgCostAdj: 0, avgHit: .5 };
  const weight = mature.map((x) => Math.min(1.5, .65 + Math.log1p(x.sample_count) / Math.log1p(1500)) * horizonWeight(x.horizon_seconds));
  const total = weight.reduce((a, b) => a + b, 0) || 1;
  return {
    score: mature.reduce((s, x, i) => s + x.score * weight[i], 0) / total,
    matureCount: mature.length,
    avgCostAdj: mature.reduce((s, x, i) => s + x.avg_cost_adjusted_bps * weight[i], 0) / total,
    avgHit: mature.reduce((s, x, i) => s + x.bayesian_hit_rate * weight[i], 0) / total,
  };
}
function challengerAction(canonical: string, direction: number, evidenceScore: number, calibrationScore: number, matureCount: number, avgCostAdj: number) {
  if (canonical === "VETO") return "KEEP_VETO";
  if (canonical === "OPEN_LONG" || canonical === "OPEN_SHORT") {
    return matureCount >= 2 && calibrationScore >= ALLOW_ACTION_MIN_SCORE && avgCostAdj > 0 ? "ALLOW_ACTION" : "DOWNGRADE_TO_WAIT";
  }
  if (canonical === "WAIT" && direction !== 0 && evidenceScore >= .18 && matureCount >= 2 && calibrationScore >= PROMOTE_MIN_SCORE && avgCostAdj > 2) {
    return direction > 0 ? "PROMOTE_CANDIDATE_LONG" : "PROMOTE_CANDIDATE_SHORT";
  }
  return "KEEP_WAIT";
}

async function recordRun(startedAt: string, status: string, observed: number, stored: number, degraded: string[], metadata: Json = {}, error?: unknown) {
  const finishedAt = new Date().toISOString(), runId = await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q = await db.from("brian_collector_runs").insert({
    run_id: runId, collector_id: COLLECTOR_ID, started_at: startedAt, finished_at: finishedAt, status,
    observed_records: observed, stored_records: stored, degraded_sources: degraded,
    error_class: error ? "ALPHA_CALIBRATION_CHALLENGER_ERROR" : null,
    error_message: error ? errorText(error).slice(0, 1500) : null,
    evidence_class: EVIDENCE, shadow_only: true, live_execution: false,
    metadata: { challenger_version: VERSION, canonical_mutation: false, ...metadata },
  });
  if (q.error) console.error("challenger run log failed", q.error.message);
}

async function runChallenger() {
  const startedAt = new Date().toISOString();
  const latestWindow = await db.from("brian_sensor_reliability_shadow_snapshots")
    .select("window_end,generated_at").order("window_end", { ascending: false }).order("generated_at", { ascending: false }).limit(1).maybeSingle();
  if (latestWindow.error) throw latestWindow.error;
  if (!latestWindow.data?.window_end) {
    await recordRun(startedAt, "DEGRADED", 0, 0, ["NO_RELIABILITY_SNAPSHOT"], { status: "NO_RELIABILITY_SNAPSHOT" });
    return { status: "NO_RELIABILITY_SNAPSHOT", canonical_mutation: false, shadow_only: true, live_execution: false };
  }
  const windowEnd = String(latestWindow.data.window_end);
  const [rel, decisions, recent] = await Promise.all([
    db.from("brian_sensor_reliability_shadow_snapshots")
      .select("independent_group,sensor_horizon,outcome_horizon_seconds,sample_count,bayesian_hit_rate_beta10_10,avg_cost_adjusted_signed_bps,avg_signed_bps")
      .eq("window_end", windowEnd).order("sample_count", { ascending: false }).limit(500),
    db.from("brian_alpha_decisions")
      .select("decision_id,observed_at,asset_id,action,direction,evidence_score,support_groups,conflict_groups,estimated_round_trip_cost_bps,veto_reason,reason")
      .order("observed_at", { ascending: false }).limit(120),
    db.from("brian_alpha_calibration_challenger")
      .select("decision_id").order("observed_at", { ascending: false }).limit(600),
  ]);
  if (rel.error) throw rel.error; if (decisions.error) throw decisions.error; if (recent.error) throw recent.error;
  const reliability: ReliabilityRow[] = (rel.data ?? []).map((r) => ({
    independent_group: String(r.independent_group), sensor_horizon: String(r.sensor_horizon), outcome_horizon_seconds: Math.trunc(finite(r.outcome_horizon_seconds)),
    sample_count: Math.trunc(finite(r.sample_count)), bayesian_hit_rate_beta10_10: finite(r.bayesian_hit_rate_beta10_10, .5),
    avg_cost_adjusted_signed_bps: finite(r.avg_cost_adjusted_signed_bps), avg_signed_bps: finite(r.avg_signed_bps),
  }));
  const seen = new Set((recent.data ?? []).map((x) => String(x.decision_id))), inserts: Json[] = [];
  for (const d of decisions.data ?? []) {
    const decisionId = String(d.decision_id); if (seen.has(decisionId)) continue;
    const support = Array.isArray(d.support_groups) ? [...new Set(d.support_groups.map(String))] : [];
    const scores = support.map((g) => bestGroupScore(g, reliability)).filter((x): x is GroupScore => x !== null);
    const agg = combine(scores), canonical = String(d.action), direction = Math.trunc(finite(d.direction)), evidenceScore = finite(d.evidence_score);
    const challenger = challengerAction(canonical, direction, evidenceScore, agg.score, agg.matureCount, agg.avgCostAdj);
    const id = await sha(`${VERSION}|${decisionId}|${windowEnd}|${challenger}|${agg.score.toFixed(8)}`);
    inserts.push({
      challenger_id: id, decision_id: decisionId, observed_at: String(d.observed_at), evaluated_at: new Date().toISOString(), asset_id: String(d.asset_id),
      canonical_action: canonical, canonical_direction: direction, canonical_evidence_score: evidenceScore,
      challenger_action: challenger, calibration_score: agg.score, mature_support_count: agg.matureCount,
      avg_support_cost_adjusted_bps: agg.avgCostAdj, avg_support_bayesian_hit_rate: agg.avgHit,
      reliability_window_end: windowEnd,
      rationale: {
        support_groups: support, group_scores: scores, canonical_veto_reason: d.veto_reason, canonical_reason: d.reason,
        thresholds: { min_mature_samples: MIN_MATURE_SAMPLES, allow_action_score: ALLOW_ACTION_MIN_SCORE, promote_candidate_score: PROMOTE_MIN_SCORE },
        role: "SHADOW_CHALLENGER_ONLY", canonical_mutation: false,
      },
      evidence_class: EVIDENCE, shadow_only: true, live_execution: false,
    });
  }
  if (inserts.length) {
    const q = await db.from("brian_alpha_calibration_challenger").insert(inserts); if (q.error) throw q.error;
  }
  const promoted = inserts.filter((x) => String(x.challenger_action).startsWith("PROMOTE_CANDIDATE")).length;
  const downgraded = inserts.filter((x) => x.challenger_action === "DOWNGRADE_TO_WAIT").length;
  const allowed = inserts.filter((x) => x.challenger_action === "ALLOW_ACTION").length;
  await recordRun(startedAt, "SUCCESS", decisions.data?.length ?? 0, inserts.length, [], { reliability_window_end: windowEnd, inserted: inserts.length, promoted_candidates: promoted, downgraded_actions: downgraded, allowed_actions: allowed });
  return { status: "CAPTURED", challenger_version: VERSION, reliability_window_end: windowEnd, evaluated: decisions.data?.length ?? 0, inserted: inserts.length, promoted_candidates: promoted, downgraded_actions: downgraded, allowed_actions: allowed, canonical_mutation: false, shadow_only: true, live_execution: false };
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return json({ error: "POST required" }, 405);
  try { await requireCronAuth(req, db); }
  catch (e) { const message = errorText(e), unauthorized = message.includes("UNAUTHORIZED_CRON"); return json({ status: unauthorized ? "UNAUTHORIZED" : "FAILED_CLOSED", error: message, canonical_mutation: false, shadow_only: true, live_execution: false }, unauthorized ? 401 : 503); }
  const startedAt = new Date().toISOString();
  try {
    const last = await db.from("brian_collector_runs").select("started_at").eq("collector_id", COLLECTOR_ID).in("status", ["SUCCESS", "DEGRADED"]).order("started_at", { ascending: false }).limit(1).maybeSingle();
    if (last.error) throw last.error;
    if (last.data?.started_at) { const age = (Date.now() - Date.parse(String(last.data.started_at))) / 1000; if (Number.isFinite(age) && age < MIN_INTERVAL_SECONDS) return json({ status: "SKIPPED_RATE_GUARD", age_seconds: age, canonical_mutation: false, shadow_only: true, live_execution: false }); }
    const lease = await withCollectorLease(db, COLLECTOR_ID, LEASE_SECONDS, async () => await runChallenger());
    if (lease.contended) return json({ status: "SKIPPED_LEASE_CONTENDED", canonical_mutation: false, shadow_only: true, live_execution: false });
    return json(lease.value!);
  } catch (e) {
    console.error("brian-alpha-calibration-challenger-v1 failed", errorText(e));
    try { await recordRun(startedAt, "FAILED", 0, 0, [], {}, e); } catch { /* primary error wins */ }
    return json({ status: "FAILED_CLOSED", error: errorText(e), canonical_mutation: false, shadow_only: true, live_execution: false }, 500);
  }
});
