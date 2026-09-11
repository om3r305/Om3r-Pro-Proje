import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import {
  assessWorldSource,
  buildEvolutionDashboardModel,
  deriveCapabilitySnapshots,
  detectCapabilityGaps,
  discoverWorldSources,
  EVOLUTION_CORE_VERSION,
  type CollectorRunLike,
  type IntelEventLike,
} from "../_shared/evolution_core.ts";
import { EVOLUTION_EVIDENCE_CLASS } from "../_shared/evolution_contract.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

const COLLECTOR_ID = "brian-evolution-orchestrator-v1";
const LEASE_SECONDS = 240;
const RUN_LOOKBACK_MS = 24 * 60 * 60 * 1000;
const EVENT_LOOKBACK_MS = 12 * 60 * 60 * 1000;

function out(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

async function sha256(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function isoBefore(at: string, deltaMs: number): string {
  const base = Date.parse(at);
  return new Date((Number.isFinite(base) ? base : Date.now()) - deltaMs).toISOString();
}

async function insertRows(table: string, rows: Record<string, unknown>[]): Promise<number> {
  if (!rows.length) return 0;
  let stored = 0;
  for (let i = 0; i < rows.length; i += 200) {
    const chunk = rows.slice(i, i + 200);
    const result = await db.from(table).insert(chunk);
    if (result.error) throw new Error(`${table}:${result.error.message}`);
    stored += chunk.length;
  }
  return stored;
}

async function loadInputs(observedAt: string): Promise<{ runs: CollectorRunLike[]; events: IntelEventLike[] }> {
  const [runsQ, eventsQ] = await Promise.all([
    db.from("brian_collector_runs")
      .select("collector_id,started_at,finished_at,status,observed_records,stored_records,degraded_sources,error_class,error_message")
      .gte("started_at", isoBefore(observedAt, RUN_LOOKBACK_MS))
      .order("started_at", { ascending: false })
      .limit(5000),
    db.from("brian_intel_events")
      .select("event_id,source_id,provenance_uri,source_kind,trust_class,first_observed_at,published_at,claim,asset")
      .gte("first_observed_at", isoBefore(observedAt, EVENT_LOOKBACK_MS))
      .order("first_observed_at", { ascending: false })
      .limit(1500),
  ]);
  if (runsQ.error) throw new Error(`collector_runs:${runsQ.error.message}`);
  if (eventsQ.error) throw new Error(`intel_events:${eventsQ.error.message}`);
  return {
    runs: (runsQ.data ?? []) as CollectorRunLike[],
    events: (eventsQ.data ?? []) as IntelEventLike[],
  };
}

async function persistJournal(
  observedAt: string,
  capabilityRows: ReturnType<typeof deriveCapabilitySnapshots>,
  gaps: ReturnType<typeof detectCapabilityGaps>,
  sourceCount: number,
): Promise<number> {
  const healthy = capabilityRows.filter((row) => row.health === "HEALTHY").length;
  const missing = capabilityRows.filter((row) => row.health === "MISSING").length;
  const critical = gaps.filter((row) => row.severity === "CRITICAL").length;
  const eventId = await sha256(`${COLLECTOR_ID}|journal|${observedAt}`);
  return await insertRows("brian_evolution_events", [{
    event_id: eventId,
    entity_type: "EVOLUTION_RUN",
    entity_id: COLLECTOR_ID,
    event_type: "LAYER1_OBSERVATION",
    occurred_at: observedAt,
    stage: "EXPERIMENTAL",
    title: "Brian Evolution Layer 1 observation",
    summary: `${healthy}/${capabilityRows.length} capabilities healthy; ${missing} missing; ${critical} critical gaps; ${sourceCount} source domains observed.`,
    evidence_refs: capabilityRows.flatMap((row) => row.evidenceRefs).slice(0, 100),
    payload: {
      core_version: EVOLUTION_CORE_VERSION,
      healthy_capabilities: healthy,
      missing_capabilities: missing,
      critical_gaps: critical,
      source_domains: sourceCount,
      browser_required: false,
      canonical_mutation: false,
      direct_alpha_influence: false,
    },
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
  }]);
}

async function persistRunReceipt(args: {
  startedAt: string;
  finishedAt: string;
  status: "SUCCESS" | "DEGRADED" | "FAILED" | "SKIPPED_LEASE_CONTENDED";
  capabilities?: number;
  gaps?: number;
  sources?: number;
  assessments?: number;
  journal?: number;
  error?: unknown;
}): Promise<void> {
  const runId = await sha256(`${COLLECTOR_ID}|${args.startedAt}|${args.finishedAt}|${args.status}`);
  const row = {
    run_id: runId,
    started_at: args.startedAt,
    finished_at: args.finishedAt,
    status: args.status,
    capability_snapshots: args.capabilities ?? 0,
    gap_snapshots: args.gaps ?? 0,
    source_candidates: args.sources ?? 0,
    source_assessments: args.assessments ?? 0,
    journal_events: args.journal ?? 0,
    error_class: args.error ? "EVOLUTION_ORCHESTRATOR_ERROR" : null,
    error_message: args.error ? String(args.error).slice(0, 1200) : null,
    metadata: {
      core_version: EVOLUTION_CORE_VERSION,
      cloud_independent: true,
      canonical_mutation: false,
      direct_alpha_influence: false,
    },
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
  };
  const q = await db.from("brian_evolution_orchestrator_runs").insert(row);
  if (q.error) console.error("evolution run receipt", q.error.message);

  const collectorRunId = await sha256(`${COLLECTOR_ID}|collector|${args.startedAt}|${args.finishedAt}|${args.status}`);
  const collectorStatus = args.status === "FAILED" ? "FAILED" : "SUCCESS";
  const c = await db.from("brian_collector_runs").insert({
    run_id: collectorRunId,
    collector_id: COLLECTOR_ID,
    started_at: args.startedAt,
    finished_at: args.finishedAt,
    status: collectorStatus,
    observed_records: (args.capabilities ?? 0) + (args.sources ?? 0),
    stored_records: (args.capabilities ?? 0) + (args.gaps ?? 0) + (args.sources ?? 0) + (args.assessments ?? 0) + (args.journal ?? 0),
    degraded_sources: [],
    error_class: args.error ? "EVOLUTION_ORCHESTRATOR_ERROR" : null,
    error_message: args.error ? String(args.error).slice(0, 1000) : null,
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
  });
  if (c.error) console.error("collector run receipt", c.error.message);
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ error: "POST required" }, 405);
  const startedAt = new Date().toISOString();
  try {
    await requireCronAuth(req, db);
  } catch (error) {
    return out({ error: String(error), shadow_only: true, live_execution: false }, 401);
  }

  try {
    const lease = await withCollectorLease(db, COLLECTOR_ID, LEASE_SECONDS, async () => {
      const observedAt = new Date().toISOString();
      const input = await loadInputs(observedAt);
      const selfRun: CollectorRunLike = {
        collector_id: COLLECTOR_ID,
        started_at: startedAt,
        finished_at: observedAt,
        status: "SUCCESS",
        observed_records: input.events.length,
        stored_records: 0,
      };
      const capabilities = deriveCapabilitySnapshots([...input.runs, selfRun], observedAt);
      const gaps = detectCapabilityGaps(capabilities);
      const discoveredSources = discoverWorldSources(input.events, observedAt);

      const capabilityRows: Record<string, unknown>[] = [];
      for (const row of capabilities) {
        capabilityRows.push({
          snapshot_id: await sha256(`${row.capabilityId}|${row.observedAt}|${EVOLUTION_CORE_VERSION}`),
          capability_id: row.capabilityId,
          observed_at: row.observedAt,
          domain: row.domain,
          name: row.name,
          version: row.version,
          stage: row.stage,
          health: row.health,
          description: row.description,
          source_ids: row.sourceIds,
          dependencies: row.dependencies,
          limitations: row.limitations,
          evidence_refs: row.evidenceRefs,
          metadata: row.metadata,
          evidence_class: row.evidenceClass,
          shadow_only: true,
          live_execution: false,
        });
      }

      const gapRows: Record<string, unknown>[] = [];
      for (const gap of gaps) {
        gapRows.push({
          snapshot_id: await sha256(`${gap.gapId}|${observedAt}`),
          gap_id: gap.gapId,
          observed_at: observedAt,
          capability_id: gap.capabilityId,
          domain: gap.domain,
          severity: gap.severity,
          reason: gap.reason,
          suggested_action: gap.suggestedAction,
          evidence_refs: gap.evidenceRefs,
          metadata: { detector_version: EVOLUTION_CORE_VERSION },
          evidence_class: EVOLUTION_EVIDENCE_CLASS,
          shadow_only: true,
          live_execution: false,
        });
      }

      const sourceRows: Record<string, unknown>[] = [];
      const assessmentRows: Record<string, unknown>[] = [];
      for (const source of discoveredSources) {
        const assessment = assessWorldSource(source, observedAt);
        const stage = assessment.eligibleForResearch ? "VERIFYING" : source.stage;
        sourceRows.push({
          candidate_id: await sha256(`${source.sourceId}|${observedAt}|${stage}`),
          source_id: source.sourceId,
          discovered_at: source.discoveredAt,
          canonical_uri: source.canonicalUri,
          provider: source.provider,
          source_kind: source.sourceKind,
          authority_class: source.authorityClass,
          access_mode: source.accessMode,
          stage,
          freshness_seconds: source.freshnessSeconds,
          corroboration_required: source.corroborationRequired,
          manipulation_risk: source.manipulationRisk,
          rationale: source.rationale,
          metadata: {
            ...source.metadata,
            trust_score: assessment.trustScore,
            direct_alpha_influence: false,
            lifecycle_transition: stage === "VERIFYING" ? "DISCOVERED->VERIFYING" : "DISCOVERED",
          },
          evidence_class: EVOLUTION_EVIDENCE_CLASS,
          shadow_only: true,
          live_execution: false,
        });
        assessmentRows.push({
          assessment_id: await sha256(`${source.sourceId}|assessment|${observedAt}`),
          source_id: source.sourceId,
          assessed_at: observedAt,
          authority_score: assessment.authorityScore,
          freshness_score: assessment.freshnessScore,
          manipulation_penalty: assessment.manipulationPenalty,
          corroboration_penalty: assessment.corroborationPenalty,
          access_penalty: assessment.accessPenalty,
          trust_score: assessment.trustScore,
          eligible_for_research: assessment.eligibleForResearch,
          eligible_for_decision_evidence: false,
          reasons: assessment.reasons,
          metadata: {
            candidate_stage: stage,
            assessed_candidate_stage: source.stage,
            canonical_mutation: false,
          },
          evidence_class: EVOLUTION_EVIDENCE_CLASS,
          shadow_only: true,
          live_execution: false,
        });
      }

      const storedCapabilities = await insertRows("brian_evolution_capability_snapshots", capabilityRows);
      const storedGaps = await insertRows("brian_evolution_gap_snapshots", gapRows);
      const storedSources = await insertRows("brian_world_source_candidates", sourceRows);
      const storedAssessments = await insertRows("brian_world_source_assessments", assessmentRows);
      const storedJournal = await persistJournal(observedAt, capabilities, gaps, discoveredSources.length);

      const dashboard = buildEvolutionDashboardModel(capabilities, gaps, discoveredSources, observedAt, observedAt);
      const finishedAt = new Date().toISOString();
      await persistRunReceipt({
        startedAt,
        finishedAt,
        status: "SUCCESS",
        capabilities: storedCapabilities,
        gaps: storedGaps,
        sources: storedSources,
        assessments: storedAssessments,
        journal: storedJournal,
      });

      return {
        status: "SUCCESS",
        collector_id: COLLECTOR_ID,
        observed_at: observedAt,
        core_version: EVOLUTION_CORE_VERSION,
        capability_snapshots: storedCapabilities,
        gap_snapshots: storedGaps,
        source_candidates: storedSources,
        source_assessments: storedAssessments,
        journal_events: storedJournal,
        dashboard,
        cloud_independent: true,
        canonical_mutation: false,
        direct_alpha_influence: false,
        shadow_only: true,
        live_execution: false,
      };
    });

    if (lease.contended) {
      const finishedAt = new Date().toISOString();
      await persistRunReceipt({ startedAt, finishedAt, status: "SKIPPED_LEASE_CONTENDED" });
      return out({
        status: "SKIPPED_LEASE_CONTENDED",
        collector_id: COLLECTOR_ID,
        shadow_only: true,
        live_execution: false,
      });
    }
    return out(lease.value);
  } catch (error) {
    const finishedAt = new Date().toISOString();
    await persistRunReceipt({ startedAt, finishedAt, status: "FAILED", error });
    return out({
      status: "FAILED",
      collector_id: COLLECTOR_ID,
      error: String(error),
      shadow_only: true,
      live_execution: false,
    }, 500);
  }
});
