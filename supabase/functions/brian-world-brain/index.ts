import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { EVOLUTION_EVIDENCE_CLASS } from "../_shared/evolution_contract.ts";
import { buildWorldBrainBatch, WORLD_BRAIN_VERSION, type WorldBrainInputEvent } from "../_shared/world_brain.ts";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const COLLECTOR_ID = "brian-world-brain-v1";
const LEASE_SECONDS = 300;
const LOOKBACK_MS = 12 * 60 * 60 * 1000;

function out(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

async function sha256(value: string): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function persistRows(table: string, rows: Record<string, unknown>[], onConflict: string): Promise<number> {
  if (!rows.length) return 0;
  let stored = 0;
  for (let i = 0; i < rows.length; i += 150) {
    const chunk = rows.slice(i, i + 150);
    const q = await db.from(table).upsert(chunk, { onConflict, ignoreDuplicates: true });
    if (q.error) throw new Error(`${table}:${q.error.message}`);
    stored += chunk.length;
  }
  return stored;
}

function fromIso(ms: number): string {
  return new Date(ms).toISOString();
}

async function loadEvents(nowMs: number): Promise<WorldBrainInputEvent[]> {
  const q = await db.from("brian_intel_events")
    .select("event_id,asset,event_kind,source_kind,source_id,published_at,first_observed_at,claim,direction,magnitude,trust_class,entity_confidence,provenance_uri,metadata")
    .gte("first_observed_at", fromIso(nowMs - LOOKBACK_MS))
    .order("first_observed_at", { ascending: false })
    .limit(2500);
  if (q.error) throw new Error(`intel_events:${q.error.message}`);
  return (q.data ?? []) as WorldBrainInputEvent[];
}

async function journal(observedAt: string, counts: Record<string, number>): Promise<number> {
  const eventId = await sha256(`${COLLECTOR_ID}|journal|${observedAt}`);
  const q = await db.from("brian_evolution_events").insert({
    event_id: eventId,
    entity_type: "WORLD_BRAIN_RUN",
    entity_id: COLLECTOR_ID,
    event_type: "LAYER2_WORLD_MODEL_SNAPSHOT",
    occurred_at: observedAt,
    stage: "EXPERIMENTAL",
    title: "Brian World Brain prospective snapshot",
    summary: `${counts.event_frames ?? 0} event frames, ${counts.entity_observations ?? 0} entity mentions, ${counts.narratives ?? 0} narratives, ${counts.mechanisms ?? 0} causal mechanisms and ${counts.scenarios ?? 0} scenario branches.`,
    evidence_refs: [],
    payload: {
      world_brain_version: WORLD_BRAIN_VERSION,
      ...counts,
      causal_claims_are_research_hypotheses: true,
      direct_alpha_influence: false,
      canonical_mutation: false,
    },
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
  });
  if (q.error) throw new Error(`evolution_journal:${q.error.message}`);
  return 1;
}

async function runReceipt(args: {
  startedAt: string;
  finishedAt: string;
  status: "SUCCESS" | "DEGRADED" | "FAILED" | "SKIPPED_LEASE_CONTENDED";
  input?: number;
  counts?: Record<string, number>;
  error?: unknown;
}): Promise<void> {
  const counts = args.counts ?? {};
  const runId = await sha256(`${COLLECTOR_ID}|${args.startedAt}|${args.finishedAt}|${args.status}`);
  const q = await db.from("brian_world_brain_runs").insert({
    run_id: runId,
    started_at: args.startedAt,
    finished_at: args.finishedAt,
    status: args.status,
    input_events: args.input ?? 0,
    event_frames: counts.event_frames ?? 0,
    entity_observations: counts.entity_observations ?? 0,
    relation_assertions: counts.relations ?? 0,
    narrative_snapshots: counts.narratives ?? 0,
    future_events: counts.future_events ?? 0,
    causal_mechanisms: counts.mechanisms ?? 0,
    scenario_snapshots: counts.scenarios ?? 0,
    asset_impacts: counts.asset_impacts ?? 0,
    error_class: args.error ? "WORLD_BRAIN_ERROR" : null,
    error_message: args.error ? String(args.error).slice(0, 1200) : null,
    metadata: {
      version: WORLD_BRAIN_VERSION,
      lookback_hours: LOOKBACK_MS / 3600000,
      causal_claims_are_research_hypotheses: true,
      direct_alpha_influence: false,
    },
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
    direct_alpha_influence: false,
  });
  if (q.error) console.error("world brain run receipt", q.error.message);

  const collectorRunId = await sha256(`${COLLECTOR_ID}|collector|${args.startedAt}|${args.finishedAt}|${args.status}`);
  const collectorStatus = args.status === "FAILED" ? "FAILED" : args.status === "SKIPPED_LEASE_CONTENDED" ? "SKIPPED" : "SUCCESS";
  const c = await db.from("brian_collector_runs").insert({
    run_id: collectorRunId,
    collector_id: COLLECTOR_ID,
    started_at: args.startedAt,
    finished_at: args.finishedAt,
    status: collectorStatus,
    observed_records: args.input ?? 0,
    stored_records: Object.values(counts).reduce((sum, value) => sum + Number(value || 0), 0),
    degraded_sources: [],
    error_class: args.error ? "WORLD_BRAIN_ERROR" : null,
    error_message: args.error ? String(args.error).slice(0, 1000) : null,
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
    metadata: { version: WORLD_BRAIN_VERSION, direct_alpha_influence: false },
  });
  if (c.error) console.error("world brain collector receipt", c.error.message);
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
      const now = Date.now();
      const observedAt = new Date(now).toISOString();
      const events = await loadEvents(now);
      const batch = buildWorldBrainBatch(events, observedAt);

      const eventFrames = batch.eventFrames.map((row) => ({
        frame_id: row.frameId, event_id: row.eventId, observed_at: row.observedAt, published_at: row.publishedAt,
        event_kind: row.eventKind, source_id: row.sourceId, claim: row.claim, primary_asset: row.primaryAsset,
        entity_ids: row.entityIds, narrative_ids: row.narrativeIds, direction_hint: row.directionHint,
        source_trust_class: row.sourceTrustClass, provenance_uri: row.provenanceUri, stage: row.stage,
        evidence_refs: row.evidenceRefs, evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true,
        live_execution: false, direct_alpha_influence: false,
      }));
      const entityRows = batch.entityObservations.map((row) => ({
        observation_id: row.observationId, entity_id: row.entityId, canonical_name: row.canonicalName,
        entity_type: row.entityType, observed_at: row.observedAt, event_id: row.eventId, match_kind: row.matchKind,
        confidence: row.confidence, provenance_uri: row.provenanceUri, evidence_refs: row.evidenceRefs,
        evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));
      const relations = batch.relationAssertions.map((row) => ({
        assertion_id: row.assertionId, src_entity_id: row.srcEntityId, dst_entity_id: row.dstEntityId,
        relation: row.relation, observed_at: row.observedAt, event_id: row.eventId, confidence: row.confidence,
        mechanism: row.mechanism, stage: row.stage, evidence_refs: row.evidenceRefs,
        evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));
      const narratives = batch.narrativeSnapshots.map((row) => ({
        snapshot_id: row.snapshotId, narrative_id: row.narrativeId, label: row.label, observed_at: row.observedAt,
        event_ids: row.eventIds, entity_ids: row.entityIds, strength: row.strength, breadth: row.breadth,
        direction_balance: row.directionBalance, stage: row.stage, evidence_refs: row.evidenceRefs,
        evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));
      const futureEvents = batch.futureEvents.map((row) => ({
        future_event_id: row.futureEventId, event_kind: row.eventKind, scheduled_at: row.scheduledAt,
        first_observed_at: row.firstObservedAt, title: row.title, entity_ids: row.entityIds, asset_ids: row.assetIds,
        source_event_id: row.sourceEventId, confidence: row.confidence, stage: row.stage, evidence_refs: row.evidenceRefs,
        evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));
      const mechanisms = batch.causalMechanisms.map((row) => ({
        mechanism_id: row.mechanismId, narrative_id: row.narrativeId, observed_at: row.observedAt, cause: row.cause,
        transmission: row.transmission, affected_assets: row.affectedAssets, confidence: row.confidence, stage: row.stage,
        evidence_refs: row.evidenceRefs, counter_evidence_required: true, evidence_class: EVOLUTION_EVIDENCE_CLASS,
        shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));
      const scenarios = batch.scenarios.map((row) => ({
        scenario_id: row.scenarioId, mechanism_id: row.mechanismId, observed_at: row.observedAt, branch: row.branch,
        assumptions: row.assumptions, invalidators: row.invalidators, asset_impacts: row.assetImpacts,
        confidence: row.confidence, stage: row.stage, evidence_refs: row.evidenceRefs,
        evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));
      const impacts = batch.assetImpacts.map((row) => ({
        impact_id: row.impactId, asset_id: row.assetId, observed_at: row.observedAt, mechanism_id: row.mechanismId,
        scenario_id: row.scenarioId, conditional_direction: row.conditionalDirection, confidence: row.confidence,
        rationale: row.rationale, stage: row.stage, evidence_refs: row.evidenceRefs,
        evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));

      const counts = {
        event_frames: await persistRows("brian_world_event_frames", eventFrames, "frame_id"),
        entity_observations: await persistRows("brian_world_entity_observations", entityRows, "observation_id"),
        relations: await persistRows("brian_world_relation_assertions", relations, "assertion_id"),
        narratives: await persistRows("brian_world_narrative_snapshots", narratives, "snapshot_id"),
        future_events: await persistRows("brian_world_future_events", futureEvents, "future_event_id"),
        mechanisms: await persistRows("brian_world_causal_mechanisms", mechanisms, "mechanism_id"),
        scenarios: await persistRows("brian_world_scenario_snapshots", scenarios, "scenario_id"),
        asset_impacts: await persistRows("brian_world_asset_impact_candidates", impacts, "impact_id"),
      };
      await journal(observedAt, counts);
      const finishedAt = new Date().toISOString();
      await runReceipt({ startedAt, finishedAt, status: "SUCCESS", input: events.length, counts });
      return {
        status: "SUCCESS",
        collector_id: COLLECTOR_ID,
        observed_at: observedAt,
        world_brain_version: WORLD_BRAIN_VERSION,
        input_events: events.length,
        ...counts,
        causal_claims_are_research_hypotheses: true,
        direct_alpha_influence: false,
        canonical_mutation: false,
        cloud_independent: true,
        shadow_only: true,
        live_execution: false,
      };
    });

    if (lease.contended) {
      const finishedAt = new Date().toISOString();
      await runReceipt({ startedAt, finishedAt, status: "SKIPPED_LEASE_CONTENDED" });
      return out({ status: "SKIPPED_LEASE_CONTENDED", collector_id: COLLECTOR_ID, shadow_only: true, live_execution: false });
    }
    return out(lease.value);
  } catch (error) {
    const finishedAt = new Date().toISOString();
    await runReceipt({ startedAt, finishedAt, status: "FAILED", error });
    return out({ status: "FAILED", collector_id: COLLECTOR_ID, error: String(error), shadow_only: true, live_execution: false }, 500);
  }
});
