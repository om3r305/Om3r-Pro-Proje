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
const EVENT_OVERLAP_MS = 15 * 60 * 1000;
const MAX_INCREMENTAL_EVENTS = 200;
const WRITE_CHUNK = 10;
const CLASSIFICATION_GUARD_VERSION = "world-brain-classifier-guard.v1";
const SEMANTIC_DEDUPE_VERSION = "world-semantic-dedupe.v1";

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

function transientDbError(message: string): boolean {
  return /statement timeout|connection.*timed out|could not query the database|schema cache|PGRST002|PGRST000|upstream request timeout/i.test(message);
}

async function persistChunk(table: string, chunk: Record<string, unknown>[], onConflict: string): Promise<number> {
  if (!chunk.length) return 0;
  const q = await db.from(table).upsert(chunk, { onConflict, ignoreDuplicates: true }).select(onConflict);
  if (!q.error) return Array.isArray(q.data) ? q.data.length : 0;
  const message = String(q.error.message ?? q.error);
  if (chunk.length > 1 && transientDbError(message)) {
    const mid = Math.ceil(chunk.length / 2);
    const left = await persistChunk(table, chunk.slice(0, mid), onConflict);
    const right = await persistChunk(table, chunk.slice(mid), onConflict);
    return left + right;
  }
  throw new Error(`${table}:${message}`);
}

async function persistRows(table: string, rows: Record<string, unknown>[], onConflict: string): Promise<number> {
  if (!rows.length) return 0;
  let stored = 0;
  for (let i = 0; i < rows.length; i += WRITE_CHUNK) {
    stored += await persistChunk(table, rows.slice(i, i + WRITE_CHUNK), onConflict);
  }
  return stored;
}

function fromIso(ms: number): string {
  return new Date(ms).toISOString();
}

function semanticText(value: unknown): string {
  return String(value ?? "").toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, " ").trim().slice(0, 700);
}
function semanticTrustRank(event: WorldBrainInputEvent): number {
  const trust = String(event.trust_class ?? "").toUpperCase();
  if (trust === "OFFICIAL_PRIMARY") return 4;
  if (trust === "INDEPENDENT_PROFESSIONAL") return 3;
  if (trust === "VERIFIED_SECONDARY") return 2;
  return 1;
}
function semanticKindRank(event: WorldBrainInputEvent): number {
  const kind = String(event.event_kind ?? "").toUpperCase();
  if (/^OFFICIAL_SOURCE_ITEM$|^OFFICIAL_PRIMARY_RELEASE$|^OFFICIAL_EXCHANGE_STATUS$/.test(kind)) return 4;
  if (/INSTITUTIONAL_RELEASE/.test(kind)) return 3;
  if (/BREAKING_SCOUT/.test(kind)) return 2;
  return 1;
}
function dedupeSemanticEvents(events: WorldBrainInputEvent[]): WorldBrainInputEvent[] {
  const groups = new Map<string, { winner: WorldBrainInputEvent; earliest: string; ids: string[] }>();
  for (const event of events) {
    const claim = semanticText(event.claim);
    if (!claim) continue;
    const source = semanticText(event.source_id || "unknown");
    const dateBasis = String(event.published_at ?? event.first_observed_at ?? "").slice(0, 10);
    const key = `${source}|${dateBasis}|${claim}`;
    const prior = groups.get(key);
    if (!prior) {
      groups.set(key, { winner: event, earliest: String(event.first_observed_at ?? ""), ids: [event.event_id] });
      continue;
    }
    prior.ids.push(event.event_id);
    const eventAt = String(event.first_observed_at ?? "");
    if (eventAt && (!prior.earliest || eventAt < prior.earliest)) prior.earliest = eventAt;
    const a = prior.winner;
    const better =
      semanticTrustRank(event) > semanticTrustRank(a) ||
      (semanticTrustRank(event) === semanticTrustRank(a) && semanticKindRank(event) > semanticKindRank(a)) ||
      (semanticTrustRank(event) === semanticTrustRank(a) && semanticKindRank(event) === semanticKindRank(a) &&
       eventAt && String(a.first_observed_at ?? "") && eventAt < String(a.first_observed_at));
    if (better) prior.winner = event;
  }
  const out: WorldBrainInputEvent[] = [];
  for (const group of groups.values()) {
    const meta = (group.winner.metadata && typeof group.winner.metadata === "object")
      ? group.winner.metadata as Record<string, unknown> : {};
    out.push({
      ...group.winner,
      first_observed_at: group.earliest || group.winner.first_observed_at,
      metadata: {
        ...meta,
        semantic_dedupe: {
          version: SEMANTIC_DEDUPE_VERSION,
          collapsed_count: group.ids.length,
          source_event_ids: group.ids.slice(0, 24),
        },
      },
    });
  }
  return out.sort((a,b)=>String(a.first_observed_at ?? "").localeCompare(String(b.first_observed_at ?? "")));
}

function narrativeAllowed(event: WorldBrainInputEvent | undefined, narrativeId: string): boolean {
  if (!event) return false;
  const claim = String(event.claim ?? "");
  if (narrativeId === "narrative:GEOPOLITICS") {
    return /\bwar\b|\barmed conflict\b|\bconflict\b|\bsanctions?\b|\bmissiles?\b|\binvasion\b|\bceasefire\b|\bgeopolit/i.test(claim);
  }
  if (narrativeId === "narrative:CRYPTO_REGULATION") {
    const crypto = /\bcrypto\b|digital asset|\bbitcoin\b|\bethereum\b|\bstablecoin\b|\bblockchain\b|\betf\b|\btoken\b/i.test(claim);
    const regulation = /crypto regulation|digital asset regulation|securities and exchange commission|\bsec\b|\bregulat(?:e|ed|es|ing|ion|ory)\b|\benforcement\b|\blawsuit\b|securities law|\brulemaking\b|\bapproval\b|\bban\b/i.test(claim);
    return crypto && regulation;
  }
  if (narrativeId === "narrative:PRODUCT_LAUNCH") {
    return /product launch|product event|\bunveil(?:s|ed|ing)?\b|\bnew iphone\b|\bnew gpu\b|\blaunch(?:es|ed|ing)?\b.{0,48}\b(product|platform|service|device|model|gpu|iphone|token|mainnet)\b/i.test(claim);
  }
  return true;
}

function applyClassificationGuard(batch: ReturnType<typeof buildWorldBrainBatch>, events: WorldBrainInputEvent[]) {
  const eventById = new Map(events.map((event) => [event.event_id, event]));
  const eventFrames = batch.eventFrames.map((row) => ({
    ...row,
    narrativeIds: row.narrativeIds.filter((id) => narrativeAllowed(eventById.get(row.eventId), id)),
  }));

  const allowedByNarrative = new Map<string, Set<string>>();
  for (const frame of eventFrames) {
    for (const narrativeId of frame.narrativeIds) {
      const set = allowedByNarrative.get(narrativeId) ?? new Set<string>();
      set.add(frame.eventId);
      allowedByNarrative.set(narrativeId, set);
    }
  }

  const narrativeSnapshots = batch.narrativeSnapshots.flatMap((row) => {
    const allowedEvents = allowedByNarrative.get(row.narrativeId);
    if (!allowedEvents) return [];
    const eventIds = row.eventIds.filter((eventId) => allowedEvents.has(eventId));
    if (!eventIds.length) return [];
    const frameEntities = eventFrames
      .filter((frame) => eventIds.includes(frame.eventId))
      .flatMap((frame) => frame.entityIds);
    return [{ ...row, eventIds, entityIds: [...new Set(frameEntities)] }];
  });

  const allowedNarratives = new Set(narrativeSnapshots.map((row) => row.narrativeId));
  const causalMechanisms = batch.causalMechanisms.filter((row) => allowedNarratives.has(row.narrativeId));
  const allowedMechanisms = new Set(causalMechanisms.map((row) => row.mechanismId));
  const scenarios = batch.scenarios.filter((row) => allowedMechanisms.has(row.mechanismId));
  const allowedScenarios = new Set(scenarios.map((row) => row.scenarioId));
  const assetImpacts = batch.assetImpacts.filter((row) => allowedMechanisms.has(row.mechanismId) && allowedScenarios.has(row.scenarioId));

  return {
    ...batch,
    eventFrames,
    narrativeSnapshots,
    causalMechanisms,
    scenarios,
    assetImpacts,
  };
}

const EVENT_SELECT = "event_id,asset,event_kind,source_kind,source_id,published_at,first_observed_at,claim,direction,magnitude,trust_class,entity_confidence,provenance_uri,metadata";

async function loadContextEvents(nowMs: number): Promise<WorldBrainInputEvent[]> {
  const q = await db.from("brian_intel_events")
    .select(EVENT_SELECT)
    .gte("first_observed_at", fromIso(nowMs - LOOKBACK_MS))
    .order("first_observed_at", { ascending: false })
    .limit(2500);
  if (q.error) throw new Error(`intel_events_context:${q.error.message}`);
  return (q.data ?? []) as WorldBrainInputEvent[];
}

async function loadIncrementalEvents(nowMs: number): Promise<WorldBrainInputEvent[]> {
  // Do not watermark from frame.observed_at with an overlapped ascending LIMIT:
  // once the overlap contains more than MAX_INCREMENTAL_EVENTS that pattern can
  // replay the same oldest rows forever and starve fresh news. Read a bounded
  // latest window instead; frame writes are idempotent by frame_id.
  const q = await db.from("brian_intel_events")
    .select(EVENT_SELECT)
    .gte("first_observed_at", fromIso(nowMs - LOOKBACK_MS))
    .order("first_observed_at", { ascending: false })
    .limit(MAX_INCREMENTAL_EVENTS);
  if (q.error) throw new Error(`intel_events_incremental:${q.error.message}`);
  const rows = ((q.data ?? []) as WorldBrainInputEvent[]);
  rows.reverse();
  return rows;
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
      classification_guard_version: CLASSIFICATION_GUARD_VERSION,
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
      classification_guard_version: CLASSIFICATION_GUARD_VERSION,
      semantic_dedupe_version: SEMANTIC_DEDUPE_VERSION,
      incremental_strategy: "latest_window_idempotent_v2",
      max_incremental_events: MAX_INCREMENTAL_EVENTS,
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
    metadata: { version: WORLD_BRAIN_VERSION, classification_guard_version: CLASSIFICATION_GUARD_VERSION, semantic_dedupe_version: SEMANTIC_DEDUPE_VERSION, direct_alpha_influence: false },
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
      const contextRawEvents = await loadContextEvents(now);
      const incrementalRawEvents = await loadIncrementalEvents(now);
      const contextEvents = dedupeSemanticEvents(contextRawEvents);
      const incrementalEvents = dedupeSemanticEvents(incrementalRawEvents);
      const contextBatch = applyClassificationGuard(buildWorldBrainBatch(contextEvents, observedAt), contextEvents);
      const eventBatch = applyClassificationGuard(buildWorldBrainBatch(incrementalEvents, observedAt), incrementalEvents);

      const eventFrames = eventBatch.eventFrames.map((row) => ({
        frame_id: row.frameId, event_id: row.eventId, observed_at: row.observedAt, published_at: row.publishedAt,
        event_kind: row.eventKind, source_id: row.sourceId, claim: row.claim, primary_asset: row.primaryAsset,
        entity_ids: row.entityIds, narrative_ids: row.narrativeIds, direction_hint: row.directionHint,
        source_trust_class: row.sourceTrustClass, provenance_uri: row.provenanceUri, stage: row.stage,
        evidence_refs: row.evidenceRefs, evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true,
        live_execution: false, direct_alpha_influence: false,
      }));
      const entityRows = eventBatch.entityObservations.map((row) => ({
        observation_id: row.observationId, entity_id: row.entityId, canonical_name: row.canonicalName,
        entity_type: row.entityType, observed_at: row.observedAt, event_id: row.eventId, match_kind: row.matchKind,
        confidence: row.confidence, provenance_uri: row.provenanceUri, evidence_refs: row.evidenceRefs,
        evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));
      const relations = eventBatch.relationAssertions.map((row) => ({
        assertion_id: row.assertionId, src_entity_id: row.srcEntityId, dst_entity_id: row.dstEntityId,
        relation: row.relation, observed_at: row.observedAt, event_id: row.eventId, confidence: row.confidence,
        mechanism: row.mechanism, stage: row.stage, evidence_refs: row.evidenceRefs,
        evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));
      const narratives = contextBatch.narrativeSnapshots.map((row) => ({
        snapshot_id: row.snapshotId, narrative_id: row.narrativeId, label: row.label, observed_at: row.observedAt,
        event_ids: row.eventIds, entity_ids: row.entityIds, strength: row.strength, breadth: row.breadth,
        direction_balance: row.directionBalance, stage: row.stage, evidence_refs: row.evidenceRefs,
        evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));
      const futureEvents = eventBatch.futureEvents.map((row) => ({
        future_event_id: row.futureEventId, event_kind: row.eventKind, scheduled_at: row.scheduledAt,
        first_observed_at: row.firstObservedAt, title: row.title, entity_ids: row.entityIds, asset_ids: row.assetIds,
        source_event_id: row.sourceEventId, confidence: row.confidence, stage: row.stage, evidence_refs: row.evidenceRefs,
        evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));
      const mechanisms = contextBatch.causalMechanisms.map((row) => ({
        mechanism_id: row.mechanismId, narrative_id: row.narrativeId, observed_at: row.observedAt, cause: row.cause,
        transmission: row.transmission, affected_assets: row.affectedAssets, confidence: row.confidence, stage: row.stage,
        evidence_refs: row.evidenceRefs, counter_evidence_required: true, evidence_class: EVOLUTION_EVIDENCE_CLASS,
        shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));
      const scenarios = contextBatch.scenarios.map((row) => ({
        scenario_id: row.scenarioId, mechanism_id: row.mechanismId, observed_at: row.observedAt, branch: row.branch,
        assumptions: row.assumptions, invalidators: row.invalidators, asset_impacts: row.assetImpacts,
        confidence: row.confidence, stage: row.stage, evidence_refs: row.evidenceRefs,
        evidence_class: EVOLUTION_EVIDENCE_CLASS, shadow_only: true, live_execution: false, direct_alpha_influence: false,
      }));
      const impacts = contextBatch.assetImpacts.map((row) => ({
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
      await runReceipt({ startedAt, finishedAt, status: "SUCCESS", input: contextEvents.length, counts });
      return {
        status: "SUCCESS",
        collector_id: COLLECTOR_ID,
        observed_at: observedAt,
        world_brain_version: WORLD_BRAIN_VERSION,
        classification_guard_version: CLASSIFICATION_GUARD_VERSION,
        semantic_dedupe_version: SEMANTIC_DEDUPE_VERSION,
        incremental_strategy: "latest_window_idempotent_v2",
        raw_input_events: contextRawEvents.length,
        input_events: contextEvents.length,
        semantic_duplicates_collapsed: contextRawEvents.length-contextEvents.length,
        raw_incremental_events: incrementalRawEvents.length,
        incremental_events: incrementalEvents.length,
        incremental_duplicates_collapsed: incrementalRawEvents.length-incrementalEvents.length,
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
