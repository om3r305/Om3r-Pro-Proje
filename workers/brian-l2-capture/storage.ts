import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";
import {
  buildRawL2Segment,
  type L2RawSource,
  type RawL2Item,
} from "../../supabase/functions/_shared/l2_raw_segment.ts";
import type {
  CaptureEnvelope,
  DiffDisposition,
  SnapshotDisposition,
} from "../../supabase/functions/_shared/l2_capture_session.ts";
import type {
  DepthDiffEvent,
  DepthSnapshotEvent,
} from "../../supabase/functions/_shared/l2_book.ts";

const RAW_BUCKET = "brian-intelligence-raw";
const PROVIDER = "binance_public";
const EVIDENCE_CLASS = "PROSPECTIVE_DEVELOPMENT_SHADOW";

type NormalizedDepthEvent = DepthDiffEvent | DepthSnapshotEvent;
type CaptureDisposition = DiffDisposition | SnapshotDisposition;

export interface PendingNormalizedCapture {
  envelope: CaptureEnvelope<NormalizedDepthEvent>;
  transport: L2RawSource;
  disposition: CaptureDisposition;
}

export interface PendingL2Capture {
  raw: RawL2Item;
  normalized: PendingNormalizedCapture | null;
}

export interface L2SessionEventInput {
  collectorSessionId: string;
  eventKind:
    | "STARTED"
    | "WS_CONNECTED"
    | "SYNC_STARTED"
    | "SNAPSHOT_TOO_OLD"
    | "SYNCED"
    | "GAP_INVALIDATED"
    | "RESYNC_STARTED"
    | "WS_DISCONNECTED"
    | "STOPPED"
    | "FAILED";
  venue: "binance";
  symbol?: string | null;
  connectionGeneration: number;
  syncGeneration?: number | null;
  arrivalSeqBoundary?: number | null;
  observedAt?: string;
  details?: Record<string, unknown>;
}

function bytes(value: string): Uint8Array {
  return new TextEncoder().encode(value);
}

async function sha256Hex(value: string | Uint8Array): Promise<string> {
  const input = typeof value === "string" ? bytes(value) : value;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", input));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function normalizedPayload(event: NormalizedDepthEvent): Record<string, unknown> {
  if (event.kind === "depth_diff") {
    return {
      firstUpdateId: event.firstUpdateId,
      finalUpdateId: event.finalUpdateId,
      bidMutations: event.bidMutations,
      askMutations: event.askMutations,
    };
  }
  return {
    lastUpdateId: event.lastUpdateId,
    bids: event.bids,
    asks: event.asks,
  };
}

function requireSameCaptureLineage(raw: RawL2Item, normalized: PendingNormalizedCapture): void {
  const { envelope, transport } = normalized;
  if (raw.collectorSessionId !== envelope.collectorSessionId || raw.arrivalSeq !== envelope.arrivalSeq) {
    throw new Error("raw/normalized L2 capture session or arrival sequence mismatch");
  }
  if (raw.connectionGeneration !== envelope.connectionGeneration || raw.syncGeneration !== envelope.syncGeneration) {
    throw new Error("raw/normalized L2 capture generation mismatch");
  }
  if (raw.symbol !== envelope.event.symbol || raw.source !== transport) {
    throw new Error("raw/normalized L2 capture symbol or transport mismatch");
  }
}

export class SupabaseL2CaptureSink {
  private readonly supabase;
  private readonly pending: PendingL2Capture[] = [];
  readonly maxBatchMessages: number;

  constructor(
    supabaseUrl: string,
    serviceRoleKey: string,
    options: { maxBatchMessages?: number } = {},
  ) {
    if (!supabaseUrl.trim() || !serviceRoleKey.trim()) throw new Error("Supabase URL and service-role key are required");
    const maxBatchMessages = options.maxBatchMessages ?? 200;
    if (!Number.isInteger(maxBatchMessages) || maxBatchMessages <= 0) {
      throw new Error("maxBatchMessages must be a positive integer");
    }
    this.maxBatchMessages = maxBatchMessages;
    this.supabase = createClient(supabaseUrl, serviceRoleKey, {
      auth: { persistSession: false, autoRefreshToken: false },
    });
  }

  append(capture: PendingL2Capture): void {
    if (capture.normalized) requireSameCaptureLineage(capture.raw, capture.normalized);
    const prior = this.pending[this.pending.length - 1]?.raw.arrivalSeq ?? null;
    if (prior !== null && capture.raw.arrivalSeq !== prior + 1) {
      throw new Error(`pending raw L2 arrival sequence must be contiguous; expected ${prior + 1}, got ${capture.raw.arrivalSeq}`);
    }
    this.pending.push(capture);
  }

  pendingCount(): number {
    return this.pending.length;
  }

  shouldFlush(): boolean {
    return this.pending.length >= this.maxBatchMessages;
  }

  async recordSessionEvent(input: L2SessionEventInput): Promise<void> {
    if (!Number.isInteger(input.connectionGeneration) || input.connectionGeneration <= 0) {
      throw new Error("connectionGeneration must be positive");
    }
    if (input.syncGeneration != null && (!Number.isInteger(input.syncGeneration) || input.syncGeneration <= 0)) {
      throw new Error("syncGeneration must be positive when present");
    }
    const observedAt = input.observedAt ?? new Date().toISOString();
    if (!Number.isFinite(Date.parse(observedAt))) throw new Error("session event observedAt is invalid");
    const boundary = input.arrivalSeqBoundary ?? null;
    if (boundary != null && (!Number.isSafeInteger(boundary) || boundary < 0)) {
      throw new Error("arrivalSeqBoundary must be a non-negative safe integer");
    }
    const sessionEventId = await sha256Hex(
      `l2-session-event|${input.collectorSessionId}|${input.eventKind}|${input.symbol ?? ""}|` +
        `${input.connectionGeneration}|${input.syncGeneration ?? ""}|${boundary ?? ""}|${observedAt}|${crypto.randomUUID()}`,
    );
    const result = await this.supabase.from("brian_l2_capture_session_events").insert({
      session_event_id: sessionEventId,
      collector_session_id: input.collectorSessionId,
      event_kind: input.eventKind,
      venue: input.venue,
      symbol: input.symbol ?? null,
      connection_generation: input.connectionGeneration,
      sync_generation: input.syncGeneration ?? null,
      arrival_seq_boundary: boundary,
      observed_at: observedAt,
      details: input.details ?? {},
      evidence_class: EVIDENCE_CLASS,
      shadow_only: true,
      live_execution: false,
    });
    if (result.error) throw result.error;
  }

  /**
   * Persist one contiguous batch in causal order:
   *   1. compressed raw NDJSON bytes to immutable Storage,
   *   2. raw-capture pointer,
   *   3. raw-segment sequence index,
   *   4. normalized source-event rows referencing that raw segment.
   *
   * If any step fails, the batch is intentionally retained in memory and the caller should fail
   * the worker closed. A raw segment/capture that reached persistence before a later normalized
   * insert failed remains valuable replay evidence; a future repair can derive normalized rows
   * from raw truth, never the reverse.
   */
  async flush(): Promise<void> {
    if (!this.pending.length) return;
    const batch = [...this.pending];
    const segment = buildRawL2Segment(batch.map((item) => item.raw));
    const uncompressed = bytes(segment.ndjson);
    const compressed = gzip(uncompressed, { level: 6 });
    const payloadHash = await sha256Hex(uncompressed);
    const capturedAt = new Date().toISOString();
    const date = segment.observedAt.slice(0, 10);
    const storagePath = [
      PROVIDER,
      "l2_raw_segments",
      date,
      segment.collectorSessionId,
      `${segment.firstArrivalSeq}-${segment.lastArrivalSeq}-${payloadHash}.ndjson.gz`,
    ].join("/");

    const upload = await this.supabase.storage.from(RAW_BUCKET).upload(storagePath, compressed, {
      contentType: "application/gzip",
      cacheControl: "31536000",
      upsert: false,
    });
    if (upload.error) throw upload.error;

    const rawCaptureId = await sha256Hex(
      `${PROVIDER}|l2_raw_segment|${segment.collectorSessionId}|${segment.firstArrivalSeq}|` +
        `${segment.lastArrivalSeq}|${payloadHash}`,
    );
    const rawInsert = await this.supabase.from("brian_raw_captures").insert({
      capture_id: rawCaptureId,
      provider: PROVIDER,
      record_type: "l2_raw_segment",
      observed_at: segment.observedAt,
      captured_at: capturedAt,
      provenance_uri: "binance-spot-public-l2",
      payload_hash: payloadHash,
      payload: {
        storage_bucket: RAW_BUCKET,
        storage_path: storagePath,
        uncompressed_byte_length: uncompressed.byteLength,
        compressed_byte_length: compressed.byteLength,
        content_type: "application/x-ndjson",
        content_encoding: "gzip",
        collector_session_id: segment.collectorSessionId,
        first_arrival_seq: segment.firstArrivalSeq,
        last_arrival_seq: segment.lastArrivalSeq,
      },
    });
    if (rawInsert.error) throw rawInsert.error;

    const segmentId = await sha256Hex(
      `l2-segment|${segment.collectorSessionId}|${segment.firstArrivalSeq}|${segment.lastArrivalSeq}|${payloadHash}`,
    );
    const segmentInsert = await this.supabase.from("brian_l2_raw_segments").insert({
      segment_id: segmentId,
      raw_capture_id: rawCaptureId,
      collector_session_id: segment.collectorSessionId,
      first_arrival_seq: segment.firstArrivalSeq,
      last_arrival_seq: segment.lastArrivalSeq,
      message_count: segment.messageCount,
      observed_at: segment.observedAt,
      captured_at: capturedAt,
      compression: "gzip",
      evidence_class: EVIDENCE_CLASS,
      shadow_only: true,
      live_execution: false,
    });
    if (segmentInsert.error) throw segmentInsert.error;

    const normalizedRows: Record<string, unknown>[] = [];
    for (let rawMessageIndex = 0; rawMessageIndex < batch.length; rawMessageIndex++) {
      const item = batch[rawMessageIndex];
      if (!item.normalized) continue;
      const { envelope, transport, disposition } = item.normalized;
      const event = envelope.event;
      const eventId = await sha256Hex(`l2-source|${envelope.collectorSessionId}|${envelope.arrivalSeq}`);
      normalizedRows.push({
        event_id: eventId,
        kind: event.kind,
        venue: event.venue,
        symbol: event.symbol,
        exchange_event_at: event.exchangeEventAt,
        collector_received_at: event.collectorReceivedAt,
        ingest_at: event.ingestAt,
        age_ms: event.ageMs,
        clock_skew_ms: event.clockSkewMs,
        payload: normalizedPayload(event),
        source_lineage: {
          ...event.sourceLineage,
          capture_disposition: disposition,
          collector_session_id: envelope.collectorSessionId,
          arrival_seq: envelope.arrivalSeq,
          connection_generation: envelope.connectionGeneration,
          sync_generation: envelope.syncGeneration,
        },
        evidence_class: EVIDENCE_CLASS,
        shadow_only: true,
        live_execution: false,
        collector_session_id: envelope.collectorSessionId,
        arrival_seq: envelope.arrivalSeq,
        connection_generation: envelope.connectionGeneration,
        sync_generation: envelope.syncGeneration,
        transport,
        raw_segment_id: segmentId,
        raw_message_index: rawMessageIndex,
      });
    }

    if (normalizedRows.length) {
      const eventsInsert = await this.supabase.from("brian_l2_source_events").insert(normalizedRows);
      if (eventsInsert.error) throw eventsInsert.error;
    }

    this.pending.splice(0, batch.length);
  }
}
