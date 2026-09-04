// Brian 2026 persistent real Binance Spot L2 collector.
//
// SHADOW_RESEARCH_ONLY. Public market data only: no Binance API key, no account endpoint, no
// order endpoint, no BUY/SELL output. This is deliberately a portable long-running Deno worker
// rather than a Supabase Edge Function, because continuous WebSocket capture must outlive an
// individual edge invocation's wall-clock budget.

import {
  binanceDepthDiffAdapter,
  binanceDepthSnapshotAdapter,
  BINANCE_VENUE,
} from "../../supabase/functions/_shared/l2_book.ts";
import {
  parseBinanceCombinedDepthRaw,
  parseBinanceDepthSnapshotRaw,
} from "../../supabase/functions/_shared/binance_l2_wire.ts";
import {
  L2CaptureSession,
  type DiffCaptureResult,
  type SnapshotCaptureResult,
} from "../../supabase/functions/_shared/l2_capture_session.ts";
import type { RawL2Item } from "../../supabase/functions/_shared/l2_raw_segment.ts";
import {
  SupabaseL2CaptureSink,
  type PendingNormalizedCapture,
} from "./storage.ts";

const DEFAULT_WS_BASE = "wss://stream.binance.com:9443";
const DEFAULT_REST_BASE = "https://api.binance.com";
const DEFAULT_SNAPSHOT_LIMIT = 1000;
const DEFAULT_BATCH_MESSAGES = 200;
const DEFAULT_FLUSH_MS = 2_000;
const SNAPSHOT_RETRY_MIN_MS = 250;
const SNAPSHOT_RETRY_MAX_MS = 5_000;
const RECONNECT_MIN_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;
const PROACTIVE_CONNECTION_ROTATION_MS = 23 * 60 * 60 * 1000 + 50 * 60 * 1000;

export interface L2WorkerConfig {
  supabaseUrl: string;
  serviceRoleKey: string;
  symbols: string[];
  wsBase: string;
  restBase: string;
  snapshotLimit: number;
  batchMessages: number;
  flushMs: number;
}

class FatalCaptureError extends Error {
  override name = "FatalCaptureError";
}

function requiredEnv(name: string): string {
  const value = Deno.env.get(name)?.trim() ?? "";
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function positiveIntEnv(name: string, fallback: number): number {
  const raw = Deno.env.get(name)?.trim();
  if (!raw) return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`);
  return value;
}

export function parseSymbols(raw: string): string[] {
  const symbols = raw.split(",").map((value) => value.trim().toUpperCase()).filter(Boolean);
  const unique = [...new Set(symbols)];
  if (!unique.length) throw new Error("BRIAN_L2_SYMBOLS must contain at least one symbol");
  if (unique.length !== symbols.length) throw new Error("BRIAN_L2_SYMBOLS contains duplicates");
  if (unique.length > 1024) throw new Error("BRIAN_L2_SYMBOLS exceeds Binance's 1024-stream connection limit");
  for (const symbol of unique) {
    if (!/^[A-Z0-9]+$/.test(symbol)) throw new Error(`invalid Binance symbol: ${symbol}`);
  }
  return unique.sort();
}

export function loadConfigFromEnv(): L2WorkerConfig {
  const snapshotLimit = positiveIntEnv("BRIAN_L2_SNAPSHOT_LIMIT", DEFAULT_SNAPSHOT_LIMIT);
  if (![100, 500, 1000, 5000].includes(snapshotLimit)) {
    throw new Error("BRIAN_L2_SNAPSHOT_LIMIT must be one of 100, 500, 1000, 5000");
  }
  return {
    supabaseUrl: requiredEnv("SUPABASE_URL"),
    serviceRoleKey: requiredEnv("SUPABASE_SERVICE_ROLE_KEY"),
    symbols: parseSymbols(requiredEnv("BRIAN_L2_SYMBOLS")),
    wsBase: (Deno.env.get("BINANCE_L2_WS_BASE") ?? DEFAULT_WS_BASE).replace(/\/$/, ""),
    restBase: (Deno.env.get("BINANCE_L2_REST_BASE") ?? DEFAULT_REST_BASE).replace(/\/$/, ""),
    snapshotLimit,
    batchMessages: positiveIntEnv("BRIAN_L2_BATCH_MESSAGES", DEFAULT_BATCH_MESSAGES),
    flushMs: positiveIntEnv("BRIAN_L2_FLUSH_MS", DEFAULT_FLUSH_MS),
  };
}

export function combinedDepthWsUrl(wsBase: string, symbols: string[]): string {
  const streams = symbols.map((symbol) => `${symbol.toLowerCase()}@depth@100ms`).join("/");
  return `${wsBase}/stream?streams=${streams}`;
}

export function depthSnapshotUrl(restBase: string, symbol: string, limit: number): string {
  const query = new URLSearchParams({ symbol, limit: String(limit) });
  return `${restBase}/api/v3/depth?${query.toString()}`;
}

function delay(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(new DOMException("Aborted", "AbortError"));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  });
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function currentStateRaw(
  session: L2CaptureSession,
  connectionGeneration: number,
  symbol: string | null,
  arrivalSeq: number,
  collectorReceivedAt: string,
  rawJson: string,
  source: RawL2Item["source"],
): RawL2Item {
  const state = symbol ? session.state(symbol) : null;
  return {
    collectorSessionId: session.collectorSessionId,
    arrivalSeq,
    connectionGeneration,
    syncGeneration: state?.syncGeneration ?? null,
    source,
    symbol,
    collectorReceivedAt,
    rawJson,
  };
}

function fromDiffResult(rawJson: string, result: DiffCaptureResult): RawL2Item {
  return {
    collectorSessionId: result.envelope.collectorSessionId,
    arrivalSeq: result.envelope.arrivalSeq,
    connectionGeneration: result.envelope.connectionGeneration,
    syncGeneration: result.envelope.syncGeneration,
    source: "binance_spot_diff_depth_ws",
    symbol: result.envelope.event.symbol,
    collectorReceivedAt: result.envelope.event.collectorReceivedAt,
    rawJson,
  };
}

function fromSnapshotResult(rawJson: string, result: SnapshotCaptureResult): RawL2Item {
  return {
    collectorSessionId: result.envelope.collectorSessionId,
    arrivalSeq: result.envelope.arrivalSeq,
    connectionGeneration: result.envelope.connectionGeneration,
    syncGeneration: result.envelope.syncGeneration,
    source: "binance_spot_rest_depth_snapshot",
    symbol: result.envelope.event.symbol,
    collectorReceivedAt: result.envelope.event.collectorReceivedAt,
    rawJson,
  };
}

async function openWebSocket(url: string, signal: AbortSignal): Promise<WebSocket> {
  const ws = new WebSocket(url);
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      try { ws.close(); } catch { /* ignored */ }
      reject(new Error("Binance L2 WebSocket open timeout"));
    }, 15_000);
    const abort = () => {
      clearTimeout(timeout);
      try { ws.close(); } catch { /* ignored */ }
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal.addEventListener("abort", abort, { once: true });
    ws.addEventListener("open", () => {
      clearTimeout(timeout);
      signal.removeEventListener("abort", abort);
      resolve();
    }, { once: true });
    ws.addEventListener("error", () => {
      clearTimeout(timeout);
      signal.removeEventListener("abort", abort);
      reject(new Error("Binance L2 WebSocket failed to open"));
    }, { once: true });
  });
  return ws;
}

export class BrianRealL2Worker {
  readonly config: L2WorkerConfig;
  readonly session: L2CaptureSession;
  readonly sink: SupabaseL2CaptureSink;
  private stopped = false;
  private activeSocket: WebSocket | null = null;
  private firstConnection = true;

  constructor(config: L2WorkerConfig) {
    this.config = config;
    this.session = new L2CaptureSession(crypto.randomUUID(), config.symbols);
    this.sink = new SupabaseL2CaptureSink(config.supabaseUrl, config.serviceRoleKey, {
      maxBatchMessages: config.batchMessages,
    });
  }

  stop(): void {
    this.stopped = true;
    try { this.activeSocket?.close(1000, "Brian L2 worker stopping"); } catch { /* ignored */ }
  }

  async run(): Promise<void> {
    let reconnectDelay = RECONNECT_MIN_MS;
    while (!this.stopped) {
      try {
        await this.runOneConnection();
        reconnectDelay = RECONNECT_MIN_MS;
      } catch (error) {
        if (this.stopped || isAbort(error)) break;
        if (error instanceof FatalCaptureError) throw error;
        console.error("Brian L2 transport cycle failed; reconnecting", error);
        await delay(reconnectDelay);
        reconnectDelay = Math.min(RECONNECT_MAX_MS, reconnectDelay * 2);
      }
    }

    await this.sink.flush();
    const connectionGeneration = Math.max(1, this.session.state(this.config.symbols[0]).connectionGeneration);
    await this.sink.recordSessionEvent({
      collectorSessionId: this.session.collectorSessionId,
      eventKind: "STOPPED",
      venue: BINANCE_VENUE,
      connectionGeneration,
      arrivalSeqBoundary: this.session.currentArrivalSeq(),
      details: { reason: "worker_stop", shadow_only: true },
    });
  }

  private async runOneConnection(): Promise<void> {
    const connectionGeneration = this.session.startConnection();
    if (this.firstConnection) {
      await this.sink.recordSessionEvent({
        collectorSessionId: this.session.collectorSessionId,
        eventKind: "STARTED",
        venue: BINANCE_VENUE,
        connectionGeneration,
        arrivalSeqBoundary: this.session.currentArrivalSeq(),
        details: { symbols: this.config.symbols, snapshot_limit: this.config.snapshotLimit },
      });
      this.firstConnection = false;
    }

    const connectionAbort = new AbortController();
    const ws = await openWebSocket(combinedDepthWsUrl(this.config.wsBase, this.config.symbols), connectionAbort.signal);
    this.activeSocket = ws;

    let fatal: FatalCaptureError | null = null;
    let processing: Promise<void> = Promise.resolve();
    const snapshotLoops = new Map<string, Promise<void>>();

    const closeForFatal = (error: unknown) => {
      if (!fatal) fatal = new FatalCaptureError(asError(error).message, { cause: error });
      try { ws.close(1011, "Brian L2 fail-closed"); } catch { /* ignored */ }
    };

    const enqueue = <T>(work: () => Promise<T> | T): Promise<T> => {
      const task = processing.then(async () => {
        if (fatal) throw fatal;
        return await work();
      });
      processing = task.then(
        () => undefined,
        (error) => {
          closeForFatal(error);
        },
      );
      return task;
    };

    const ensureSnapshotLoop = (symbol: string) => {
      if (snapshotLoops.has(symbol) || connectionAbort.signal.aborted) return;
      const loop = this.snapshotLoop(symbol, connectionGeneration, connectionAbort.signal, enqueue)
        .catch((error) => {
          if (!isAbort(error)) closeForFatal(error);
        })
        .finally(() => snapshotLoops.delete(symbol));
      snapshotLoops.set(symbol, loop);
    };

    ws.addEventListener("message", (message) => {
      if (typeof message.data !== "string") {
        closeForFatal(new Error("Binance L2 WebSocket delivered non-text payload"));
        return;
      }
      const rawJson = message.data;
      const collectorReceivedAt = new Date().toISOString();
      void enqueue(async () => {
        const result = await this.handleWsDepthRaw(
          rawJson,
          collectorReceivedAt,
          connectionGeneration,
        );
        if (result.disposition === "gap_resync") ensureSnapshotLoop(result.envelope.event.symbol);
      }).catch(() => undefined);
    });

    await this.sink.recordSessionEvent({
      collectorSessionId: this.session.collectorSessionId,
      eventKind: "WS_CONNECTED",
      venue: BINANCE_VENUE,
      connectionGeneration,
      arrivalSeqBoundary: this.session.currentArrivalSeq(),
      details: { endpoint: this.config.wsBase, streams: this.config.symbols.length },
    });

    for (const symbol of this.config.symbols) {
      const state = this.session.state(symbol);
      await this.sink.recordSessionEvent({
        collectorSessionId: this.session.collectorSessionId,
        eventKind: "SYNC_STARTED",
        venue: BINANCE_VENUE,
        symbol,
        connectionGeneration,
        syncGeneration: state.syncGeneration,
        arrivalSeqBoundary: this.session.currentArrivalSeq(),
      });
      ensureSnapshotLoop(symbol);
    }

    const periodicFlush = setInterval(() => {
      void enqueue(() => this.sink.flush()).catch(() => undefined);
    }, this.config.flushMs);
    const rotate = setTimeout(() => {
      try { ws.close(1000, "proactive 24h Binance connection rotation"); } catch { /* ignored */ }
    }, PROACTIVE_CONNECTION_ROTATION_MS);

    const closeInfo = await new Promise<{ code: number; reason: string }>((resolve) => {
      ws.addEventListener("close", (event) => resolve({ code: event.code, reason: event.reason }), { once: true });
      ws.addEventListener("error", () => {
        try { ws.close(); } catch { /* ignored */ }
      });
    });

    clearInterval(periodicFlush);
    clearTimeout(rotate);
    connectionAbort.abort();
    await Promise.allSettled([...snapshotLoops.values()]);
    await processing;

    try {
      await this.sink.flush();
      await this.sink.recordSessionEvent({
        collectorSessionId: this.session.collectorSessionId,
        eventKind: "WS_DISCONNECTED",
        venue: BINANCE_VENUE,
        connectionGeneration,
        arrivalSeqBoundary: this.session.currentArrivalSeq(),
        details: { code: closeInfo.code, reason: closeInfo.reason },
      });
    } catch (error) {
      closeForFatal(error);
    } finally {
      this.activeSocket = null;
    }

    if (fatal) throw fatal;
    if (!this.stopped) throw new Error(`Binance L2 WebSocket closed (${closeInfo.code}: ${closeInfo.reason})`);
  }

  private async handleWsDepthRaw(
    rawJson: string,
    collectorReceivedAt: string,
    connectionGeneration: number,
  ): Promise<DiffCaptureResult> {
    // Authoritative order is reserved at the raw boundary, before parsing/normalization.
    const arrivalSeq = this.session.reserveArrivalSeq();
    let symbol: string | null = null;
    try {
      const wire = parseBinanceCombinedDepthRaw(rawJson);
      symbol = wire.data.s.toUpperCase();
      const stateBefore = this.session.state(symbol); // proves configured-symbol lineage
      const ingestAt = new Date().toISOString();
      const normalized = binanceDepthDiffAdapter.normalize(wire.data, { collectorReceivedAt, ingestAt });
      if (!normalized.ok) throw new Error(`Binance depth diff normalization failed: ${normalized.reason}`);

      const result = this.session.acceptDepthDiff(normalized.event, arrivalSeq);
      const pending: PendingNormalizedCapture = {
        envelope: result.envelope,
        transport: "binance_spot_diff_depth_ws",
        disposition: result.disposition,
      };
      this.sink.append({ raw: fromDiffResult(rawJson, result), normalized: pending });

      if (result.disposition === "gap_resync") {
        // Persist the gap-revealing raw+normalized event before publishing lifecycle state that
        // refers to its sequence boundary.
        await this.sink.flush();
        await this.sink.recordSessionEvent({
          collectorSessionId: this.session.collectorSessionId,
          eventKind: "GAP_INVALIDATED",
          venue: BINANCE_VENUE,
          symbol,
          connectionGeneration,
          syncGeneration: result.previousSyncGeneration,
          arrivalSeqBoundary: arrivalSeq,
          details: {
            first_update_id: normalized.event.firstUpdateId,
            final_update_id: normalized.event.finalUpdateId,
          },
        });
        await this.sink.recordSessionEvent({
          collectorSessionId: this.session.collectorSessionId,
          eventKind: "RESYNC_STARTED",
          venue: BINANCE_VENUE,
          symbol,
          connectionGeneration,
          syncGeneration: result.envelope.syncGeneration,
          arrivalSeqBoundary: arrivalSeq,
          details: { reason: "sequence_gap" },
        });
      } else if (this.sink.shouldFlush()) {
        await this.sink.flush();
      }

      // keep stateBefore read above intentionally: it makes unknown-symbol rejection occur before
      // adapter/state mutation, while the raw arrival itself still has an audit sequence.
      void stateBefore;
      return result;
    } catch (error) {
      // A malformed/unknown raw market-data message must still be retained before failing closed.
      let raw: RawL2Item;
      try {
        raw = currentStateRaw(
          this.session,
          connectionGeneration,
          symbol,
          arrivalSeq,
          collectorReceivedAt,
          rawJson,
          "binance_spot_diff_depth_ws",
        );
      } catch {
        raw = {
          collectorSessionId: this.session.collectorSessionId,
          arrivalSeq,
          connectionGeneration,
          syncGeneration: null,
          source: "binance_spot_diff_depth_ws",
          symbol,
          collectorReceivedAt,
          rawJson,
        };
      }
      this.sink.append({ raw, normalized: null });
      await this.sink.flush();
      throw new FatalCaptureError(`invalid Binance L2 wire event at arrival_seq ${arrivalSeq}: ${asError(error).message}`, { cause: error });
    }
  }

  private async snapshotLoop<T>(
    symbol: string,
    connectionGeneration: number,
    signal: AbortSignal,
    enqueue: <R>(work: () => Promise<R> | R) => Promise<R>,
  ): Promise<void> {
    let retryMs = SNAPSHOT_RETRY_MIN_MS;
    while (!signal.aborted && !this.stopped) {
      const state = this.session.state(symbol);
      if (state.connectionGeneration !== connectionGeneration || state.status !== "SYNCING") return;

      let response: Response;
      try {
        response = await fetch(depthSnapshotUrl(this.config.restBase, symbol, this.config.snapshotLimit), {
          headers: { "accept": "application/json", "user-agent": "Brian-2026-L2-Shadow/1.0" },
          signal,
        });
      } catch (error) {
        if (isAbort(error)) throw error;
        await delay(retryMs, signal);
        retryMs = Math.min(SNAPSHOT_RETRY_MAX_MS, retryMs * 2);
        continue;
      }
      if (!response.ok) {
        await response.body?.cancel();
        await delay(retryMs, signal);
        retryMs = Math.min(SNAPSHOT_RETRY_MAX_MS, retryMs * 2);
        continue;
      }

      const rawJson = await response.text();
      const collectorReceivedAt = new Date().toISOString();
      const result = await enqueue(() => this.handleSnapshotRaw(
        symbol,
        rawJson,
        collectorReceivedAt,
        connectionGeneration,
      ));
      if (result.disposition === "synced") return;
      retryMs = result.disposition === "snapshot_too_old"
        ? SNAPSHOT_RETRY_MIN_MS
        : Math.min(SNAPSHOT_RETRY_MAX_MS, Math.max(500, retryMs * 2));
      await delay(retryMs, signal);
    }
  }

  private async handleSnapshotRaw(
    symbol: string,
    rawJson: string,
    collectorReceivedAt: string,
    connectionGeneration: number,
  ): Promise<SnapshotCaptureResult> {
    const arrivalSeq = this.session.reserveArrivalSeq();
    try {
      const wire = parseBinanceDepthSnapshotRaw(rawJson);
      const ingestAt = new Date().toISOString();
      const normalized = binanceDepthSnapshotAdapter.normalize(wire, {
        collectorReceivedAt,
        ingestAt,
        symbol,
      });
      if (!normalized.ok) throw new Error(`Binance depth snapshot normalization failed: ${normalized.reason}`);

      const result = this.session.acceptDepthSnapshot(normalized.event, arrivalSeq);
      const pending: PendingNormalizedCapture = {
        envelope: result.envelope,
        transport: "binance_spot_rest_depth_snapshot",
        disposition: result.disposition,
      };
      this.sink.append({ raw: fromSnapshotResult(rawJson, result), normalized: pending });
      // Snapshot transitions are rare and important: persist their source truth before the
      // lifecycle event that announces the transition.
      await this.sink.flush();

      if (result.disposition === "synced") {
        await this.sink.recordSessionEvent({
          collectorSessionId: this.session.collectorSessionId,
          eventKind: "SYNCED",
          venue: BINANCE_VENUE,
          symbol,
          connectionGeneration,
          syncGeneration: result.envelope.syncGeneration,
          arrivalSeqBoundary: arrivalSeq,
          details: {
            last_applied_update_id: this.session.state(symbol).lastAppliedUpdateId,
            issues: result.issues,
          },
        });
      } else if (result.disposition === "snapshot_too_old") {
        await this.sink.recordSessionEvent({
          collectorSessionId: this.session.collectorSessionId,
          eventKind: "SNAPSHOT_TOO_OLD",
          venue: BINANCE_VENUE,
          symbol,
          connectionGeneration,
          syncGeneration: result.envelope.syncGeneration,
          arrivalSeqBoundary: arrivalSeq,
          details: { issues: result.issues },
        });
      } else {
        const state = this.session.state(symbol);
        await this.sink.recordSessionEvent({
          collectorSessionId: this.session.collectorSessionId,
          eventKind: "RESYNC_STARTED",
          venue: BINANCE_VENUE,
          symbol,
          connectionGeneration,
          syncGeneration: state.syncGeneration,
          arrivalSeqBoundary: arrivalSeq,
          details: { reason: "invalid_snapshot_or_replay", issues: result.issues },
        });
      }
      return result;
    } catch (error) {
      const raw = currentStateRaw(
        this.session,
        connectionGeneration,
        symbol,
        arrivalSeq,
        collectorReceivedAt,
        rawJson,
        "binance_spot_rest_depth_snapshot",
      );
      this.sink.append({ raw, normalized: null });
      await this.sink.flush();
      throw new FatalCaptureError(`invalid Binance L2 snapshot at arrival_seq ${arrivalSeq}: ${asError(error).message}`, { cause: error });
    }
  }
}

async function main(): Promise<void> {
  const worker = new BrianRealL2Worker(loadConfigFromEnv());
  const stop = () => worker.stop();
  try { Deno.addSignalListener("SIGINT", stop); } catch { /* platform may not expose signal */ }
  try { Deno.addSignalListener("SIGTERM", stop); } catch { /* platform may not expose signal */ }

  try {
    await worker.run();
  } catch (error) {
    console.error("Brian real L2 capture terminated fail-closed", error);
    // Do not silently restart after a data-integrity/persistence failure. An external supervisor
    // may start a brand-new collector session after the operator sees the failure.
    Deno.exitCode = 1;
  }
}

if (import.meta.main) await main();
