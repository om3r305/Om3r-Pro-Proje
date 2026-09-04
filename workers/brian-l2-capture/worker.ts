import {
  binanceDepthDiffAdapter,
  binanceDepthSnapshotAdapter,
  BINANCE_VENUE,
} from "../../supabase/functions/_shared/l2_book.ts";
import {
  isBinanceServerShutdownRaw,
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
  PROACTIVE_CONNECTION_ROTATION_MS,
  RECONNECT_MAX_MS,
  RECONNECT_MIN_MS,
  SNAPSHOT_RETRY_MAX_MS,
  SNAPSHOT_RETRY_MIN_MS,
  combinedDepthWsUrl,
  depthSnapshotUrl,
  type L2WorkerConfig,
} from "./config.ts";
import {
  SupabaseL2CaptureSink,
  type PendingNormalizedCapture,
} from "./storage.ts";

export class FatalCaptureError extends Error {
  override name = "FatalCaptureError";
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
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
      if (ws.readyState !== WebSocket.OPEN) {
        clearTimeout(timeout);
        signal.removeEventListener("abort", abort);
        reject(new Error("Binance L2 WebSocket failed to open"));
      }
    });
  });
  return ws;
}

function rawFromCurrentState(
  session: L2CaptureSession,
  connectionGeneration: number,
  symbol: string | null,
  arrivalSeq: number,
  collectorReceivedAt: string,
  rawJson: string,
  source: RawL2Item["source"],
): RawL2Item {
  let syncGeneration: number | null = null;
  if (symbol) {
    try { syncGeneration = session.state(symbol).syncGeneration; } catch { /* unknown lineage */ }
  }
  return {
    collectorSessionId: session.collectorSessionId,
    arrivalSeq,
    connectionGeneration,
    syncGeneration,
    source,
    symbol,
    collectorReceivedAt,
    rawJson,
  };
}

function rawFromDiff(rawJson: string, result: DiffCaptureResult): RawL2Item {
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

function rawFromSnapshot(rawJson: string, result: SnapshotCaptureResult): RawL2Item {
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
    let ws: WebSocket;
    try {
      ws = await openWebSocket(
        combinedDepthWsUrl(this.config.wsBase, this.config.symbols),
        connectionAbort.signal,
      );
    } catch (error) {
      connectionAbort.abort();
      throw error;
    }
    this.activeSocket = ws;

    const closePromise = new Promise<{ code: number; reason: string }>((resolve) => {
      ws.addEventListener("close", (event) => resolve({ code: event.code, reason: event.reason }), { once: true });
    });

    let fatal: FatalCaptureError | null = null;
    const readFatal = (): FatalCaptureError | null => fatal;
    let processing: Promise<void> = Promise.resolve();
    const snapshotLoops = new Map<string, Promise<void>>();

    const markFatal = (error: unknown) => {
      if (!fatal) fatal = new FatalCaptureError(asError(error).message, { cause: error });
      try { ws.close(1011, "Brian L2 fail-closed"); } catch { /* ignored */ }
    };

    const enqueue = <T>(work: () => Promise<T> | T): Promise<T> => {
      const task = processing.then(async () => {
        if (fatal) throw fatal;
        return await work();
      });
      processing = task.then(() => undefined, (error) => markFatal(error));
      return task;
    };

    const ensureSnapshotLoop = (symbol: string) => {
      if (snapshotLoops.has(symbol) || connectionAbort.signal.aborted) return;
      const loop = this.snapshotLoop(symbol, connectionGeneration, connectionAbort.signal, enqueue)
        .catch((error) => {
          if (!isAbort(error)) markFatal(error);
        })
        .finally(() => snapshotLoops.delete(symbol));
      snapshotLoops.set(symbol, loop);
    };

    ws.addEventListener("message", (message) => {
      if (typeof message.data !== "string") {
        markFatal(new Error("Binance L2 WebSocket delivered non-text payload"));
        return;
      }
      const rawJson = message.data;
      if (isBinanceServerShutdownRaw(rawJson)) {
        try { ws.close(1000, "Binance serverShutdown"); } catch { /* ignored */ }
        return;
      }
      const collectorReceivedAt = new Date().toISOString();
      let arrivalSeq: number;
      try {
        // Reserve the authoritative order at the raw transport callback, before JSON parsing,
        // normalization, persistence, or any book-state interpretation.
        arrivalSeq = this.session.reserveArrivalSeq();
      } catch (error) {
        markFatal(error);
        return;
      }
      void enqueue(async () => {
        const result = await this.handleWsDepthRaw(
          rawJson,
          collectorReceivedAt,
          connectionGeneration,
          arrivalSeq,
        );
        if (result.disposition === "gap_resync") ensureSnapshotLoop(result.envelope.event.symbol);
      }).catch(() => undefined);
    });
    ws.addEventListener("error", () => {
      try { ws.close(); } catch { /* ignored */ }
    });

    try {
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
    } catch (error) {
      markFatal(error);
    }

    const periodicFlush = setInterval(() => {
      void enqueue(() => this.sink.flush()).catch(() => undefined);
    }, this.config.flushMs);
    const rotate = setTimeout(() => {
      try { ws.close(1000, "proactive 24h Binance connection rotation"); } catch { /* ignored */ }
    }, PROACTIVE_CONNECTION_ROTATION_MS);

    const closeInfo = await closePromise;
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
      const fatalForLog = readFatal();
      if (fatalForLog) {
        await this.sink.recordSessionEvent({
          collectorSessionId: this.session.collectorSessionId,
          eventKind: "FAILED",
          venue: BINANCE_VENUE,
          connectionGeneration,
          arrivalSeqBoundary: this.session.currentArrivalSeq(),
          details: { error: fatalForLog.message },
        });
      }
    } catch (error) {
      if (!fatal) fatal = new FatalCaptureError(`L2 persistence/session finalization failed: ${asError(error).message}`, { cause: error });
    } finally {
      this.activeSocket = null;
    }

    const fatalAfterFinalize = readFatal();
    if (fatalAfterFinalize) throw fatalAfterFinalize;
    if (!this.stopped) {
      throw new Error(`Binance L2 WebSocket closed (${closeInfo.code}: ${closeInfo.reason})`);
    }
  }

  private async handleWsDepthRaw(
    rawJson: string,
    collectorReceivedAt: string,
    connectionGeneration: number,
    arrivalSeq: number,
  ): Promise<DiffCaptureResult> {
    let symbol: string | null = null;
    let result: DiffCaptureResult;

    try {
      const wire = parseBinanceCombinedDepthRaw(rawJson);
      symbol = wire.data.s.toUpperCase();
      this.session.state(symbol);
      const normalized = binanceDepthDiffAdapter.normalize(wire.data, {
        collectorReceivedAt,
        ingestAt: new Date().toISOString(),
      });
      if (!normalized.ok) throw new Error(`Binance depth diff normalization failed: ${normalized.reason}`);
      result = this.session.acceptDepthDiff(normalized.event, arrivalSeq);
    } catch (error) {
      this.sink.append({
        raw: rawFromCurrentState(
          this.session,
          connectionGeneration,
          symbol,
          arrivalSeq,
          collectorReceivedAt,
          rawJson,
          "binance_spot_diff_depth_ws",
        ),
        normalized: null,
      });
      await this.sink.flush();
      throw new FatalCaptureError(
        `invalid Binance L2 wire event at arrival_seq ${arrivalSeq}: ${asError(error).message}`,
        { cause: error },
      );
    }

    const pending: PendingNormalizedCapture = {
      envelope: result.envelope,
      transport: "binance_spot_diff_depth_ws",
      disposition: result.disposition,
    };
    this.sink.append({ raw: rawFromDiff(rawJson, result), normalized: pending });

    if (result.disposition === "gap_resync") {
      await this.sink.flush();
      await this.sink.recordSessionEvent({
        collectorSessionId: this.session.collectorSessionId,
        eventKind: "GAP_INVALIDATED",
        venue: BINANCE_VENUE,
        symbol: result.envelope.event.symbol,
        connectionGeneration,
        syncGeneration: result.previousSyncGeneration,
        arrivalSeqBoundary: arrivalSeq,
        details: {
          first_update_id: result.envelope.event.firstUpdateId,
          final_update_id: result.envelope.event.finalUpdateId,
        },
      });
      await this.sink.recordSessionEvent({
        collectorSessionId: this.session.collectorSessionId,
        eventKind: "RESYNC_STARTED",
        venue: BINANCE_VENUE,
        symbol: result.envelope.event.symbol,
        connectionGeneration,
        syncGeneration: result.envelope.syncGeneration,
        arrivalSeqBoundary: arrivalSeq,
        details: { reason: "sequence_gap" },
      });
    } else if (this.sink.shouldFlush()) {
      await this.sink.flush();
    }

    return result;
  }

  private async snapshotLoop(
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
      // The completed REST snapshot response is also a raw transport arrival in the same global
      // collector order as WS messages. Reserve before enqueue/parse for deterministic replay.
      const arrivalSeq = this.session.reserveArrivalSeq();
      const result = await enqueue(() => this.handleSnapshotRaw(
        symbol,
        rawJson,
        collectorReceivedAt,
        connectionGeneration,
        arrivalSeq,
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
    arrivalSeq: number,
  ): Promise<SnapshotCaptureResult> {
    let result: SnapshotCaptureResult;

    try {
      const wire = parseBinanceDepthSnapshotRaw(rawJson);
      const normalized = binanceDepthSnapshotAdapter.normalize(wire, {
        collectorReceivedAt,
        ingestAt: new Date().toISOString(),
        symbol,
      });
      if (!normalized.ok) throw new Error(`Binance depth snapshot normalization failed: ${normalized.reason}`);
      result = this.session.acceptDepthSnapshot(normalized.event, arrivalSeq);
    } catch (error) {
      this.sink.append({
        raw: rawFromCurrentState(
          this.session,
          connectionGeneration,
          symbol,
          arrivalSeq,
          collectorReceivedAt,
          rawJson,
          "binance_spot_rest_depth_snapshot",
        ),
        normalized: null,
      });
      await this.sink.flush();
      throw new FatalCaptureError(
        `invalid Binance L2 snapshot at arrival_seq ${arrivalSeq}: ${asError(error).message}`,
        { cause: error },
      );
    }

    const pending: PendingNormalizedCapture = {
      envelope: result.envelope,
      transport: "binance_spot_rest_depth_snapshot",
      disposition: result.disposition,
    };
    this.sink.append({ raw: rawFromSnapshot(rawJson, result), normalized: pending });
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
  }
}
