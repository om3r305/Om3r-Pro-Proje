// Brian 2026 real-L2 capture session semantics.
//
// Infrastructure only. Raw transport arrivals receive their authoritative total-order sequence
// *before* normalization/interpretation. The normalized event then consumes that reservation and
// is assigned to the snapshot/resync generation determined by the state transition. No trading
// signal, action, target, order, or promotion decision exists here.

import {
  BINANCE_VENUE,
  synchronizeDepthBookStartup,
  type DepthDiffEvent,
  type DepthSnapshotEvent,
} from "./l2_book.ts";

export type CaptureSyncStatus = "SYNCING" | "SYNCED";
export type DiffDisposition = "buffered" | "applied" | "stale" | "gap_resync";
export type SnapshotDisposition = "synced" | "snapshot_too_old" | "invalid_resync";

export interface CaptureEnvelope<TEvent extends DepthDiffEvent | DepthSnapshotEvent> {
  collectorSessionId: string;
  arrivalSeq: number;
  connectionGeneration: number;
  syncGeneration: number;
  event: TEvent;
}

export interface SymbolCaptureStateView {
  symbol: string;
  connectionGeneration: number;
  syncGeneration: number;
  status: CaptureSyncStatus;
  lastAppliedUpdateId: string | null;
  bufferedDiffCount: number;
}

interface BufferedDiff {
  arrivalSeq: number;
  event: DepthDiffEvent;
}

interface SymbolCaptureState {
  symbol: string;
  connectionGeneration: number;
  syncGeneration: number;
  status: CaptureSyncStatus;
  lastAppliedUpdateId: string | null;
  bufferedDiffs: BufferedDiff[];
}

export interface DiffCaptureResult {
  envelope: CaptureEnvelope<DepthDiffEvent>;
  disposition: DiffDisposition;
  previousSyncGeneration?: number;
}

export interface SnapshotCaptureResult {
  envelope: CaptureEnvelope<DepthSnapshotEvent>;
  disposition: SnapshotDisposition;
  issues: string[];
}

function normalizeSymbol(value: string): string {
  const symbol = value.trim().toUpperCase();
  if (!symbol) throw new Error("symbol is required");
  return symbol;
}

function validateSessionId(value: string): string {
  const out = value.trim();
  if (!out) throw new Error("collectorSessionId is required");
  return out;
}

/** Sequence-only classifier matching the documented Binance local-book rule used by l2_book.ts. */
export function classifyDepthDiffSequence(
  lastAppliedUpdateId: string,
  diff: Pick<DepthDiffEvent, "firstUpdateId" | "finalUpdateId">,
): "stale" | "gap" | "apply" {
  const last = BigInt(lastAppliedUpdateId);
  const first = BigInt(diff.firstUpdateId);
  const final = BigInt(diff.finalUpdateId);
  if (final <= last) return "stale";
  if (first > last + 1n) return "gap";
  return "apply";
}

export class L2CaptureSession {
  readonly collectorSessionId: string;
  private reservedArrivalSeq = 0;
  private consumedArrivalSeq = 0;
  private connectionGeneration = 0;
  private readonly states = new Map<string, SymbolCaptureState>();

  constructor(collectorSessionId: string, symbols: string[]) {
    this.collectorSessionId = validateSessionId(collectorSessionId);
    if (!symbols.length) throw new Error("at least one L2 symbol is required");
    for (const input of symbols) {
      const symbol = normalizeSymbol(input);
      if (this.states.has(symbol)) throw new Error(`duplicate L2 symbol: ${symbol}`);
      this.states.set(symbol, {
        symbol,
        connectionGeneration: 0,
        syncGeneration: 0,
        status: "SYNCING",
        lastAppliedUpdateId: null,
        bufferedDiffs: [],
      });
    }
  }

  /** A transport reconnect invalidates every local sequence baseline. Each symbol starts a new
   * sync generation and must obtain a fresh REST snapshot before being considered SYNCED. */
  startConnection(): number {
    this.connectionGeneration += 1;
    for (const state of this.states.values()) {
      state.connectionGeneration = this.connectionGeneration;
      state.syncGeneration += 1;
      state.status = "SYNCING";
      state.lastAppliedUpdateId = null;
      state.bufferedDiffs = [];
    }
    return this.connectionGeneration;
  }

  /** Reserve the authoritative total-order slot at the raw transport boundary, before JSON
   * normalization or any market interpretation happens. */
  reserveArrivalSeq(): number {
    this.reservedArrivalSeq += 1;
    if (!Number.isSafeInteger(this.reservedArrivalSeq)) {
      throw new Error("arrival sequence exhausted safe integer range");
    }
    return this.reservedArrivalSeq;
  }

  currentArrivalSeq(): number {
    return this.reservedArrivalSeq;
  }

  currentConsumedArrivalSeq(): number {
    return this.consumedArrivalSeq;
  }

  symbols(): string[] {
    return [...this.states.keys()].sort();
  }

  state(symbolInput: string): SymbolCaptureStateView {
    const state = this.requireState(symbolInput);
    return {
      symbol: state.symbol,
      connectionGeneration: state.connectionGeneration,
      syncGeneration: state.syncGeneration,
      status: state.status,
      lastAppliedUpdateId: state.lastAppliedUpdateId,
      bufferedDiffCount: state.bufferedDiffs.length,
    };
  }

  acceptDepthDiff(event: DepthDiffEvent, arrivalSeq: number): DiffCaptureResult {
    this.assertBinanceDepthEvent(event);
    const state = this.requireState(event.symbol);
    this.requireStartedConnection(state);
    this.consumeReservedArrival(arrivalSeq);

    if (state.status === "SYNCING") {
      state.bufferedDiffs.push({ arrivalSeq, event });
      return {
        envelope: this.envelope(state, arrivalSeq, event),
        disposition: "buffered",
      };
    }

    const classification = classifyDepthDiffSequence(state.lastAppliedUpdateId!, event);
    if (classification === "stale") {
      return {
        envelope: this.envelope(state, arrivalSeq, event),
        disposition: "stale",
      };
    }

    if (classification === "gap") {
      const previousSyncGeneration = state.syncGeneration;
      state.syncGeneration += 1;
      state.status = "SYNCING";
      state.lastAppliedUpdateId = null;
      // The gap-revealing message belongs to and becomes the first buffered event of the new
      // resync generation. A REST snapshot that lags it must therefore fail `snapshot_too_old`
      // rather than falsely declaring a clean sync with an empty buffer.
      state.bufferedDiffs = [{ arrivalSeq, event }];
      return {
        envelope: this.envelope(state, arrivalSeq, event),
        disposition: "gap_resync",
        previousSyncGeneration,
      };
    }

    state.lastAppliedUpdateId = event.finalUpdateId;
    return {
      envelope: this.envelope(state, arrivalSeq, event),
      disposition: "applied",
    };
  }

  acceptDepthSnapshot(event: DepthSnapshotEvent, arrivalSeq: number): SnapshotCaptureResult {
    this.assertBinanceDepthEvent(event);
    const state = this.requireState(event.symbol);
    this.requireStartedConnection(state);
    if (state.status !== "SYNCING") {
      throw new Error(`snapshot for ${state.symbol} received while capture state is already SYNCED`);
    }
    this.consumeReservedArrival(arrivalSeq);

    const envelope = this.envelope(state, arrivalSeq, event);
    const buffered = [...state.bufferedDiffs]
      .sort((a, b) => a.arrivalSeq - b.arrivalSeq)
      .map((item) => item.event);
    const sync = synchronizeDepthBookStartup(buffered, event);
    const issues = sync.issues.map((issue) => `${issue.kind}:${issue.reason}`);

    if (sync.outcome === "synced" && sync.state?.lastAppliedUpdateId != null) {
      state.status = "SYNCED";
      state.lastAppliedUpdateId = sync.state.lastAppliedUpdateId.toString();
      state.bufferedDiffs = [];
      return { envelope, disposition: "synced", issues };
    }

    if (sync.outcome === "snapshot_too_old") {
      // Keep the existing buffered diffs in the same generation. A newer snapshot can still
      // bridge them; clearing the buffer here would erase the evidence needed to prove it.
      return { envelope, disposition: "snapshot_too_old", issues };
    }

    // The generation itself is untrustworthy (real gap, crossed snapshot/book, lineage problem).
    // Start a clean generation. Subsequent WS diffs buffer while the worker fetches a fresh
    // snapshot; the failed snapshot and old buffered events remain persisted for forensics.
    state.syncGeneration += 1;
    state.status = "SYNCING";
    state.lastAppliedUpdateId = null;
    state.bufferedDiffs = [];
    return { envelope, disposition: "invalid_resync", issues };
  }

  private consumeReservedArrival(arrivalSeq: number): void {
    if (!Number.isSafeInteger(arrivalSeq) || arrivalSeq <= 0) {
      throw new Error("arrivalSeq must be a positive safe integer reservation");
    }
    if (arrivalSeq > this.reservedArrivalSeq) {
      throw new Error(`arrivalSeq ${arrivalSeq} was not reserved`);
    }
    const expected = this.consumedArrivalSeq + 1;
    if (arrivalSeq !== expected) {
      throw new Error(`arrivalSeq must be consumed in strict order; expected ${expected}, got ${arrivalSeq}`);
    }
    this.consumedArrivalSeq = arrivalSeq;
  }

  private envelope<TEvent extends DepthDiffEvent | DepthSnapshotEvent>(
    state: SymbolCaptureState,
    arrivalSeq: number,
    event: TEvent,
  ): CaptureEnvelope<TEvent> {
    return {
      collectorSessionId: this.collectorSessionId,
      arrivalSeq,
      connectionGeneration: state.connectionGeneration,
      syncGeneration: state.syncGeneration,
      event,
    };
  }

  private requireState(symbolInput: string): SymbolCaptureState {
    const symbol = normalizeSymbol(symbolInput);
    const state = this.states.get(symbol);
    if (!state) throw new Error(`symbol is not part of this L2 capture session: ${symbol}`);
    return state;
  }

  private requireStartedConnection(state: SymbolCaptureState): void {
    if (state.connectionGeneration <= 0 || state.syncGeneration <= 0) {
      throw new Error(`startConnection() must be called before accepting L2 events for ${state.symbol}`);
    }
  }

  private assertBinanceDepthEvent(event: DepthDiffEvent | DepthSnapshotEvent): void {
    if (event.venue !== BINANCE_VENUE) {
      throw new Error(`real L2 capture currently supports Binance only, got ${event.venue}`);
    }
  }
}
