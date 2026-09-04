import { assertEquals, assertThrows } from "jsr:@std/assert@^1.0.0";
import {
  classifyDepthDiffSequence,
  L2CaptureSession,
} from "./l2_capture_session.ts";
import {
  BINANCE_VENUE,
  EVIDENCE_CLASS,
  type DepthDiffEvent,
  type DepthSnapshotEvent,
} from "./l2_book.ts";

const T0 = "2026-09-04T10:00:00.000Z";

function meta(symbol: string) {
  return {
    venue: BINANCE_VENUE,
    symbol,
    exchangeEventAt: T0,
    collectorReceivedAt: T0,
    ingestAt: T0,
    ageMs: 0,
    clockSkewMs: 0,
    sourceLineage: { fixture: true },
    evidenceClass: EVIDENCE_CLASS,
    shadowOnly: true as const,
  };
}

function diff(symbol: string, first: string, final: string, bidPrice = "100", bidSize = "1"): DepthDiffEvent {
  return {
    ...meta(symbol),
    kind: "depth_diff",
    firstUpdateId: first,
    finalUpdateId: final,
    bidMutations: [{ price: bidPrice, size: bidSize }],
    askMutations: [],
  };
}

function snapshot(symbol: string, last: string): DepthSnapshotEvent {
  return {
    ...meta(symbol),
    kind: "depth_snapshot",
    exchangeEventAt: null,
    lastUpdateId: last,
    bids: [{ price: "99", size: "2" }],
    asks: [{ price: "101", size: "2" }],
  };
}

Deno.test("l2 capture: arrival_seq is a deterministic total order even when timestamps are identical", () => {
  const session = new L2CaptureSession("session-a", ["BTCUSDT"]);
  session.startConnection();
  const a = session.acceptDepthDiff(diff("BTCUSDT", "101", "102"));
  const b = session.acceptDepthDiff(diff("BTCUSDT", "103", "104"));
  const c = session.acceptDepthSnapshot(snapshot("BTCUSDT", "102"));
  assertEquals([a.envelope.arrivalSeq, b.envelope.arrivalSeq, c.envelope.arrivalSeq], [1, 2, 3]);
});

Deno.test("l2 capture: startup buffers WS diffs then synchronizes against a covering snapshot", () => {
  const session = new L2CaptureSession("session-a", ["BTCUSDT"]);
  assertEquals(session.startConnection(), 1);
  assertEquals(session.acceptDepthDiff(diff("BTCUSDT", "101", "102")).disposition, "buffered");
  assertEquals(session.acceptDepthDiff(diff("BTCUSDT", "103", "104")).disposition, "buffered");
  const synced = session.acceptDepthSnapshot(snapshot("BTCUSDT", "102"));
  assertEquals(synced.disposition, "synced");
  assertEquals(session.state("BTCUSDT"), {
    symbol: "BTCUSDT",
    connectionGeneration: 1,
    syncGeneration: 1,
    status: "SYNCED",
    lastAppliedUpdateId: "104",
    bufferedDiffCount: 0,
  });
});

Deno.test("l2 capture: snapshot older than first buffered diff stays in same generation and retains evidence", () => {
  const session = new L2CaptureSession("session-a", ["BTCUSDT"]);
  session.startConnection();
  session.acceptDepthDiff(diff("BTCUSDT", "110", "111"));
  const old = session.acceptDepthSnapshot(snapshot("BTCUSDT", "100"));
  assertEquals(old.disposition, "snapshot_too_old");
  assertEquals(session.state("BTCUSDT").syncGeneration, 1);
  assertEquals(session.state("BTCUSDT").bufferedDiffCount, 1);
  const fresh = session.acceptDepthSnapshot(snapshot("BTCUSDT", "110"));
  assertEquals(fresh.disposition, "synced");
  assertEquals(session.state("BTCUSDT").lastAppliedUpdateId, "111");
});

Deno.test("l2 capture: stale diff is persisted/ordered but does not move the applied update baseline", () => {
  const session = new L2CaptureSession("session-a", ["BTCUSDT"]);
  session.startConnection();
  session.acceptDepthDiff(diff("BTCUSDT", "101", "102"));
  session.acceptDepthSnapshot(snapshot("BTCUSDT", "100"));
  assertEquals(session.state("BTCUSDT").lastAppliedUpdateId, "102");
  const stale = session.acceptDepthDiff(diff("BTCUSDT", "100", "102"));
  assertEquals(stale.disposition, "stale");
  assertEquals(stale.envelope.arrivalSeq, 3);
  assertEquals(session.state("BTCUSDT").lastAppliedUpdateId, "102");
});

Deno.test("l2 capture: overlapping diff that extends history applies normally", () => {
  assertEquals(classifyDepthDiffSequence("102", diff("BTCUSDT", "101", "104")), "apply");
});

Deno.test("l2 capture: true sequence gap opens a new sync generation and buffers the gap-revealing diff", () => {
  const session = new L2CaptureSession("session-a", ["BTCUSDT"]);
  session.startConnection();
  session.acceptDepthDiff(diff("BTCUSDT", "101", "102"));
  session.acceptDepthSnapshot(snapshot("BTCUSDT", "100"));
  const gap = session.acceptDepthDiff(diff("BTCUSDT", "110", "111"));
  assertEquals(gap.disposition, "gap_resync");
  assertEquals(gap.previousSyncGeneration, 1);
  assertEquals(gap.envelope.syncGeneration, 2);
  assertEquals(session.state("BTCUSDT"), {
    symbol: "BTCUSDT",
    connectionGeneration: 1,
    syncGeneration: 2,
    status: "SYNCING",
    lastAppliedUpdateId: null,
    bufferedDiffCount: 1,
  });
  const tooOld = session.acceptDepthSnapshot(snapshot("BTCUSDT", "105"));
  assertEquals(tooOld.disposition, "snapshot_too_old");
  const recovered = session.acceptDepthSnapshot(snapshot("BTCUSDT", "110"));
  assertEquals(recovered.disposition, "synced");
  assertEquals(session.state("BTCUSDT").lastAppliedUpdateId, "111");
});

Deno.test("l2 capture: transport reconnect invalidates every symbol baseline and starts fresh generations", () => {
  const session = new L2CaptureSession("session-a", ["BTCUSDT", "ETHUSDT"]);
  session.startConnection();
  session.acceptDepthDiff(diff("BTCUSDT", "101", "102"));
  session.acceptDepthSnapshot(snapshot("BTCUSDT", "100"));
  session.acceptDepthDiff(diff("ETHUSDT", "201", "202"));
  session.acceptDepthSnapshot(snapshot("ETHUSDT", "200"));
  assertEquals(session.startConnection(), 2);
  assertEquals(session.state("BTCUSDT").status, "SYNCING");
  assertEquals(session.state("ETHUSDT").status, "SYNCING");
  assertEquals(session.state("BTCUSDT").syncGeneration, 2);
  assertEquals(session.state("ETHUSDT").syncGeneration, 2);
  assertEquals(session.state("BTCUSDT").lastAppliedUpdateId, null);
});

Deno.test("l2 capture: unknown symbol and pre-connection events fail closed", () => {
  const session = new L2CaptureSession("session-a", ["BTCUSDT"]);
  assertThrows(() => session.acceptDepthDiff(diff("BTCUSDT", "1", "2")), Error, "startConnection");
  session.startConnection();
  assertThrows(() => session.acceptDepthDiff(diff("ETHUSDT", "1", "2")), Error, "not part");
});
