import { assert, assertEquals } from "jsr:@std/assert@1";
import {
  buildPredictionStatus,
  calibrationSection,
  checkPilotKey,
  parseIngestBody,
  PILOT_INGEST_MAX_BODY_BYTES,
  PREDICTION_STATUS_SCHEMA,
  sha256Hex,
} from "./prediction_status.ts";

const at = new Date("2026-10-20T08:00:00Z");

Deno.test("status: missing or failed sections are UNAVAILABLE, never defaulted", () => {
  const p = buildPredictionStatus({ prediction: null, calibrationInputs: undefined, pilot: { nope: 1 } }, {}, at);
  assertEquals(p.schema_version, PREDICTION_STATUS_SCHEMA);
  assertEquals(p.prediction, { status: "UNAVAILABLE", reason: "PREDICTION_STATUS_MISSING" });
  assertEquals(p.calibration, { status: "UNAVAILABLE", reason: "CALIBRATION_INPUTS_MISSING" });
  assertEquals(p.niulai4_pilot, { status: "UNAVAILABLE", reason: "PILOT_STATUS_MISSING" });
  assertEquals([p.shadow_only, p.live_execution, p.trade_signal], [true, false, false]);
  const e = buildPredictionStatus({ prediction: { status: "OK" }, calibrationInputs: { rows: [] }, pilot: { status: "RUNNING" } }, { pilot: "READ_FAILED:PostgresError" }, at);
  assertEquals(e.prediction.status, "OK");
  assertEquals(e.niulai4_pilot, { status: "UNAVAILABLE", reason: "READ_FAILED:PostgresError" });
  assertEquals(e.observed_at, "2026-10-20T08:00:00.000Z");
});

Deno.test("status: calibration below 20 events is INSUFFICIENT_DATA with no interval", () => {
  const rows = Array.from({ length: 5 }, (_, i) => ({ eventId: `e${i}`, marketKey: `m${i}`, forecast: 0.6, outcome: i % 2 }));
  const c = calibrationSection({ as_of: "x", lead_seconds: 86400, rows, excluded: { no_forecast_before_lead: 2 } }) as Record<string, unknown>;
  assertEquals(c.status, "INSUFFICIENT_DATA");
  const report = c.report as { events: number; brierInterval: unknown; tradeSignal: boolean };
  assertEquals(report.events, 5);
  assertEquals(report.brierInterval, null);
  assertEquals(report.tradeSignal, false);
  assertEquals(c.excluded, { no_forecast_before_lead: 2 });
});

Deno.test("ingest auth: not configured, weak key, wrong key, right key", async () => {
  const good = "k".repeat(48);
  const hash = await sha256Hex(good);
  assertEquals(await checkPilotKey(good, null), { ok: false, status: 503, error: "PILOT_INGEST_NOT_CONFIGURED" });
  assertEquals(await checkPilotKey(good, { enabled: false, ingest_key_sha256: hash }), { ok: false, status: 503, error: "PILOT_INGEST_NOT_CONFIGURED" });
  assertEquals(await checkPilotKey(good, { enabled: true, ingest_key_sha256: null }), { ok: false, status: 503, error: "PILOT_INGEST_NOT_CONFIGURED" });
  assertEquals((await checkPilotKey("short", { enabled: true, ingest_key_sha256: hash })).ok, false);
  assertEquals((await checkPilotKey("x".repeat(48), { enabled: true, ingest_key_sha256: hash })).ok, false);
  assertEquals(await checkPilotKey(good, { enabled: true, ingest_key_sha256: hash }), { ok: true });
});

Deno.test("ingest body: size cap, invalid JSON, non-object", () => {
  assertEquals(parseIngestBody("x".repeat(PILOT_INGEST_MAX_BODY_BYTES + 1)), { ok: false, error: "BODY_TOO_LARGE" });
  assertEquals(parseIngestBody("{"), { ok: false, error: "INVALID_JSON" });
  assertEquals(parseIngestBody("[1]"), { ok: false, error: "PAYLOAD_NOT_OBJECT" });
  const ok = parseIngestBody('{"kind":"HEARTBEAT"}');
  assert(ok.ok);
});
