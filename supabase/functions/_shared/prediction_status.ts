// Read-only status payload for the UI: Prediction Intelligence + NIULAI4 pilot (pure, no I/O).
//
// The edge function reads three jsonb values from CORE (brian_prediction_status_v1,
// brian_prediction_calibration_inputs_v1, brian_niulai4_pilot_status_v1) and this module shapes
// them into one versioned contract (docs/BRIAN_PREDICTION_UI_CONTRACT.md). Every section carries
// its own status and timestamp; a section that could not be read is UNAVAILABLE with a reason --
// it is never filled with defaults that look like data.

import { computeCalibration, type CalibrationInput, type CalibrationReport } from "./prediction_calibration.ts";

export const PREDICTION_STATUS_SCHEMA = "brian-prediction-status.v1";

type Json = Record<string, unknown>;
export type Section<T> = ({ status: string } & T) | { status: "UNAVAILABLE"; reason: string };

export interface PredictionStatusPayload {
  schema_version: typeof PREDICTION_STATUS_SCHEMA;
  observed_at: string;
  shadow_only: true;
  live_execution: false;
  trade_signal: false;
  prediction: Section<Json>;
  calibration: Section<Json>;
  niulai4_pilot: Section<Json>;
  // Research Lab ledger (claims, latest test result per claim, assumptions). Additive in v1.
  research: Section<Json>;
}

export function unavailable(reason: string): { status: "UNAVAILABLE"; reason: string } {
  return { status: "UNAVAILABLE", reason: reason.slice(0, 120) };
}

// Calibration from the point-in-time inputs. Below 20 independent events it is
// INSUFFICIENT_DATA (no interval), exactly as computeCalibration reports it.
export function calibrationSection(inputs: unknown): Section<Json> {
  if (!inputs || typeof inputs !== "object") return unavailable("CALIBRATION_INPUTS_MISSING");
  const j = inputs as Json;
  const rows = Array.isArray(j.rows) ? j.rows : [];
  const typed: CalibrationInput[] = rows.map((r) => {
    const x = r as Json;
    return {
      eventId: String(x.eventId ?? ""),
      marketKey: String(x.marketKey ?? ""),
      forecast: Number(x.forecast),
      outcome: (Number(x.outcome) === 1 ? 1 : Number(x.outcome) === 0 ? 0 : NaN) as 0 | 1,
    };
  });
  const report: CalibrationReport = computeCalibration(typed);
  return {
    status: report.status,
    as_of: j.as_of ?? null,
    lead_seconds: j.lead_seconds ?? null,
    method: j.method ?? null,
    excluded: j.excluded ?? null,
    report,
  };
}

export function sectionFrom(value: unknown, missing: string): Section<Json> {
  if (!value || typeof value !== "object" || typeof (value as Json).status !== "string") return unavailable(missing);
  return value as { status: string } & Json;
}

export function buildPredictionStatus(
  parts: { prediction: unknown; calibrationInputs: unknown; pilot: unknown; research?: unknown },
  errors: { prediction?: string; calibration?: string; pilot?: string; research?: string },
  observedAt: Date,
): PredictionStatusPayload {
  return {
    schema_version: PREDICTION_STATUS_SCHEMA,
    observed_at: observedAt.toISOString(),
    shadow_only: true,
    live_execution: false,
    trade_signal: false,
    prediction: errors.prediction ? unavailable(errors.prediction) : sectionFrom(parts.prediction, "PREDICTION_STATUS_MISSING"),
    calibration: errors.calibration ? unavailable(errors.calibration) : calibrationSection(parts.calibrationInputs),
    niulai4_pilot: errors.pilot ? unavailable(errors.pilot) : sectionFrom(parts.pilot, "PILOT_STATUS_MISSING"),
    research: errors.research ? unavailable(errors.research) : sectionFrom(parts.research, "RESEARCH_STATUS_MISSING"),
  };
}

// ---------------------------------------------------------------------------------------------
// Pilot ingest request checks (the DB function re-validates everything).
// ---------------------------------------------------------------------------------------------
export const PILOT_INGEST_MAX_BODY_BYTES = 300_000;

export async function sha256Hex(s: string): Promise<string> {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)));
  return [...d].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
}

export type IngestAuth = { ok: true } | { ok: false; status: number; error: string };

// The pilot key is accepted only when the single auth row is enabled and its hash matches.
// A key shorter than 32 characters is refused before hashing (no weak keys).
export async function checkPilotKey(presented: string, row: { enabled: boolean; ingest_key_sha256: string | null } | null): Promise<IngestAuth> {
  if (!row || !row.enabled || !row.ingest_key_sha256) return { ok: false, status: 503, error: "PILOT_INGEST_NOT_CONFIGURED" };
  if (presented.length < 32) return { ok: false, status: 401, error: "UNAUTHORIZED_PILOT" };
  return constantTimeEqual(await sha256Hex(presented), row.ingest_key_sha256)
    ? { ok: true }
    : { ok: false, status: 401, error: "UNAUTHORIZED_PILOT" };
}

export function parseIngestBody(text: string): { ok: true; value: Json } | { ok: false; error: string } {
  if (new TextEncoder().encode(text).length > PILOT_INGEST_MAX_BODY_BYTES) return { ok: false, error: "BODY_TOO_LARGE" };
  try {
    const v = JSON.parse(text);
    if (!v || typeof v !== "object" || Array.isArray(v)) return { ok: false, error: "PAYLOAD_NOT_OBJECT" };
    return { ok: true, value: v as Json };
  } catch {
    return { ok: false, error: "INVALID_JSON" };
  }
}
