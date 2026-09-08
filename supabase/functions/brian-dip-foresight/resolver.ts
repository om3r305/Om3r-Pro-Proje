import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
import {
  evaluatePath,
  type J,
  METRIC_VERSION,
  POLICY_VERSION,
  RESOLVER_VERSION,
  SYMBOL,
} from "../_shared/dip_v8.ts";
import { pricePath } from "../_shared/dip_v8_market.ts";
export async function resolvePending(
  db: SupabaseClient,
  assertOwned: () => void,
  fetcher: typeof fetch = fetch,
): Promise<J> {
  const began = Date.now();
  const q = await db.from("brian_dip_v8_decisions").select(
    "occurrence_id,decision_at,due_at,direction,target_price,invalidation_price,entry_price,checked_until,last_price",
  ).eq("policy_version", POLICY_VERSION).eq("metric_version", METRIC_VERSION)
    .is("resolved_at", null).order("checked_until", {
      ascending: true,
      nullsFirst: true,
    }).order("decision_at", { ascending: true }).limit(24);
  if (q.error) throw Error("FORECAST_READ_FAILED:" + q.error.message);
  let resolved = 0, progressed = 0, deferred = 0;
  for (const row of q.data || []) {
    if (Date.now() - began > 32_000) {
      deferred++;
      break;
    }
    assertOwned();
    const start = Date.parse(row.checked_until || row.decision_at),
      due = Date.parse(row.due_at),
      entry = Number(row.last_price || row.entry_price);
    try {
      const path = await pricePath(start, due, Date.now(), entry, fetcher);
      if (path.end === start) continue;
      const result = evaluatePath({
        direction: row.direction,
        target: Number(row.target_price),
        stop: Number(row.invalidation_price),
        start,
        due,
        now: Date.now(),
        end: path.end,
        entry,
        segments: path.segments,
      });
      if (result.reason === "INDETERMINATE") {
        deferred++;
        continue;
      }
      const done = result.reason !== "PENDING",
        recordedAt = new Date().toISOString();
      const values: J = {
        checked_until: new Date(result.checkedUntil).toISOString(),
        last_price: result.price,
      };
      if (done) {
        Object.assign(values, {
          resolved_at: recordedAt,
          hit: result.hit,
          resolution: {
            ...result,
            resolver_version: RESOLVER_VERSION,
            recorded_at: recordedAt,
            event_time_precision: result.eventRangeStart === result.eventAt
              ? "TRADE_OR_DEADLINE"
              : "CANDLE_RANGE",
          },
        });
      }
      assertOwned();
      let update = db.from("brian_dip_v8_decisions").update(values).eq(
        "occurrence_id",
        row.occurrence_id,
      ).is("resolved_at", null);
      update = row.checked_until
        ? update.eq("checked_until", row.checked_until)
        : update.is("checked_until", null);
      const saved = await update.select("occurrence_id");
      if (saved.error) throw Error(saved.error.message);
      if (saved.data?.length) {
        progressed++;
        if (done) resolved++;
      }
    } catch (e) {
      deferred++;
      console.error(
        "dip-v8-resolver deferred",
        row.occurrence_id,
        e instanceof Error ? e.message : String(e),
      );
    }
  }
  return {
    status: "OK",
    resolved,
    progressed,
    deferred,
    resolver_version: RESOLVER_VERSION,
  };
}
export async function readForesight(
  db: SupabaseClient,
  sessionId?: string,
): Promise<J> {
  const s = await db.from("brian_dip_session_events").select(
    "session_id,event_kind,config",
  ).order("requested_at", { ascending: false }).order("event_id", {
    ascending: false,
  }).limit(1).maybeSingle();
  if (s.error) throw Error("SESSION_READ_FAILED:" + s.error.message);
  const session = s.data;
  const base = {
    forecasts: {},
    focus: [SYMBOL],
    shadow_only: true,
    live_execution: false,
    policy_version: POLICY_VERSION,
    metric_version: METRIC_VERSION,
  };
  if (!session) return { ...base, status: "NO_ACTIVE_SESSION" };
  if (sessionId && sessionId !== session.session_id) {
    return { ...base, status: "STALE_SESSION", session_id: session.session_id };
  }
  if (session.config?.policy_version !== POLICY_VERSION) {
    return {
      ...base,
      status: "WAIT_V8_CLEAN_RESTART",
      session_id: session.session_id,
    };
  }
  const q = await db.from("brian_dip_v8_runtime").select(
    "runtime,snapshot,updated_at,state_version",
  ).eq("session_id", session.session_id).maybeSingle();
  if (q.error || !q.data) throw Error("RUNTIME_READ_FAILED");
  const t = q.data.runtime.latestThesis;
  if (!t) {
    return {
      ...base,
      status: "WAIT_STRUCTURE",
      session_id: session.session_id,
    };
  }
  const f = {
    ...t,
    symbol: SYMBOL,
    target: t.target_price,
    invalidation: t.invalidation_price,
    structural_invalidation: t.structural_invalidation_price,
    peak: t.direction === "UP" ? t.target_price : t.invalidation_price,
    trough: t.direction === "UP" ? t.invalidation_price : t.target_price,
    confidence: t.raw_conviction,
    accuracy: t.calibrated_probability,
    samples: t.calibration_samples,
    horizon_min: 90,
    metric_version: METRIC_VERSION,
    candles: [],
    position: q.data.runtime.pos,
    updated_at: q.data.updated_at,
    state_version: q.data.state_version,
  };
  return {
    ...base,
    status: "OK",
    session_id: session.session_id,
    forecasts: { [SYMBOL]: f },
    meaning: {
      raw_conviction: "Yapısal puan; başarı olasılığı değildir",
      calibrated_probability:
        "Aynı politika/rejimde episode başına ilk ölçüm; belirsizlik ayrı gösterilir",
    },
  };
}
