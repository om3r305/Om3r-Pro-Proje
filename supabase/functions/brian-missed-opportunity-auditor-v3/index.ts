import { createClient } from "npm:@supabase/supabase-js@2";
import { withCollectorLease } from "https://raw.githubusercontent.com/om3r305/Om3r-Pro-Proje/688203fed061a7d45653aeba4a11b0d52b473334/supabase/functions/_shared/collector_lease.ts";
import { requireCronAuth } from "https://raw.githubusercontent.com/om3r305/Om3r-Pro-Proje/688203fed061a7d45653aeba4a11b0d52b473334/supabase/functions/_shared/cron_auth.ts";
import {
  resolveAlphaAuditHorizon,
  type AlphaAuditAction,
  type AlphaAuditPricePoint,
} from "https://raw.githubusercontent.com/om3r305/Om3r-Pro-Proje/688203fed061a7d45653aeba4a11b0d52b473334/supabase/functions/_shared/alpha_audit.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

// Keep the collector id stable so Control Center health remains backward compatible.
const COLLECTOR_ID = "brian-missed-opportunity-auditor-v2";
const AUDITOR_RUNTIME_VERSION = "brian.alpha-auditor-v3.bounded-queue";
const EVIDENCE = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const HORIZONS = [300, 900, 3600] as const;
const LEASE_SECONDS = 120;
const PENDING_BATCH = 120;
const LOOKBACK = "12 hours";

type Decision = {
  decision_id: string;
  observed_at: string;
  asset_id: string;
  observed_reference_price: string | number | null;
  action: AlphaAuditAction;
  direction: -1 | 0 | 1;
  evidence_score: number;
  estimated_round_trip_cost_bps: number | null;
  source_observation_ids: string[];
};

type ExistingOutcome = { decision_id: string; horizon_seconds: number };

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

function utf8(s: string) { return new TextEncoder().encode(s); }
async function sha(s: string) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", utf8(s)));
  return [...d].map((b) => b.toString(16).padStart(2, "0")).join("");
}
function finite(v: unknown, fallback = 0) {
  if (v === null || v === undefined || v === "") return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}
function errorText(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  if (error && typeof error === "object") {
    try { return JSON.stringify(error); } catch { return Object.prototype.toString.call(error); }
  }
  return String(error);
}

async function recordRun(
  startedAt: string,
  status: "SUCCESS" | "FAILED",
  observed: number,
  stored: number,
  metadata: Record<string, unknown>,
  error?: unknown,
) {
  try {
    const finishedAt = new Date().toISOString();
    const runId = await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}|${AUDITOR_RUNTIME_VERSION}`);
    await supabase.from("brian_collector_runs").insert({
      run_id: runId,
      collector_id: COLLECTOR_ID,
      started_at: startedAt,
      finished_at: finishedAt,
      status,
      observed_records: observed,
      stored_records: stored,
      degraded_sources: [],
      error_class: error ? "ALPHA_AUDITOR_V3" : null,
      error_message: error ? errorText(error).slice(0, 2000) : null,
      evidence_class: EVIDENCE,
      shadow_only: true,
      live_execution: false,
      metadata: { auditor_runtime_version: AUDITOR_RUNTIME_VERSION, ...metadata },
    });
  } catch (logError) {
    console.error("auditor run logging failed", errorText(logError));
  }
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return json({ error: "POST required" }, 405);
  try {
    await requireCronAuth(req, supabase);
  } catch (error) {
    const message = errorText(error);
    const unauthorized = message.includes("UNAUTHORIZED_CRON");
    return json({
      status: unauthorized ? "UNAUTHORIZED" : "FAILED_CLOSED",
      error: message,
      shadow_only: true,
      live_execution: false,
    }, unauthorized ? 401 : 503);
  }

  const startedAt = new Date().toISOString();
  try {
    const lease = await withCollectorLease(supabase, COLLECTOR_ID, LEASE_SECONDS, async () => {
      const nowMs = Date.now();
      const nowIso = new Date(nowMs).toISOString();

      // DB-side NOT EXISTS keeps the queue bounded and prevents repeatedly loading hundreds of
      // already-resolved decisions. The 12h lookback also catches overnight outages gradually.
      const pendingResp = await supabase.rpc("brian_alpha_pending_audit_decisions", {
        p_now: nowIso,
        p_lookback: LOOKBACK,
        p_limit: PENDING_BATCH,
      });
      if (pendingResp.error) throw pendingResp.error;
      const decisions = (pendingResp.data ?? []) as Decision[];
      if (!decisions.length) {
        await recordRun(startedAt, "SUCCESS", 0, 0, {
          status: "NO_PENDING_AUDITS",
          pending_batch_limit: PENDING_BATCH,
          lookback: LOOKBACK,
        });
        return json({
          status: "NO_PENDING_AUDITS",
          auditor_runtime_version: AUDITOR_RUNTIME_VERSION,
          shadow_only: true,
          live_execution: false,
        });
      }

      const ids = decisions.map((d) => d.decision_id);
      const existingResp = await supabase.from("brian_alpha_decision_outcomes")
        .select("decision_id,horizon_seconds")
        .in("decision_id", ids)
        .limit(PENDING_BATCH * HORIZONS.length + 32);
      if (existingResp.error) throw existingResp.error;
      const existing = new Set(
        ((existingResp.data ?? []) as ExistingOutcome[])
          .map((r) => `${r.decision_id}|${r.horizon_seconds}`),
      );

      const byAsset = new Map<string, Decision[]>();
      for (const d of decisions) {
        const rows = byAsset.get(d.asset_id) ?? [];
        rows.push(d);
        byAsset.set(d.asset_id, rows);
      }

      const outcomeRows: Record<string, unknown>[] = [];
      const missedRows: Record<string, unknown>[] = [];
      let skippedMissingReferencePrice = 0;
      let skippedUnresolved = 0;
      let alphaReferencePathPoints = 0;
      let intrabarPathPoints = 0;

      for (const [asset, assetDecisions] of byAsset) {
        assetDecisions.sort((a, b) => Date.parse(a.observed_at) - Date.parse(b.observed_at));
        const auditableDecisions = assetDecisions.filter((d) => {
          const ref = Number(d.observed_reference_price);
          if (Number.isFinite(ref) && ref > 0) return true;
          skippedMissingReferencePrice++;
          return false;
        });
        if (!auditableDecisions.length) continue;

        const minAt = auditableDecisions[0].observed_at;
        const maxTargetMs = Math.min(
          nowMs,
          Math.max(...auditableDecisions.map((d) => Date.parse(d.observed_at) + 3600_000)) + 120_000,
        );
        const maxTarget = new Date(maxTargetMs).toISOString();

        const alphaPathResp = await supabase.from("brian_alpha_decisions")
          .select("observed_at,observed_reference_price,estimated_round_trip_cost_bps")
          .eq("asset_id", asset)
          .gte("observed_at", minAt)
          .lte("observed_at", maxTarget)
          .not("observed_reference_price", "is", null)
          .order("observed_at", { ascending: true })
          .limit(5000);
        if (alphaPathResp.error) throw alphaPathResp.error;
        const alphaPoints: AlphaAuditPricePoint[] = (alphaPathResp.data ?? []).map((row) => ({
          observed_at: String(row.observed_at),
          observed_mid_price: row.observed_reference_price as string | number,
          estimated_round_trip_cost_bps: row.estimated_round_trip_cost_bps as string | number | null,
        }));
        alphaReferencePathPoints += alphaPoints.length;

        const intrabarResp = await supabase.from("brian_intrabar_reaction_events")
          .select("observed_at,observed_mid_price,estimated_round_trip_cost_bps")
          .eq("asset_id", asset)
          .gte("observed_at", minAt)
          .lte("observed_at", maxTarget)
          .order("observed_at", { ascending: true })
          .limit(5000);
        if (intrabarResp.error) throw intrabarResp.error;
        const intrabarPoints = (intrabarResp.data ?? []) as AlphaAuditPricePoint[];
        intrabarPathPoints += intrabarPoints.length;

        const points = [...alphaPoints, ...intrabarPoints]
          .filter((p) => Number.isFinite(Date.parse(p.observed_at)))
          .sort((a, b) => Date.parse(a.observed_at) - Date.parse(b.observed_at));

        for (const d of auditableDecisions) {
          for (const horizon of HORIZONS) {
            if (Date.parse(d.observed_at) + horizon * 1000 > nowMs) continue;
            if (existing.has(`${d.decision_id}|${horizon}`)) continue;

            const resolved = resolveAlphaAuditHorizon({
              observedAt: d.observed_at,
              action: d.action,
              direction: d.direction,
              referencePrice: d.observed_reference_price,
              estimatedRoundTripCostBps: d.estimated_round_trip_cost_bps,
            }, horizon, points);
            if (!resolved) {
              skippedUnresolved++;
              continue;
            }

            const outcomeId = await sha(`alpha-outcome|${d.decision_id}|${horizon}`);
            outcomeRows.push({
              outcome_id: outcomeId,
              decision_id: d.decision_id,
              asset_id: d.asset_id,
              horizon_seconds: horizon,
              observed_at: d.observed_at,
              resolved_at: resolved.resolvedAt,
              reference_price: resolved.reference,
              resolved_price: resolved.resolved,
              gross_return: resolved.gross,
              direction_adjusted_return: resolved.directionAdjusted,
              mfe: resolved.mfe,
              mae: resolved.mae,
              classification: resolved.classification,
              explanation: resolved.explanation,
              metadata: {
                auditor_runtime_version: AUDITOR_RUNTIME_VERSION,
                reference_price_source: "brian_alpha_decisions.observed_reference_price",
                path_price_sources: [
                  "brian_alpha_decisions.observed_reference_price",
                  "brian_intrabar_reaction_events.observed_mid_price",
                ],
                original_action: d.action,
                estimated_round_trip_cost_bps: resolved.costBps,
                up_excursion: resolved.upExcursion,
                down_excursion: resolved.downExcursion,
                long_opportunity: resolved.longOpportunity,
                short_opportunity: resolved.shortOpportunity,
              },
              evidence_class: EVIDENCE,
              shadow_only: true,
              live_execution: false,
            });

            if (d.action === "WAIT" || d.action === "VETO") {
              const bestExcursion = Math.max(resolved.upExcursion, -resolved.downExcursion);
              const receiptId = await sha(`alpha-missed|${d.decision_id}|${horizon}`);
              missedRows.push({
                receipt_id: receiptId,
                asset_id: d.asset_id,
                horizon: `${horizon}s`,
                observed_at: d.observed_at,
                resolved_at: resolved.resolvedAt,
                opportunity_score: Math.max(0, Math.min(1, finite(d.evidence_score))),
                brian_action: "WAIT",
                hindsight_gross_return: bestExcursion,
                hindsight_net_return: bestExcursion - resolved.costBps / 10_000,
                mfe: resolved.upExcursion,
                mae: resolved.downExcursion,
                classification: resolved.classification,
                explanation: `${d.action}: ${resolved.explanation}`,
                source_observation_ids: d.source_observation_ids ?? [],
                evidence_class: EVIDENCE,
                shadow_only: true,
                live_execution: false,
              });
            }
          }
        }
      }

      if (outcomeRows.length) {
        const ins = await supabase.from("brian_alpha_decision_outcomes")
          .upsert(outcomeRows, { onConflict: "decision_id,horizon_seconds", ignoreDuplicates: true });
        if (ins.error) throw ins.error;
      }
      if (missedRows.length) {
        const ins = await supabase.from("brian_missed_opportunity_receipts")
          .upsert(missedRows, { onConflict: "receipt_id", ignoreDuplicates: true });
        if (ins.error) throw ins.error;
      }

      await recordRun(startedAt, "SUCCESS", decisions.length, outcomeRows.length + missedRows.length, {
        pending_batch_limit: PENDING_BATCH,
        lookback: LOOKBACK,
        outcome_count: outcomeRows.length,
        missed_receipt_count: missedRows.length,
        skipped_missing_reference_price: skippedMissingReferencePrice,
        skipped_unresolved: skippedUnresolved,
        alpha_reference_path_points: alphaReferencePathPoints,
        intrabar_path_points: intrabarPathPoints,
        audit_path_semantics: "bounded_pending_queue_alpha_reference_primary_plus_intrabar_secondary",
        audit_cost_semantics: "fail_closed_no_zero_fallback",
        reference_price_source: "brian_alpha_decisions.observed_reference_price",
        horizons_seconds: HORIZONS,
      });

      return json({
        status: "RESOLVED",
        auditor_runtime_version: AUDITOR_RUNTIME_VERSION,
        decisions_considered: decisions.length,
        skipped_missing_reference_price: skippedMissingReferencePrice,
        skipped_unresolved: skippedUnresolved,
        outcomes_written: outcomeRows.length,
        missed_receipts_written: missedRows.length,
        alpha_reference_path_points: alphaReferencePathPoints,
        intrabar_path_points: intrabarPathPoints,
        horizons_seconds: HORIZONS,
        reference_price_source: "brian_alpha_decisions.observed_reference_price",
        shadow_only: true,
        live_execution: false,
      });
    });

    if (lease.contended) {
      return json({ status: "SKIPPED_LEASE_CONTENDED", shadow_only: true, live_execution: false });
    }
    return lease.value!;
  } catch (error) {
    const message = errorText(error);
    console.error("brian-missed-opportunity-auditor-v3 failed", message);
    await recordRun(startedAt, "FAILED", 0, 0, {
      pending_batch_limit: PENDING_BATCH,
      lookback: LOOKBACK,
    }, error);
    return json({
      status: "FAILED_CLOSED",
      auditor_runtime_version: AUDITOR_RUNTIME_VERSION,
      error: message,
      shadow_only: true,
      live_execution: false,
    }, 500);
  }
});
