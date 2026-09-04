import { createClient } from "npm:@supabase/supabase-js@2";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import {
  resolveAlphaAuditHorizon,
  type AlphaAuditAction,
  type AlphaAuditPricePoint,
} from "../_shared/alpha_audit.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });
const COLLECTOR_ID = "brian-missed-opportunity-auditor-v2";
const EVIDENCE = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const HORIZONS = [300, 900, 3600] as const;
const LEASE_SECONDS = 120;

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

function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } }); }
function utf8(s: string) { return new TextEncoder().encode(s); }
async function sha(s: string) { const d = new Uint8Array(await crypto.subtle.digest("SHA-256", utf8(s))); return [...d].map((b) => b.toString(16).padStart(2, "0")).join(""); }
function finite(v: unknown, fallback = 0) { const n = Number(v); return Number.isFinite(n) ? n : fallback; }
function errorText(error: unknown) { return error instanceof Error ? `${error.name}: ${error.message}` : String(error); }

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return json({ error: "POST required" }, 405);
  try {
    await requireCronAuth(req, supabase);
  } catch (error) {
    const message = errorText(error);
    const unauthorized = message.includes("UNAUTHORIZED_CRON");
    return json({ status: unauthorized ? "UNAUTHORIZED" : "FAILED_CLOSED", error: message, shadow_only: true, live_execution: false }, unauthorized ? 401 : 503);
  }

  const startedAt = new Date().toISOString();
  try {
    const lease = await withCollectorLease(supabase, COLLECTOR_ID, LEASE_SECONDS, async () => {
      const nowMs = Date.now();
      const oldest = new Date(nowMs - 3 * 60 * 60_000).toISOString();
      const newestResolvable = new Date(nowMs - 5 * 60_000).toISOString();
      const decResp = await supabase.from("brian_alpha_decisions")
        .select("decision_id,observed_at,asset_id,observed_reference_price,action,direction,evidence_score,estimated_round_trip_cost_bps,source_observation_ids")
        .gte("observed_at", oldest).lte("observed_at", newestResolvable).order("observed_at", { ascending: true }).limit(500);
      if (decResp.error) throw decResp.error;
      const decisions = (decResp.data ?? []) as Decision[];
      if (!decisions.length) return json({ status: "NO_RESOLVABLE_DECISIONS", shadow_only: true, live_execution: false });
      const ids = decisions.map((d) => d.decision_id);
      const existingResp = await supabase.from("brian_alpha_decision_outcomes").select("decision_id,horizon_seconds").in("decision_id", ids).limit(2000);
      if (existingResp.error) throw existingResp.error;
      const existing = new Set((existingResp.data ?? []).map((r) => `${r.decision_id}|${r.horizon_seconds}`));

      const byAsset = new Map<string, Decision[]>();
      for (const d of decisions) { const rows = byAsset.get(d.asset_id) ?? []; rows.push(d); byAsset.set(d.asset_id, rows); }
      const outcomeRows: Record<string, unknown>[] = [];
      const missedRows: Record<string, unknown>[] = [];
      let skippedMissingReferencePrice = 0;
      let skippedUnresolved = 0;

      for (const [asset, assetDecisions] of byAsset) {
        const auditableDecisions = assetDecisions.filter((d) => {
          const ref = Number(d.observed_reference_price);
          if (Number.isFinite(ref) && ref > 0) return true;
          skippedMissingReferencePrice++;
          return false;
        });
        if (!auditableDecisions.length) continue;
        const minAt = auditableDecisions[0].observed_at;
        const maxTarget = new Date(Math.min(nowMs, Math.max(...auditableDecisions.map((d) => Date.parse(d.observed_at) + 3600_000)) + 120_000)).toISOString();
        const priceResp = await supabase.from("brian_intrabar_reaction_events")
          .select("observed_at,observed_mid_price,estimated_round_trip_cost_bps")
          .eq("asset_id", asset).gte("observed_at", minAt).lte("observed_at", maxTarget).order("observed_at", { ascending: true }).limit(5000);
        if (priceResp.error) throw priceResp.error;
        const points = (priceResp.data ?? []) as AlphaAuditPricePoint[];

        for (const d of auditableDecisions) for (const horizon of HORIZONS) {
          if (Date.parse(d.observed_at) + horizon * 1000 > nowMs) continue;
          if (existing.has(`${d.decision_id}|${horizon}`)) continue;
          const resolved = resolveAlphaAuditHorizon({
            observedAt: d.observed_at,
            action: d.action,
            direction: d.direction,
            referencePrice: d.observed_reference_price,
            estimatedRoundTripCostBps: d.estimated_round_trip_cost_bps,
          }, horizon, points);
          if (!resolved) { skippedUnresolved++; continue; }

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
              reference_price_source: "brian_alpha_decisions.observed_reference_price",
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
            // The shared historical receipt table supports WAIT rather than VETO; preserve the
            // exact original ALPHA action in the outcome metadata above instead of fabricating a
            // BUY/SELL action here.
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

      if (outcomeRows.length) { const ins = await supabase.from("brian_alpha_decision_outcomes").insert(outcomeRows); if (ins.error) throw ins.error; }
      if (missedRows.length) { const ins = await supabase.from("brian_missed_opportunity_receipts").insert(missedRows); if (ins.error) throw ins.error; }
      const finishedAt = new Date().toISOString();
      const runId = await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|SUCCESS`);
      await supabase.from("brian_collector_runs").insert({
        run_id: runId, collector_id: COLLECTOR_ID, started_at: startedAt, finished_at: finishedAt,
        status: "SUCCESS", observed_records: decisions.length, stored_records: outcomeRows.length + missedRows.length,
        degraded_sources: [], evidence_class: EVIDENCE, shadow_only: true, live_execution: false,
        metadata: {
          outcome_count: outcomeRows.length,
          missed_receipt_count: missedRows.length,
          skipped_missing_reference_price: skippedMissingReferencePrice,
          skipped_unresolved: skippedUnresolved,
          audit_cost_semantics: "fail_closed_no_zero_fallback",
          reference_price_source: "brian_alpha_decisions.observed_reference_price",
          horizons_seconds: HORIZONS,
        },
      });
      return json({
        status: "RESOLVED",
        decisions_considered: decisions.length,
        skipped_missing_reference_price: skippedMissingReferencePrice,
        skipped_unresolved: skippedUnresolved,
        outcomes_written: outcomeRows.length,
        missed_receipts_written: missedRows.length,
        horizons_seconds: HORIZONS,
        reference_price_source: "brian_alpha_decisions.observed_reference_price",
        shadow_only: true,
        live_execution: false,
      });
    });
    if (lease.contended) return json({ status: "SKIPPED_LEASE_CONTENDED", shadow_only: true, live_execution: false });
    return lease.value!;
  } catch (error) {
    console.error("brian-missed-opportunity-auditor-v2 failed", errorText(error));
    return json({ status: "FAILED_CLOSED", error: errorText(error), shadow_only: true, live_execution: false }, 500);
  }
});
