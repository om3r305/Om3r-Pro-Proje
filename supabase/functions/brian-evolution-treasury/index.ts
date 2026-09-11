import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { EVOLUTION_EVIDENCE_CLASS } from "../_shared/evolution_contract.ts";
import {
  BRIAN_TREASURY_GATE_VERSION,
  planPromotionGatedTreasuryCycle,
  type PromotionGateState,
} from "../_shared/evolution_treasury_gate.ts";
import {
  BRIAN_TREASURY_STARTING_EQUITY_USD,
  BRIAN_TREASURY_VERSION,
  initialTreasuryState,
  type TreasuryOpportunity,
  type TreasuryPosition,
  type TreasuryState,
} from "../_shared/evolution_treasury.ts";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const COLLECTOR_ID = "brian-evolution-treasury-v1";
const LEASE_SECONDS = 55;
const MIN_INTERVAL_SECONDS = 45;
const MAX_EDGE_ROWS = 100;

type SnapshotRow = {
  snapshot_id: string;
  observed_at: string;
  starting_equity_usd: number | string;
  cash_usd: number | string;
  realized_pnl_usd: number | string;
  cumulative_costs_usd: number | string;
  positions: unknown;
};

type EdgeRow = {
  decision_id: string;
  observed_at: string;
  asset_id: string;
  direction: number;
  estimated_round_trip_cost_bps: number | string | null;
  expected_net_edge_bps: number | string | null;
  recommendation: string;
  eligible: boolean;
  mature_group_count: number | string;
  reliability_weights: unknown;
  pit_clear: boolean;
};

function out(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

async function sha(value: string): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function finite(value: unknown): number | null {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function errorText(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function parsePositions(value: unknown): TreasuryPosition[] {
  if (!Array.isArray(value)) return [];
  const out: TreasuryPosition[] = [];
  for (const raw of value) {
    if (!raw || typeof raw !== "object") continue;
    const row = raw as Record<string, unknown>;
    const direction = Number(row.direction);
    const entryPrice = finite(row.entryPrice);
    const capitalUsd = finite(row.capitalUsd);
    const roundTripCostBps = finite(row.roundTripCostBps);
    if (!String(row.positionId ?? "") || !String(row.assetId ?? "") || (direction !== 1 && direction !== -1) || entryPrice == null || entryPrice <= 0 || capitalUsd == null || capitalUsd <= 0 || roundTripCostBps == null || roundTripCostBps < 0) continue;
    out.push({
      positionId: String(row.positionId),
      assetId: String(row.assetId),
      direction: direction as -1 | 1,
      openedAt: String(row.openedAt),
      entryPrice,
      capitalUsd,
      entryExpectedNetEdgeBps: finite(row.entryExpectedNetEdgeBps) ?? 0,
      latestExpectedNetEdgeBps: finite(row.latestExpectedNetEdgeBps) ?? 0,
      roundTripCostBps,
      sourceDecisionId: String(row.sourceDecisionId ?? "unknown"),
      highWaterPnlBps: finite(row.highWaterPnlBps) ?? 0,
    });
  }
  return out;
}

async function loadState(now: string): Promise<{ state: TreasuryState; previousSnapshotId: string | null }> {
  const q = await db.from("brian_treasury_shadow_latest")
    .select("snapshot_id,observed_at,starting_equity_usd,cash_usd,realized_pnl_usd,cumulative_costs_usd,positions")
    .limit(1).maybeSingle();
  if (q.error) throw new Error(`treasury_state:${q.error.message}`);
  if (!q.data) return { state: initialTreasuryState(now, BRIAN_TREASURY_STARTING_EQUITY_USD), previousSnapshotId: null };
  const row = q.data as SnapshotRow;
  const starting = finite(row.starting_equity_usd) ?? BRIAN_TREASURY_STARTING_EQUITY_USD;
  const cash = finite(row.cash_usd);
  const realized = finite(row.realized_pnl_usd);
  const costs = finite(row.cumulative_costs_usd);
  if (cash == null || cash < 0 || realized == null || costs == null || costs < 0) throw new Error("treasury_state: invalid latest snapshot");
  return {
    previousSnapshotId: String(row.snapshot_id),
    state: {
      observedAt: String(row.observed_at),
      startingEquityUsd: starting,
      cashUsd: cash,
      realizedPnlUsd: realized,
      cumulativeCostsUsd: costs,
      positions: parsePositions(row.positions),
    },
  };
}

async function loadPromotionGate(): Promise<PromotionGateState> {
  const decisionsQ = await db.from("brian_evolution_promotion_decisions")
    .select("decision_id,experiment_id,decided_at,decision")
    .order("decided_at", { ascending: false }).limit(200);
  if (decisionsQ.error) throw new Error(`promotion_gate_decisions:${decisionsQ.error.message}`);
  const decisions = decisionsQ.data ?? [];
  if (!decisions.length) return { authorized: false, reason: "no promotion decisions exist", evidenceRef: null, decidedAt: null };

  const experimentIds = [...new Set(decisions.map((row) => String(row.experiment_id)).filter(Boolean))];
  const experimentsQ = await db.from("brian_evolution_experiments")
    .select("experiment_id,hypothesis_id").in("experiment_id", experimentIds).limit(300);
  if (experimentsQ.error) throw new Error(`promotion_gate_experiments:${experimentsQ.error.message}`);
  const experimentToHypothesis = new Map((experimentsQ.data ?? []).map((row) => [String(row.experiment_id), String(row.hypothesis_id)]));
  const hypothesisIds = [...new Set([...experimentToHypothesis.values()].filter(Boolean))];
  if (!hypothesisIds.length) return { authorized: false, reason: "promotion decisions have no hypothesis lineage", evidenceRef: null, decidedAt: null };

  const hypothesisQ = await db.from("brian_evolution_hypothesis_snapshots")
    .select("hypothesis_id,observed_at,metadata").in("hypothesis_id", hypothesisIds)
    .order("observed_at", { ascending: false }).limit(1000);
  if (hypothesisQ.error) throw new Error(`promotion_gate_hypotheses:${hypothesisQ.error.message}`);
  const kindByHypothesis = new Map<string, string>();
  for (const row of hypothesisQ.data ?? []) {
    const id = String(row.hypothesis_id);
    if (kindByHypothesis.has(id)) continue;
    const metadata = (row.metadata ?? {}) as Record<string, unknown>;
    kindByHypothesis.set(id, String(metadata.hypothesis_kind ?? ""));
  }

  const seenExperiments = new Set<string>();
  for (const row of decisions) {
    const experimentId = String(row.experiment_id);
    if (!experimentId || seenExperiments.has(experimentId)) continue;
    seenExperiments.add(experimentId);
    const hypothesisId = experimentToHypothesis.get(experimentId);
    if (!hypothesisId || kindByHypothesis.get(hypothesisId) !== "EXPECTED_EDGE") continue;

    const decision = String(row.decision);
    const decidedAt = String(row.decided_at ?? "");
    const evidenceRef = String(row.decision_id);
    if (decision === "PROMOTE_CANDIDATE") {
      return {
        authorized: true,
        reason: `newest EXPECTED_EDGE prospective verdict promoted at ${decidedAt}`,
        evidenceRef,
        decidedAt,
      };
    }
    return {
      authorized: false,
      reason: `newest EXPECTED_EDGE prospective verdict is ${decision || "UNKNOWN"}`,
      evidenceRef,
      decidedAt,
    };
  }
  return { authorized: false, reason: "no EXPECTED_EDGE promotion decision exists", evidenceRef: null, decidedAt: null };
}

function averageReliability(value: unknown): number {
  if (!value || typeof value !== "object" || Array.isArray(value)) return 0.5;
  const values = Object.values(value as Record<string, unknown>).map(finite).filter((row): row is number => row != null && row >= 0 && row <= 1);
  return values.length ? values.reduce((sum, row) => sum + row, 0) / values.length : 0.5;
}

async function loadOpportunities(state: TreasuryState): Promise<TreasuryOpportunity[]> {
  const edgeQ = await db.from("brian_alpha_expected_edge_latest_by_asset")
    .select("decision_id,observed_at,asset_id,direction,estimated_round_trip_cost_bps,expected_net_edge_bps,recommendation,eligible,mature_group_count,reliability_weights,pit_clear")
    .order("observed_at", { ascending: false }).limit(MAX_EDGE_ROWS);
  if (edgeQ.error) throw new Error(`treasury_edges:${edgeQ.error.message}`);
  const edges = (edgeQ.data ?? []) as EdgeRow[];
  if (!edges.length) return [];

  const decisionIds = [...new Set(edges.map((row) => String(row.decision_id)).filter(Boolean))];
  const decisionQ = await db.from("brian_alpha_decisions")
    .select("decision_id,observed_reference_price").in("decision_id", decisionIds).limit(MAX_EDGE_ROWS * 2);
  if (decisionQ.error) throw new Error(`treasury_reference_prices:${decisionQ.error.message}`);
  const priceByDecision = new Map<string, number>();
  for (const row of decisionQ.data ?? []) {
    const price = finite(row.observed_reference_price);
    if (price != null && price > 0) priceByDecision.set(String(row.decision_id), price);
  }
  const positionByAsset = new Map(state.positions.map((position) => [position.assetId, position]));
  const opportunities: TreasuryOpportunity[] = [];
  for (const row of edges) {
    const decisionId = String(row.decision_id);
    const assetId = String(row.asset_id);
    const referencePrice = priceByDecision.get(decisionId);
    const direction = Number(row.direction);
    if (!assetId || referencePrice == null || (direction !== 1 && direction !== -1)) continue;
    const existing = positionByAsset.get(assetId);
    const cost = finite(row.estimated_round_trip_cost_bps) ?? existing?.roundTripCostBps ?? null;
    const edge = finite(row.expected_net_edge_bps) ?? (existing ? 0 : null);
    if (cost == null || cost < 0 || edge == null) continue;
    opportunities.push({
      assetId,
      direction: direction as -1 | 1,
      observedAt: String(row.observed_at),
      referencePrice,
      expectedNetEdgeBps: edge,
      roundTripCostBps: cost,
      reliabilityConfidence: averageReliability(row.reliability_weights),
      matureGroupCount: Math.max(0, Math.trunc(Number(row.mature_group_count ?? 0))),
      pitClear: row.pit_clear === true,
      recommendation: String(row.recommendation ?? "DOWNGRADE_TO_WAIT"),
      sourceDecisionId: decisionId,
    });
  }
  return opportunities;
}

async function commitCycle(input: {
  observedAt: string;
  previousSnapshotId: string | null;
  plan: ReturnType<typeof planPromotionGatedTreasuryCycle>;
  opportunities: TreasuryOpportunity[];
}): Promise<{ cycleId: string; snapshotId: string }> {
  const cycleId = await sha(`${BRIAN_TREASURY_VERSION}|${input.previousSnapshotId ?? "genesis"}|${input.observedAt}`);
  const snapshotId = await sha(`treasury-snapshot|${cycleId}`);
  const actions: Record<string, unknown>[] = [];
  for (let index = 0; index < input.plan.actions.length; index++) {
    const action = input.plan.actions[index];
    actions.push({
      action_id: await sha(`treasury-action|${cycleId}|${index}|${action.kind}|${action.assetId}|${action.positionId ?? "none"}`),
      observed_at: input.observedAt,
      kind: action.kind,
      asset_id: action.assetId,
      direction: action.direction,
      capital_usd: action.capitalUsd,
      reference_price: action.referencePrice,
      cost_usd: action.costUsd,
      expected_net_edge_bps: action.expectedNetEdgeBps,
      source_decision_id: action.sourceDecisionId,
      reason: action.reason,
      position_id: action.positionId,
      metadata: { treasury_version: BRIAN_TREASURY_VERSION, gate_version: BRIAN_TREASURY_GATE_VERSION },
    });
  }
  const snapshot = {
    snapshot_id: snapshotId,
    cycle_id: cycleId,
    observed_at: input.observedAt,
    previous_snapshot_id: input.previousSnapshotId,
    starting_equity_usd: input.plan.state.startingEquityUsd,
    cash_usd: input.plan.state.cashUsd,
    equity_usd: input.plan.afterEquityUsd,
    realized_pnl_usd: input.plan.state.realizedPnlUsd,
    cumulative_costs_usd: input.plan.state.cumulativeCostsUsd,
    deployment_usd: input.plan.deploymentUsd,
    deployment_pct: input.plan.deploymentPct,
    cash_reserve_pct: input.plan.cashReservePct,
    positions: input.plan.state.positions,
    promotion_gate_open: input.plan.promotionGate.authorized,
    promotion_gate_ref: input.plan.promotionGate.evidenceRef,
    promotion_gate_reason: input.plan.promotionGate.reason,
    treasury_version: BRIAN_TREASURY_VERSION,
    gate_version: BRIAN_TREASURY_GATE_VERSION,
    blocked_reasons: input.plan.blockedReasons,
    metadata: {
      opportunities_observed: input.opportunities.length,
      promotion_gate_decided_at: input.plan.promotionGate.decidedAt ?? null,
      shadow_only: true,
      live_execution: false,
      canonical_alpha_mutation: false,
    },
  };
  const q = await db.rpc("brian_commit_treasury_shadow_cycle", { p_snapshot: snapshot, p_actions: actions });
  if (q.error) throw new Error(`treasury_commit:${q.error.message}`);
  return { cycleId, snapshotId };
}

async function recordRun(startedAt: string, status: "SUCCESS" | "FAILED" | "SKIPPED", observed: number, stored: number, metadata: Record<string, unknown> = {}, error?: unknown) {
  const finishedAt = new Date().toISOString();
  const runId = await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q = await db.from("brian_collector_runs").insert({
    run_id: runId,
    collector_id: COLLECTOR_ID,
    started_at: startedAt,
    finished_at: finishedAt,
    status,
    observed_records: observed,
    stored_records: stored,
    degraded_sources: [],
    error_class: error ? "EVOLUTION_TREASURY_ERROR" : null,
    error_message: error ? errorText(error).slice(0, 1200) : null,
    metadata: {
      treasury_version: BRIAN_TREASURY_VERSION,
      gate_version: BRIAN_TREASURY_GATE_VERSION,
      shadow_only: true,
      live_execution: false,
      ...metadata,
    },
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
  });
  if (q.error) console.error("treasury run receipt", q.error.message);
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ error: "POST required" }, 405);
  const startedAt = new Date().toISOString();
  try {
    await requireCronAuth(req, db);
  } catch (error) {
    return out({ status: "UNAUTHORIZED", error: errorText(error), shadow_only: true, live_execution: false }, 401);
  }

  try {
    const last = await db.from("brian_collector_runs").select("started_at")
      .eq("collector_id", COLLECTOR_ID).in("status", ["SUCCESS", "DEGRADED"])
      .order("started_at", { ascending: false }).limit(1).maybeSingle();
    if (last.error) throw last.error;
    if (last.data?.started_at) {
      const age = (Date.now() - Date.parse(String(last.data.started_at))) / 1000;
      if (Number.isFinite(age) && age < MIN_INTERVAL_SECONDS) return out({ status: "SKIPPED_RATE_GUARD", age_seconds: age, shadow_only: true, live_execution: false });
    }

    const lease = await withCollectorLease(db, COLLECTOR_ID, LEASE_SECONDS, async () => {
      const observedAt = new Date().toISOString();
      const loaded = await loadState(observedAt);
      const [promotionGate, opportunities] = await Promise.all([loadPromotionGate(), loadOpportunities(loaded.state)]);
      const plan = planPromotionGatedTreasuryCycle({
        state: loaded.state,
        opportunities,
        observedAt,
        promotionGate,
        positionIdFor: (opportunity) => `treasury:${opportunity.sourceDecisionId}`,
      });
      const committed = await commitCycle({ observedAt, previousSnapshotId: loaded.previousSnapshotId, plan, opportunities });
      await recordRun(startedAt, "SUCCESS", opportunities.length, 1 + plan.actions.length, {
        cycle_id: committed.cycleId,
        snapshot_id: committed.snapshotId,
        promotion_gate_open: plan.promotionGate.authorized,
        promotion_gate_ref: plan.promotionGate.evidenceRef,
        promotion_gate_decided_at: plan.promotionGate.decidedAt ?? null,
        actions: plan.actions.length,
        positions: plan.state.positions.length,
        equity_usd: plan.afterEquityUsd,
        cash_usd: plan.state.cashUsd,
        deployment_usd: plan.deploymentUsd,
      });
      return {
        status: "SUCCESS",
        collector_id: COLLECTOR_ID,
        cycle_id: committed.cycleId,
        observed_at: observedAt,
        promotion_gate: plan.promotionGate,
        opportunities: opportunities.length,
        actions: plan.actions,
        treasury: {
          starting_equity_usd: plan.state.startingEquityUsd,
          equity_usd: plan.afterEquityUsd,
          cash_usd: plan.state.cashUsd,
          deployment_usd: plan.deploymentUsd,
          deployment_pct: plan.deploymentPct,
          cash_reserve_pct: plan.cashReservePct,
          realized_pnl_usd: plan.state.realizedPnlUsd,
          cumulative_costs_usd: plan.state.cumulativeCostsUsd,
          positions: plan.state.positions,
        },
        blocked_reasons: plan.blockedReasons,
        cloud_independent: true,
        canonical_alpha_mutation: false,
        shadow_only: true,
        live_execution: false,
      };
    });
    if (lease.contended) {
      await recordRun(startedAt, "SKIPPED", 0, 0);
      return out({ status: "SKIPPED_LEASE_CONTENDED", shadow_only: true, live_execution: false });
    }
    return out(lease.value);
  } catch (error) {
    await recordRun(startedAt, "FAILED", 0, 0, {}, error);
    return out({ status: "FAILED", error: errorText(error), shadow_only: true, live_execution: false }, 500);
  }
});
