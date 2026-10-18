/** Audit-only cost evidence cannot become execution permission or use a future snapshot. */
export function selectDecisionAuditCost(observedAt: string, action: string,
  executionCost: unknown, metadata?: Record<string, unknown>) {
  if (executionCost !== null && executionCost !== undefined && executionCost !== "" &&
      Number.isFinite(Number(executionCost)) && Number(executionCost) >= 0) {
    return { costBps: Number(executionCost), source: "decision_execution_cost" };
  }
  const cost = metadata?.audit_cost as Record<string, unknown> | undefined;
  const age = Date.parse(observedAt) - Date.parse(String(cost?.captured_at));
  const bps = cost?.estimated_round_trip_cost_bps;
  if ((action === "WAIT" || action === "VETO") && cost?.role === "AUDIT_ONLY" &&
      cost.quality === "L2_OBSERVED" && cost.policy === "max_buy_sell_same_snapshot_v1" &&
      typeof cost.source_id === "string" && cost.source_id.startsWith("binance_public_rest_depth:") &&
      Number(cost.requested_notional_usd) > 0 && Number.isFinite(age) && age >= 0 && age <= 120_000 &&
      typeof bps === "number" && Number.isFinite(bps) && bps >= 0) {
    return { costBps: bps, source: "observed_audit_only" };
  }
  return { costBps: null, source: "unavailable_use_existing_in_horizon_policy" };
}
