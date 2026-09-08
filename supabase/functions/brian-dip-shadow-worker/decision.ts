import {
  type Cal,
  clip,
  ENGINE_VERSION,
  hash,
  type J,
  MAX_HOLD_MS,
  n,
  POLICY_VERSION,
  type Runtime,
  sizeLong,
  type Struct,
  SYMBOL,
} from "../_shared/dip_v8.ts";
import type { Market } from "../_shared/dip_v8_market.ts";
import { structure } from "./structure.ts";

export const DECISION_REVISION = "dip-v8-micro-reclaim-20260908.1";
export const MICRO_RECLAIM_OFI_MIN = .25;

export function microReclaimReady(
  s1: Struct,
  s5: Struct,
  ofi: number,
  px: number,
): boolean {
  const high = s1.lastHigh, low = s1.lastLow;
  if (!high || !low || !(high.p > low.p)) return false;
  const span = high.p - low.p;
  return s1.trend === "UP" &&
    s5.trend === "UP" &&
    high.label === "HH" &&
    low.label === "HL" &&
    ofi >= MICRO_RECLAIM_OFI_MIN &&
    px > low.p &&
    px >= low.p + Math.max(span * .35, s1.atr * .25);
}

export function selectEconomicTarget(
  direction: "UP" | "DOWN" | "WAIT",
  entry: number,
  costBps: number,
  structs: Struct[],
): number | null {
  if (direction === "WAIT" || !(entry > 0)) return null;
  const kind = direction === "UP" ? "H" : "L";
  const candidates: number[] = [];
  for (const s of structs) {
    const equal = direction === "UP" ? s.equalHigh : s.equalLow;
    if (equal) candidates.push(equal);
    for (const p of s.pivots) if (p.kind === kind) candidates.push(p.p);
  }
  const levels = [...new Set(candidates)]
    .filter((x) => Number.isFinite(x) && x > 0 &&
      (direction === "UP" ? x > entry : x < entry))
    .sort((a, b) => direction === "UP" ? a - b : b - a);
  if (!levels.length) return null;
  const hurdle = 2.5 * costBps;
  return levels.find((x) => Math.abs(x - entry) / entry * 10000 >= hurdle) ??
    levels[0];
}

export async function candidate(
  m: Market,
  sessionId: string,
  rt: Runtime,
  cfg: J,
  at: number,
  getCalibration: (
    setup: string,
    direction: string,
    regime: string,
  ) => Promise<Cal>,
) {
  const [s1, s5, s15, s1h, s4h] = await Promise.all(
    ["1m", "5m", "15m", "1h", "4h"].map((tf) => structure(tf, m.bars[tf])),
  );
  const all = [s1, s5, s15, s1h, s4h],
    combinedFp = await hash(all.map((s) => s.fingerprint).join("|")),
    px = m.book.mid;
  let regime = "RANGE";
  if (s4h.trend === s1h.trend && s1h.trend !== "RANGE") {
    regime = "TREND_" + s1h.trend;
  } else if (s15.choch || s5.choch) regime = "TRANSITION";
  let setup = "NONE",
    direction: "UP" | "DOWN" | "WAIT" = "WAIT",
    trigger: Struct | null = null;
  for (const s of [s1, s5]) {
    if (s.sweep) {
      setup = "SWEEP_RECLAIM";
      direction = s.sweep === "BULL" ? "UP" : "DOWN";
      trigger = s;
      break;
    }
  }
  if (!trigger) {
    for (const s of [s1, s5]) {
      if (s.failedBreak) {
        setup = "FAILED_BREAK";
        direction = s.failedBreak === "BULL" ? "UP" : "DOWN";
        trigger = s;
        break;
      }
    }
  }
  if (!trigger && s5.bos) {
    const p = s5.bos === "UP" ? s5.lastHigh : s5.lastLow;
    if (p && Math.abs(px - p.p) <= s5.atr * .35) {
      setup = "BOS_RETEST";
      direction = s5.bos;
      trigger = s5;
    }
  }
  // Earlier long continuation: confirmed 1m HH/HL inside a 5m uptrend plus
  // aggressive positive flow. This creates a candidate only; all normal risk,
  // cost, R:R, calibration and sizing gates still apply below.
  if (!trigger && !rt.pos && microReclaimReady(s1, s5, m.flow.ofi, px)) {
    setup = "MICRO_RECLAIM";
    direction = "UP";
    trigger = s1;
  }
  const triggerPivot = trigger
    ? setup === "BOS_RETEST"
      ? (direction === "UP" ? trigger.lastHigh : trigger.lastLow)
      : setup === "MICRO_RECLAIM"
      ? trigger.lastLow
      : (direction === "UP" ? trigger.lastLow : trigger.lastHigh)
    : null;
  const inv = triggerPivot?.p ?? null,
    fee = n(cfg.fee_bps, 10),
    slip = n(cfg.slippage_bps, 1);
  const entry = direction === "DOWN"
    ? m.book.bid * (1 - slip / 10000)
    : m.book.ask * (1 + slip / 10000);
  const costBps = 2 * fee + 2 * slip + m.book.spreadBps;
  // Use only levels already known at decision time. Scan recent 1m/5m/15m
  // pivots, and prefer the nearest level that actually clears the cost hurdle.
  // If none clears it, preserve the nearest structural level so the normal
  // TARGET_BELOW_COST veto remains observable.
  const target = selectEconomicTarget(direction, entry, costBps, [s1, s5, s15]);
  const signalAt = trigger
    ? m.bars[trigger.tf].at(-1)!.ct + 1
    : m.bars["1m"].at(-1)!.ct + 1;
  const episode = await hash(
    [SYMBOL, setup, direction, trigger?.tf, triggerPivot?.t, inv].join("|"),
  );
  const occurrence = await hash(
    [sessionId, POLICY_VERSION, episode, signalAt].join("|"),
  );
  const cal = await getCalibration(setup, direction, regime);
  const geometry = !!(target && inv &&
    (direction === "UP"
      ? inv < entry && entry < target
      : direction === "DOWN" && target < entry && entry < inv));
  const reward = target ? Math.abs(target - entry) : 0,
    risk = inv ? Math.abs(entry - inv) : 0,
    rr = geometry && risk ? reward / risk : 0;
  const targetBps = reward / entry * 10000;
  const veto: string[] = [];
  if (direction === "WAIT") veto.push("NO_SIGNAL");
  else {
    if (!geometry || risk < s1.atr * .05) {
      veto.push("STRUCTURE_LEVEL_INCOMPLETE");
    }
    if (targetBps < 2.5 * costBps) veto.push("TARGET_BELOW_COST");
    if (rr < 2) veto.push("RR_TOO_LOW");
    if (direction === "DOWN") veto.push("SPOT_LONG_ONLY");
    const opposing = direction === "UP"
      ? m.flow.ofi <= -.08
      : m.flow.ofi >= .08;
    if (opposing) veto.push("OPPOSING_FLOW");
    if (cal.unavailable) veto.push("CALIBRATION_UNAVAILABLE");
    else if (cal.p === null) veto.push("CALIBRATING");
    else if (
      cal.lower <= (1 + costBps / Math.max(risk / entry * 10000, 1)) / (1 + rr)
    ) veto.push("CALIBRATION_NO_EDGE");
    if (
      cal.ambiguous > 0 &&
      cal.ambiguous / Math.max(1, cal.samples + cal.ambiguous) > .20
    ) veto.push("LABEL_UNCERTAINTY_HIGH");
  }
  const last5m = m.bars["5m"].at(-1)!.ct + 1;
  if (
    rt.lastLock?.episode_id === episode &&
    (rt.lastLock.fingerprint === combinedFp || last5m <= rt.lastLock.last5m_t)
  ) veto.push("THESIS_LOCKED_NO_NEW_STRUCTURE");
  if (cfg.execution_mode === "OBSERVE") veto.push("OBSERVATION_ONLY");
  let raw = .45 +
    (setup === "SWEEP_RECLAIM"
      ? .16
      : setup === "FAILED_BREAK"
      ? .14
      : setup === "MICRO_RECLAIM"
      ? .12
      : setup === "BOS_RETEST"
      ? .10
      : 0);
  if (direction !== "WAIT") {
    if (s15.choch === direction || s5.choch === direction) raw += .09;
    if (direction === "UP" ? m.flow.ofi > .08 : m.flow.ofi < -.08) raw += .08;
    if (direction === "UP" ? m.book.pressure > 1.08 : m.book.pressure < .92) {
      raw += .06;
    }
  }
  raw = clip(raw, 0, .92);
  if (direction !== "WAIT" && raw < .60) veto.push("RAW_CONVICTION_LOW");
  const size = geometry && direction === "UP" && inv
    ? sizeLong(rt.cash, entry, inv, cal, fee, slip, m.rules)
    : null;
  if (direction === "UP" && geometry && !size) {
    veto.push("MIN_NOTIONAL_OR_RISK_CAP");
  }
  const thesis: J = {
    thesis_id: occurrence,
    occurrence_id: occurrence,
    episode_id: episode,
    generated_at: new Date(at).toISOString(),
    decision_time: new Date(at).toISOString(),
    data_available_at: new Date(m.availableAt).toISOString(),
    signal_at: new Date(signalAt).toISOString(),
    setup,
    direction,
    thesis_state: veto.some((x) => x !== "CALIBRATING") ? "WAIT" : "CONFIRMED",
    regime,
    venue: "SPOT",
    entry_low: entry,
    entry_high: entry,
    invalidation_price: inv,
    target_price: target,
    structural_invalidation_price: direction === "DOWN"
      ? s5.lastHigh?.p
      : direction === "UP"
      ? s5.lastLow?.p
      : null,
    rr,
    target_distance_bps: targetBps,
    cost_bps: costBps,
    raw_conviction: direction === "WAIT" ? null : raw,
    calibrated_probability: cal.p,
    calibration_samples: cal.samples,
    calibration: cal,
    why: all.map((s) =>
      `${s.tf}:${s.trend}/${s.sweep || s.failedBreak || s.bos || "NONE"}`
    ),
    veto,
    structure: { s1, s5, s15, s1h, s4h, fingerprint: combinedFp },
    flow: {
      ...m.flow,
      kind: "RECENT_AGG_TRADE_DELTA",
      book_pressure: m.book.pressure,
      spread_bps: m.book.spreadBps,
    },
    decision_revision: DECISION_REVISION,
    policy_version: POLICY_VERSION,
    engine_version: ENGINE_VERSION,
  };
  // Store one compact immutable decision per occurrence, not repeated full market snapshots.
  const decision = geometry && direction !== "WAIT"
    ? {
      occurrence_id: occurrence,
      session_id: sessionId,
      episode_id: episode,
      symbol: SYMBOL,
      decision_at: new Date(at).toISOString(),
      due_at: new Date(at + MAX_HOLD_MS).toISOString(),
      setup,
      direction,
      regime,
      venue: "SPOT",
      entry_price: entry,
      target_price: target,
      invalidation_price: inv,
      raw_conviction: raw,
      calibrated_probability: cal.p,
      calibration_samples: cal.samples,
      policy_version: POLICY_VERSION,
      metric_version: "target-before-invalidation-v8.1",
      evidence: {
        signal_at: thesis.signal_at,
        data_available_at: thesis.data_available_at,
        structure_fingerprint: combinedFp,
        decision_revision: DECISION_REVISION,
        raw_conviction: raw,
        calibration: cal,
        veto,
        rr,
        cost_bps: costBps,
        flow: thesis.flow,
        levels: all.map((s) => ({
          tf: s.tf,
          trend: s.trend,
          high: s.lastHigh,
          low: s.lastLow,
          atr: s.atr,
        })),
        market_rules: m.rules,
      },
      shadow_only: true,
      live_execution: false,
    }
    : null;
  return {
    thesis,
    decision,
    occurrence,
    episode,
    combinedFp,
    last5m,
    direction,
    entry,
    inv,
    target,
    size,
    fee,
    slip,
    canEnter: direction === "UP" && veto.every((x) => x === "CALIBRATING") &&
      size !== null,
  };
}
