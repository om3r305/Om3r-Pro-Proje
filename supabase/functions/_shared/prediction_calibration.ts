// Prediction Intelligence -- Phase 2c: calibration of resolved binary markets (pure, no I/O).
//
// Input is one forecast per resolved market, taken point-in-time (see
// brian_prediction_calibration_inputs_v1: latest snapshot received at or before end_date - lead).
// Output: Brier score, log loss, reliability bins and an EVENT-CLUSTERED bootstrap interval.
//
// Why clustering: markets inside one Polymarket event are mutually dependent (e.g. "cut 25 bps",
// "cut 50 bps", "no change" for the same FOMC meeting). Treating them as independent rows inflates
// the effective sample and narrows intervals; resampling whole events does not. The method follows
// crollila/pm-calibration's pmcal/stats.py (MIT), re-implemented here; nothing is imported.
//
// This module measures how well MARKET probabilities were calibrated. It is not a trade signal and
// says nothing about after-cost profitability. Below the minimum number of independent events the
// status is INSUFFICIENT_DATA and no interval is reported.

export const PREDICTION_CALIBRATION_VERSION = "brian.prediction-calibration.v0";

export interface CalibrationInput {
  eventId: string;
  marketKey: string;
  forecast: number; // implied P(Yes) known at forecast time, in [0,1]
  outcome: 0 | 1; // 1 if Yes resolved true
}

export interface ReliabilityBin {
  lower: number;
  upper: number;
  count: number;
  meanForecast: number | null;
  observedRate: number | null;
}

export interface CalibrationReport {
  version: typeof PREDICTION_CALIBRATION_VERSION;
  status: "SUFFICIENT" | "INSUFFICIENT_DATA";
  markets: number;
  events: number;
  rejected: number;
  brier: number | null;
  logLoss: number | null;
  // Brier of always forecasting the sample's own base rate. In-sample, so only a reference point.
  baseRateBrier: number | null;
  brierInterval: { lower: number; upper: number; level: number; resamples: number } | null;
  bins: ReliabilityBin[];
  tradeSignal: false;
}

export interface CalibrationOptions {
  bins?: number;
  minEvents?: number;
  resamples?: number;
  level?: number;
  seed?: number;
  logLossEpsilon?: number;
}

// Deterministic PRNG so a report is reproducible from its inputs and seed.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function round(x: number, digits = 6): number {
  const f = 10 ** digits;
  return Math.round(x * f) / f;
}

export function validCalibrationInputs(rows: CalibrationInput[]): { valid: CalibrationInput[]; rejected: number } {
  const seen = new Set<string>();
  const valid: CalibrationInput[] = [];
  let rejected = 0;
  for (const r of rows) {
    const ok = typeof r.eventId === "string" && r.eventId !== "" && typeof r.marketKey === "string" && r.marketKey !== "" &&
      Number.isFinite(r.forecast) && r.forecast >= 0 && r.forecast <= 1 && (r.outcome === 0 || r.outcome === 1);
    // One forecast per market: a market repeated in the input would be double-counted.
    if (!ok || seen.has(r.marketKey)) {
      rejected++;
      continue;
    }
    seen.add(r.marketKey);
    valid.push(r);
  }
  return { valid, rejected };
}

export function computeCalibration(rows: CalibrationInput[], options: CalibrationOptions = {}): CalibrationReport {
  const binCount = Math.max(2, Math.min(20, Math.trunc(options.bins ?? 10)));
  const minEvents = Math.max(2, Math.trunc(options.minEvents ?? 20));
  const resamples = Math.max(100, Math.min(20_000, Math.trunc(options.resamples ?? 2000)));
  const level = options.level ?? 0.95;
  const eps = options.logLossEpsilon ?? 1e-6;

  const { valid, rejected } = validCalibrationInputs(rows);
  const byEvent = new Map<string, CalibrationInput[]>();
  for (const r of valid) {
    const list = byEvent.get(r.eventId) ?? [];
    list.push(r);
    byEvent.set(r.eventId, list);
  }

  const bins: ReliabilityBin[] = Array.from({ length: binCount }, (_, i) => {
    const lower = i / binCount;
    const upper = (i + 1) / binCount;
    const members = valid.filter((r) => r.forecast >= lower && (i === binCount - 1 ? r.forecast <= upper : r.forecast < upper));
    return {
      lower: round(lower, 4),
      upper: round(upper, 4),
      count: members.length,
      meanForecast: members.length ? round(members.reduce((s, r) => s + r.forecast, 0) / members.length) : null,
      observedRate: members.length ? round(members.reduce((s, r) => s + r.outcome, 0) / members.length) : null,
    };
  });

  const base = {
    version: PREDICTION_CALIBRATION_VERSION as typeof PREDICTION_CALIBRATION_VERSION,
    markets: valid.length,
    events: byEvent.size,
    rejected,
    bins,
    tradeSignal: false as const,
  };
  if (valid.length === 0) {
    return { ...base, status: "INSUFFICIENT_DATA", brier: null, logLoss: null, baseRateBrier: null, brierInterval: null };
  }

  const brierOf = (rs: CalibrationInput[]) => rs.reduce((s, r) => s + (r.forecast - r.outcome) ** 2, 0) / rs.length;
  const brier = brierOf(valid);
  const logLoss = valid.reduce((s, r) => {
    const p = Math.min(1 - eps, Math.max(eps, r.forecast));
    return s - (r.outcome * Math.log(p) + (1 - r.outcome) * Math.log(1 - p));
  }, 0) / valid.length;
  const baseRate = valid.reduce((s, r) => s + r.outcome, 0) / valid.length;
  const baseRateBrier = valid.reduce((s, r) => s + (baseRate - r.outcome) ** 2, 0) / valid.length;

  if (byEvent.size < minEvents) {
    return {
      ...base,
      status: "INSUFFICIENT_DATA",
      brier: round(brier),
      logLoss: round(logLoss),
      baseRateBrier: round(baseRateBrier),
      brierInterval: null,
    };
  }

  // Event-clustered percentile bootstrap: draw whole events with replacement.
  const events = [...byEvent.values()];
  const rand = mulberry32(options.seed ?? 20261019);
  const stats: number[] = [];
  for (let b = 0; b < resamples; b++) {
    let sum = 0;
    let n = 0;
    for (let i = 0; i < events.length; i++) {
      const ev = events[Math.floor(rand() * events.length)];
      for (const r of ev) {
        sum += (r.forecast - r.outcome) ** 2;
        n++;
      }
    }
    stats.push(sum / n);
  }
  stats.sort((a, b) => a - b);
  const alpha = (1 - level) / 2;
  return {
    ...base,
    status: "SUFFICIENT",
    brier: round(brier),
    logLoss: round(logLoss),
    baseRateBrier: round(baseRateBrier),
    brierInterval: { lower: round(quantile(stats, alpha)), upper: round(quantile(stats, 1 - alpha)), level, resamples },
  };
}
