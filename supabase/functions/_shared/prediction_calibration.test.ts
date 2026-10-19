import { type CalibrationInput, computeCalibration, validCalibrationInputs } from "./prediction_calibration.ts";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(`assertion failed: ${msg}`);
}
function near(a: number | null, b: number, msg: string, tol = 1e-6) {
  if (a === null || Math.abs(a - b) > tol) throw new Error(`${msg}: expected ${b} got ${a}`);
}

const row = (eventId: string, marketKey: string, forecast: number, outcome: 0 | 1): CalibrationInput => ({ eventId, marketKey, forecast, outcome });

Deno.test("Brier and log loss match hand computation", () => {
  const rows = [row("e1", "m1", 0.9, 1), row("e2", "m2", 0.2, 0), row("e3", "m3", 0.7, 0)];
  const r = computeCalibration(rows, { minEvents: 2, resamples: 200 });
  // Brier = (0.01 + 0.04 + 0.49) / 3 = 0.18
  near(r.brier, 0.18, "brier");
  // Log loss = -(ln .9 + ln .8 + ln .3) / 3
  near(r.logLoss, -(Math.log(0.9) + Math.log(0.8) + Math.log(0.3)) / 3, "log loss");
  // Base rate 1/3: ((2/3)^2 + (1/3)^2 + (1/3)^2) / 3
  near(r.baseRateBrier, (4 / 9 + 1 / 9 + 1 / 9) / 3, "base-rate brier");
  assert(r.tradeSignal === false, "never a trade signal");
});

Deno.test("too few independent EVENTS is INSUFFICIENT_DATA even with many markets", () => {
  // 30 markets but all inside 3 events: the effective sample is 3, not 30.
  const rows = Array.from({ length: 30 }, (_, i) => row(`e${i % 3}`, `m${i}`, 0.5, (i % 2) as 0 | 1));
  const r = computeCalibration(rows, { minEvents: 20 });
  assert(r.status === "INSUFFICIENT_DATA", "status");
  assert(r.markets === 30 && r.events === 3, "counts");
  assert(r.brierInterval === null, "no interval is reported on insufficient data");
  near(r.brier, 0.25, "point estimate is still shown");
});

Deno.test("event clustering widens the interval relative to treating correlated markets as independent", () => {
  // 25 events; within each event, 4 markets share the same error pattern (perfect within-event correlation).
  const clustered: CalibrationInput[] = [];
  const independent: CalibrationInput[] = [];
  for (let e = 0; e < 25; e++) {
    const good = e % 2 === 0;
    for (let k = 0; k < 4; k++) {
      clustered.push(row(`e${e}`, `m${e}-${k}`, good ? 0.9 : 0.6, good ? 1 : 0));
      independent.push(row(`u${e}-${k}`, `n${e}-${k}`, good ? 0.9 : 0.6, good ? 1 : 0));
    }
  }
  const c = computeCalibration(clustered, { resamples: 4000, seed: 7 });
  const u = computeCalibration(independent, { resamples: 4000, seed: 7 });
  assert(c.status === "SUFFICIENT" && u.status === "SUFFICIENT", "both sufficient");
  near(c.brier, u.brier!, "same point estimate");
  const cw = c.brierInterval!.upper - c.brierInterval!.lower;
  const uw = u.brierInterval!.upper - u.brierInterval!.lower;
  assert(cw > uw * 1.5, `clustered interval must be wider (clustered ${cw}, naive ${uw})`);
  assert(c.events === 25 && u.events === 100, "event counts");
});

Deno.test("bootstrap is reproducible for a fixed seed and the point estimate lies inside the interval", () => {
  const rows = Array.from({ length: 40 }, (_, i) => row(`e${i}`, `m${i}`, (i % 10) / 10 + 0.05, (i % 3 === 0 ? 1 : 0) as 0 | 1));
  const a = computeCalibration(rows, { seed: 42 });
  const b = computeCalibration(rows, { seed: 42 });
  assert(JSON.stringify(a) === JSON.stringify(b), "same seed, same report");
  assert(a.brierInterval!.lower <= a.brier! && a.brier! <= a.brierInterval!.upper, "point inside interval");
});

Deno.test("reliability bins count, average and observe correctly; p = 1 lands in the top bin", () => {
  const rows = [row("e1", "m1", 0.05, 0), row("e2", "m2", 0.08, 1), row("e3", "m3", 0.95, 1), row("e4", "m4", 1, 1)];
  const r = computeCalibration(rows, { bins: 10, minEvents: 2, resamples: 100 });
  const low = r.bins[0];
  const high = r.bins[9];
  assert(low.count === 2 && high.count === 2, `bin counts ${low.count}/${high.count}`);
  near(low.meanForecast, 0.065, "low bin mean forecast");
  near(low.observedRate, 0.5, "low bin observed rate");
  near(high.observedRate, 1, "high bin observed rate");
  assert(r.bins.slice(1, 9).every((b) => b.count === 0 && b.meanForecast === null), "empty bins are null, not 0");
});

Deno.test("invalid or duplicated rows are rejected, never coerced", () => {
  const rows = [
    row("e1", "m1", 0.4, 1),
    row("e1", "m1", 0.6, 1), // duplicate market
    row("e2", "m2", 1.2, 0), // impossible probability
    row("e3", "m3", Number.NaN, 0),
    { eventId: "e4", marketKey: "m4", forecast: 0.5, outcome: 2 } as unknown as CalibrationInput,
    row("", "m5", 0.5, 1), // missing event id
  ];
  const v = validCalibrationInputs(rows);
  assert(v.valid.length === 1 && v.rejected === 5, `valid ${v.valid.length}, rejected ${v.rejected}`);
  const r = computeCalibration(rows, { minEvents: 2 });
  assert(r.rejected === 5 && r.markets === 1, "report carries the rejection count");
});

Deno.test("no rows means no numbers", () => {
  const r = computeCalibration([]);
  assert(r.status === "INSUFFICIENT_DATA" && r.brier === null && r.logLoss === null && r.markets === 0, "empty");
});

Deno.test("log loss is finite even for a confident wrong forecast", () => {
  const r = computeCalibration([row("e1", "m1", 1, 0), row("e2", "m2", 0, 0)], { minEvents: 2, resamples: 100 });
  assert(Number.isFinite(r.logLoss!), "clipped, not infinite");
  near(r.brier, 0.5, "brier");
});
