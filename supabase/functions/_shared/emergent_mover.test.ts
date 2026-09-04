import { assert, assertEquals, assertRejects, assertThrows } from "jsr:@std/assert@^1.0.0";
import {
  buildEmergentMoverFrame,
  buildEmergentMoverReport,
  parseEmergentMoverFrame,
  percentileRanks,
  positivePercentileRanks,
  type EmergentMarketRow,
  type EmergentMoverFrame,
} from "./emergent_mover.ts";

function market(symbol: string, overrides: Partial<EmergentMarketRow> = {}): EmergentMarketRow {
  return {
    symbol,
    last_price: 10,
    quote_volume_24h: 20_000_000,
    trades_24h: 50_000,
    price_change_pct_24h: 2,
    high_price_24h: 11,
    low_price_24h: 9,
    spread_bps: 2,
    ...overrides,
  };
}

function frame(at: string, rows: EmergentMarketRow[], source = "binance_public"): EmergentMoverFrame {
  return buildEmergentMoverFrame(rows, { observed_at: at, source });
}

Deno.test("emergent mover: first prospective frame is baseline, never a fake mover alert", () => {
  const current = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT"), market("BBBUSDT")]);
  const report = buildEmergentMoverReport(null, current);
  assertEquals(report.comparable, false);
  assertEquals(report.comparison_issue, "no_baseline");
  assertEquals(report.candidates, []);
  assertEquals(report.newly_observed_symbols, []);
  assertEquals(report.shadow_only, true);
});

Deno.test("emergent mover: newly observed symbol is remembered but not ranked without its own baseline", () => {
  const previous = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT")]);
  const current = frame("2026-09-04T08:05:00.000Z", [
    market("AAAUSDT", { price_change_pct_24h: 2.1 }),
    market("NEWUSDT", { price_change_pct_24h: 40, quote_volume_24h: 100_000_000 }),
  ]);
  const report = buildEmergentMoverReport(previous, current);
  assertEquals(report.comparable, true);
  assertEquals(report.newly_observed_symbols, ["NEWUSDT"]);
  assert(report.candidates.every((candidate) => candidate.symbol !== "NEWUSDT"));
});

Deno.test("emergent mover: disappeared symbols are explicit and cannot leak into current ranking", () => {
  const previous = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT"), market("OLDUSDT")]);
  const current = frame("2026-09-04T08:05:00.000Z", [market("AAAUSDT", { price_change_pct_24h: 3 })]);
  const report = buildEmergentMoverReport(previous, current);
  assertEquals(report.disappeared_symbols, ["OLDUSDT"]);
  assert(report.candidates.every((candidate) => candidate.symbol !== "OLDUSDT"));
});

Deno.test("emergent mover: strongest fresh displacement rises to research attention without action/order fields", () => {
  const previous = frame("2026-09-04T08:00:00.000Z", [
    market("AAAUSDT", { price_change_pct_24h: 1, quote_volume_24h: 20_000_000, trades_24h: 30_000 }),
    market("FASTUSDT", { price_change_pct_24h: 1, quote_volume_24h: 20_000_000, trades_24h: 30_000 }),
    market("CALMUSDT", { price_change_pct_24h: 1, quote_volume_24h: 20_000_000, trades_24h: 30_000 }),
  ]);
  const current = frame("2026-09-04T08:05:00.000Z", [
    market("AAAUSDT", { price_change_pct_24h: 1.2, quote_volume_24h: 20_100_000, trades_24h: 30_050 }),
    market("FASTUSDT", { price_change_pct_24h: 8, high_price_24h: 13, low_price_24h: 8, quote_volume_24h: 35_000_000, trades_24h: 50_000 }),
    market("CALMUSDT", { price_change_pct_24h: 1.05, quote_volume_24h: 20_010_000, trades_24h: 30_010 }),
  ]);
  const report = buildEmergentMoverReport(previous, current);
  assertEquals(report.candidates[0].symbol, "FASTUSDT");
  assertEquals(report.candidates[0].observed_change_direction, 1);
  assert(report.candidates[0].attention_score > 0);
  assertEquals("action" in report.candidates[0], false);
  assertEquals("order" in report.candidates[0], false);
  assertEquals("buy" in report.candidates[0], false);
  assertEquals(report.candidates[0].evidence_class, "PROSPECTIVE_DEVELOPMENT_SHADOW");
});

Deno.test("emergent mover: downside emergence is observed as direction -1, never converted into a SELL instruction", () => {
  const previous = frame("2026-09-04T08:00:00.000Z", [market("DOWNUSDT"), market("PEERUSDT")]);
  const current = frame("2026-09-04T08:05:00.000Z", [
    market("DOWNUSDT", { price_change_pct_24h: -12, high_price_24h: 11, low_price_24h: 7, quote_volume_24h: 40_000_000, trades_24h: 80_000 }),
    market("PEERUSDT", { price_change_pct_24h: 2.1 }),
  ]);
  const candidate = buildEmergentMoverReport(previous, current).candidates.find((row) => row.symbol === "DOWNUSDT");
  assert(candidate);
  assertEquals(candidate.observed_change_direction, -1);
  assertEquals("sell" in candidate, false);
  assertEquals("side" in candidate, false);
});

Deno.test("emergent mover: stale baseline fails closed instead of comparing unrelated regimes", () => {
  const previous = frame("2026-09-04T06:00:00.000Z", [market("AAAUSDT")]);
  const current = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT", { price_change_pct_24h: 50 })]);
  const report = buildEmergentMoverReport(previous, current, { max_candidates: 15, max_comparison_age_ms: 30 * 60 * 1000 });
  assertEquals(report.comparable, false);
  assertEquals(report.comparison_issue, "comparison_too_old");
  assertEquals(report.candidates, []);
});

Deno.test("emergent mover: chronology and source lineage mismatches fail closed", () => {
  const current = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT")]);
  const future = frame("2026-09-04T08:05:00.000Z", [market("AAAUSDT")]);
  assertThrows(() => buildEmergentMoverReport(future, current), Error, "chronological");
  const otherSource = frame("2026-09-04T07:55:00.000Z", [market("AAAUSDT")], "fixture_venue");
  assertThrows(() => buildEmergentMoverReport(otherSource, current), Error, "same source");
});

Deno.test("emergent mover: frame rejects duplicate symbols and malformed market values", () => {
  assertThrows(
    () => frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT"), market("aaausdt")]),
    Error,
    "duplicate symbol",
  );
  assertThrows(
    () => frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT", { last_price: Number.NaN })]),
    Error,
    "finite",
  );
  assertThrows(
    () => frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT", { high_price_24h: 8, low_price_24h: 9 })]),
    Error,
    "high cannot be below low",
  );
  assertThrows(
    () => frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT", { trades_24h: 1.5 })]),
    Error,
    "integer",
  );
});

Deno.test("emergent mover: persisted frame parser rejects wrong schema and non-shadow lineage", () => {
  const valid = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT")]);
  assertThrows(() => parseEmergentMoverFrame({ ...valid, schema_version: "old" }), Error, "unsupported");
  assertThrows(() => parseEmergentMoverFrame({ ...valid, shadow_only: false }), Error, "shadow-only");
});

Deno.test("emergent mover: tie-aware percentile ranks do not depend on input position", () => {
  assertEquals(percentileRanks([10, 10, 20]), [0.25, 0.25, 1]);
  assertEquals(percentileRanks([20, 10, 10]), [1, 0.25, 0.25]);
  assertEquals(positivePercentileRanks([0, 0, 0]), [0, 0, 0]);
  assertEquals(positivePercentileRanks([-2, 1, 1]), [0, 0.75, 0.75]);
});

Deno.test("emergent mover: identical market content produces deterministic ranking regardless of input row order", () => {
  const oldRows = [market("AAAUSDT"), market("BBBUSDT"), market("CCCUSDT")];
  const newRows = [
    market("AAAUSDT", { price_change_pct_24h: 3, quote_volume_24h: 22_000_000 }),
    market("BBBUSDT", { price_change_pct_24h: 8, quote_volume_24h: 35_000_000, trades_24h: 75_000 }),
    market("CCCUSDT", { price_change_pct_24h: 2.2, quote_volume_24h: 21_000_000 }),
  ];
  const a = buildEmergentMoverReport(
    frame("2026-09-04T08:00:00.000Z", oldRows),
    frame("2026-09-04T08:05:00.000Z", newRows),
  );
  const b = buildEmergentMoverReport(
    frame("2026-09-04T08:00:00.000Z", [...oldRows].reverse()),
    frame("2026-09-04T08:05:00.000Z", [newRows[1], newRows[2], newRows[0]]),
  );
  assertEquals(a, b);
});

Deno.test("emergent mover: stable common symbols with no changing feature produce no fake candidates", () => {
  const rows = [market("AAAUSDT"), market("BBBUSDT"), market("CCCUSDT")];
  const previous = frame("2026-09-04T08:00:00.000Z", rows);
  const current = frame("2026-09-04T08:05:00.000Z", rows);
  const report = buildEmergentMoverReport(previous, current);
  assertEquals(report.compared_symbol_count, 3);
  assertEquals(report.candidates, []);
});

Deno.test("emergent mover: candidate score is bounded and report discloses rolling-window measurement semantics", () => {
  const previous = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT"), market("BBBUSDT")]);
  const current = frame("2026-09-04T08:05:00.000Z", [
    market("AAAUSDT", { price_change_pct_24h: 12, quote_volume_24h: 50_000_000, trades_24h: 100_000 }),
    market("BBBUSDT", { price_change_pct_24h: 2.1 }),
  ]);
  const report = buildEmergentMoverReport(previous, current);
  assert(report.candidates.every((candidate) => candidate.attention_score >= 0 && candidate.attention_score <= 1));
  assert(report.measurement_notes.some((note) => note.includes("rolling_window_displacement_not_interval_flow")));
  assertEquals(report.shadow_only, true);
});
