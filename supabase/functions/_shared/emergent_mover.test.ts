import { assert, assertEquals, assertThrows } from "jsr:@std/assert@^1.0.0";
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
  const report = buildEmergentMoverReport(null, frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT"), market("BBBUSDT")]));
  assertEquals(report.comparable, false);
  assertEquals(report.comparison_issue, "no_baseline");
  assertEquals(report.candidates, []);
  assertEquals(report.newly_observed_symbols, []);
  assertEquals(report.shadow_only, true);
});

Deno.test("emergent mover: newly observed symbol is remembered but not ranked without its own baseline", () => {
  const previous = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT")]);
  const current = frame("2026-09-04T08:05:00.000Z", [
    market("AAAUSDT", { last_price: 10.1 }),
    market("NEWUSDT", { last_price: 14, price_change_pct_24h: 40, quote_volume_24h: 100_000_000 }),
  ]);
  const report = buildEmergentMoverReport(previous, current);
  assertEquals(report.newly_observed_symbols, ["NEWUSDT"]);
  assert(report.candidates.every((candidate) => candidate.symbol !== "NEWUSDT"));
});

Deno.test("emergent mover: disappeared symbols are explicit and cannot leak into current ranking", () => {
  const previous = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT"), market("OLDUSDT")]);
  const current = frame("2026-09-04T08:05:00.000Z", [market("AAAUSDT", { last_price: 10.2 })]);
  const report = buildEmergentMoverReport(previous, current);
  assertEquals(report.disappeared_symbols, ["OLDUSDT"]);
  assert(report.candidates.every((candidate) => candidate.symbol !== "OLDUSDT"));
});

Deno.test("emergent mover: strongest fresh last-price displacement rises to research attention without action fields", () => {
  const previous = frame("2026-09-04T08:00:00.000Z", [
    market("AAAUSDT", { quote_volume_24h: 20_000_000, trades_24h: 30_000 }),
    market("FASTUSDT", { quote_volume_24h: 20_000_000, trades_24h: 30_000 }),
    market("CALMUSDT", { quote_volume_24h: 20_000_000, trades_24h: 30_000 }),
  ]);
  const current = frame("2026-09-04T08:05:00.000Z", [
    market("AAAUSDT", { last_price: 10.1, quote_volume_24h: 20_100_000, trades_24h: 30_050 }),
    market("FASTUSDT", { last_price: 11.5, price_change_pct_24h: 8, high_price_24h: 13, low_price_24h: 8, quote_volume_24h: 35_000_000, trades_24h: 50_000 }),
    market("CALMUSDT", { last_price: 10.01, quote_volume_24h: 20_010_000, trades_24h: 30_010 }),
  ]);
  const candidate = buildEmergentMoverReport(previous, current).candidates[0];
  assertEquals(candidate.symbol, "FASTUSDT");
  assertEquals(candidate.observed_change_direction, 1);
  assert(candidate.features.short_return_pct > 0);
  assertEquals("action" in candidate, false);
  assertEquals("order" in candidate, false);
  assertEquals("buy" in candidate, false);
  assertEquals(candidate.evidence_class, "PROSPECTIVE_DEVELOPMENT_SHADOW");
});

Deno.test("emergent mover: downside last-price displacement is direction -1, never a SELL instruction", () => {
  const previous = frame("2026-09-04T08:00:00.000Z", [market("DOWNUSDT"), market("PEERUSDT")]);
  const current = frame("2026-09-04T08:05:00.000Z", [
    market("DOWNUSDT", { last_price: 8, price_change_pct_24h: -12, high_price_24h: 11, low_price_24h: 7, quote_volume_24h: 40_000_000, trades_24h: 80_000 }),
    market("PEERUSDT", { last_price: 10.01 }),
  ]);
  const candidate = buildEmergentMoverReport(previous, current).candidates.find((row) => row.symbol === "DOWNUSDT");
  assert(candidate);
  assertEquals(candidate.observed_change_direction, -1);
  assertEquals("sell" in candidate, false);
  assertEquals("side" in candidate, false);
});

Deno.test("emergent mover: rolling 24h price-change displacement alone is not mislabeled as short-return direction", () => {
  const previous = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT"), market("PEERUSDT")]);
  const current = frame("2026-09-04T08:05:00.000Z", [
    market("AAAUSDT", { last_price: 10, price_change_pct_24h: 20 }),
    market("PEERUSDT", { last_price: 10 }),
  ]);
  const report = buildEmergentMoverReport(previous, current);
  const aaa = report.candidates.find((row) => row.symbol === "AAAUSDT");
  if (aaa) {
    assertEquals(aaa.features.short_return_pct, 0);
    assertEquals(aaa.observed_change_direction, 0);
    assertEquals(aaa.features.rolling_price_change_delta_pct_24h, 18);
  }
  assert(report.measurement_notes.some((note) => note.includes("rolling_window_displacement_not_interval_flow")));
});

Deno.test("emergent mover: stale baseline fails closed instead of comparing unrelated regimes", () => {
  const previous = frame("2026-09-04T06:00:00.000Z", [market("AAAUSDT")]);
  const current = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT", { last_price: 15 })]);
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
  assertThrows(() => frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT"), market("aaausdt")]), Error, "duplicate symbol");
  assertThrows(() => frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT", { last_price: Number.NaN })]), Error, "finite number");
  assertThrows(() => frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT", { high_price_24h: 8, low_price_24h: 9 })]), Error, "high cannot be below low");
  assertThrows(() => frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT", { trades_24h: 1.5 })]), Error, "integer");
});

Deno.test("emergent mover: persisted frame parser rejects wrong schema, non-shadow, null and numeric-string fields", () => {
  const valid = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT")]);
  assertThrows(() => parseEmergentMoverFrame({ ...valid, schema_version: "old" }), Error, "unsupported");
  assertThrows(() => parseEmergentMoverFrame({ ...valid, shadow_only: false }), Error, "shadow-only");
  const nullRank = structuredClone(valid) as unknown as Record<string, unknown>;
  (nullRank.rows as Array<Record<string, unknown>>)[0].liquidity_rank = null;
  assertThrows(() => parseEmergentMoverFrame(nullRank), Error, "finite number");
  const stringPrice = structuredClone(valid) as unknown as Record<string, unknown>;
  (stringPrice.rows as Array<Record<string, unknown>>)[0].last_price = "10";
  assertThrows(() => parseEmergentMoverFrame(stringPrice), Error, "finite number");
});

Deno.test("emergent mover: tie-aware percentile ranks do not fabricate acceleration", () => {
  assertEquals(percentileRanks([10, 10, 20]), [0.25, 0.25, 1]);
  assertEquals(percentileRanks([20, 10, 10]), [1, 0.25, 0.25]);
  assertEquals(positivePercentileRanks([0, 0, 0]), [0, 0, 0]);
  assertEquals(positivePercentileRanks([-2, 1, 1]), [0, 0.75, 0.75]);
});

Deno.test("emergent mover: identical market content produces deterministic ranking regardless of row order", () => {
  const oldRows = [market("AAAUSDT"), market("BBBUSDT"), market("CCCUSDT")];
  const newRows = [
    market("AAAUSDT", { last_price: 10.2, quote_volume_24h: 22_000_000 }),
    market("BBBUSDT", { last_price: 11, quote_volume_24h: 35_000_000, trades_24h: 75_000 }),
    market("CCCUSDT", { last_price: 10.05, quote_volume_24h: 21_000_000 }),
  ];
  const a = buildEmergentMoverReport(frame("2026-09-04T08:00:00.000Z", oldRows), frame("2026-09-04T08:05:00.000Z", newRows));
  const b = buildEmergentMoverReport(frame("2026-09-04T08:00:00.000Z", [...oldRows].reverse()), frame("2026-09-04T08:05:00.000Z", [newRows[1], newRows[2], newRows[0]]));
  assertEquals(a, b);
});

Deno.test("emergent mover: stable common symbols with no changing feature produce no fake candidates", () => {
  const rows = [market("AAAUSDT"), market("BBBUSDT"), market("CCCUSDT")];
  const report = buildEmergentMoverReport(frame("2026-09-04T08:00:00.000Z", rows), frame("2026-09-04T08:05:00.000Z", rows));
  assertEquals(report.compared_symbol_count, 3);
  assertEquals(report.candidates, []);
});

Deno.test("emergent mover: candidate score stays bounded and semantics remain SHADOW-only", () => {
  const previous = frame("2026-09-04T08:00:00.000Z", [market("AAAUSDT"), market("BBBUSDT")]);
  const current = frame("2026-09-04T08:05:00.000Z", [
    market("AAAUSDT", { last_price: 12, quote_volume_24h: 50_000_000, trades_24h: 100_000 }),
    market("BBBUSDT", { last_price: 10.01 }),
  ]);
  const report = buildEmergentMoverReport(previous, current);
  assert(report.candidates.every((candidate) => candidate.attention_score >= 0 && candidate.attention_score <= 1));
  assertEquals(report.shadow_only, true);
  assert(report.measurement_notes.includes("research_attention_only_no_trade_action"));
});
