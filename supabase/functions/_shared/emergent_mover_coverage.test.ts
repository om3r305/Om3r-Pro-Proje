import { assertEquals, assertThrows } from "jsr:@std/assert@^1.0.0";
import { assessRequiredTickerCoverage } from "./emergent_mover_coverage.ts";

Deno.test("emergent mover coverage: exact required universe is complete regardless of order/case", () => {
  const result = assessRequiredTickerCoverage(
    ["AAAUSDT", "BBBUSDT"],
    ["bbbusdt", "AAAUSDT"],
  );
  assertEquals(result.complete, true);
  assertEquals(result.missing_symbols, []);
  assertEquals(result.unexpected_symbols, []);
});

Deno.test("emergent mover coverage: one missing required ticker fails closed", () => {
  const result = assessRequiredTickerCoverage(
    ["AAAUSDT", "BBBUSDT", "CCCUSDT"],
    ["AAAUSDT", "CCCUSDT"],
  );
  assertEquals(result.complete, false);
  assertEquals(result.missing_symbols, ["BBBUSDT"]);
});

Deno.test("emergent mover coverage: unexpected rows cannot silently alter the cross-section", () => {
  const result = assessRequiredTickerCoverage(
    ["AAAUSDT", "BBBUSDT"],
    ["AAAUSDT", "BBBUSDT", "OTHERUSDT"],
  );
  assertEquals(result.complete, false);
  assertEquals(result.unexpected_symbols, ["OTHERUSDT"]);
});

Deno.test("emergent mover coverage: duplicate/empty symbol inputs are rejected", () => {
  assertThrows(() => assessRequiredTickerCoverage(["AAAUSDT", "aaausdt"], ["AAAUSDT"]), Error, "duplicate");
  assertThrows(() => assessRequiredTickerCoverage(["AAAUSDT"], [""]), Error, "invalid symbol");
  assertThrows(() => assessRequiredTickerCoverage([], []), Error, "cannot be empty");
});
