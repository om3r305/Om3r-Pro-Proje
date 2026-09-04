// Required-source coverage guard for the Emergent Mover cross-section.
// A partial required ticker universe would change percentile ranks and contaminate the next
// baseline, so incomplete required coverage is skipped instead of ranked.

export interface RequiredCoverageResult {
  complete: boolean;
  expected_count: number;
  valid_count: number;
  missing_symbols: string[];
  unexpected_symbols: string[];
}

function canonicalSymbols(values: string[], name: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of values) {
    if (typeof raw !== "string" || !raw.trim()) throw new Error(`${name} contains an invalid symbol`);
    const symbol = raw.trim().toUpperCase();
    if (seen.has(symbol)) throw new Error(`${name} contains duplicate symbol ${symbol}`);
    seen.add(symbol);
    out.push(symbol);
  }
  return out.sort();
}

export function assessRequiredTickerCoverage(
  expectedSymbols: string[],
  validSymbols: string[],
): RequiredCoverageResult {
  const expected = canonicalSymbols(expectedSymbols, "expectedSymbols");
  const valid = canonicalSymbols(validSymbols, "validSymbols");
  if (!expected.length) throw new Error("expectedSymbols cannot be empty");

  const expectedSet = new Set(expected);
  const validSet = new Set(valid);
  const missing = expected.filter((symbol) => !validSet.has(symbol));
  const unexpected = valid.filter((symbol) => !expectedSet.has(symbol));
  return {
    complete: missing.length === 0 && unexpected.length === 0 && valid.length === expected.length,
    expected_count: expected.length,
    valid_count: valid.length,
    missing_symbols: missing,
    unexpected_symbols: unexpected,
  };
}
