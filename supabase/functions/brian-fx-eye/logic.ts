// Pure, side-effect-free parsing/scoring logic for brian-fx-eye, split out of index.ts so it
// can be exercised directly by logic.test.ts without a live network/Supabase dependency.

export interface EcbDay {
  date: string;
  rates: Record<string, number>;
}

/** Parses the ECB eurofxref-hist XML feed into one entry per trading day, newest first. */
export function parse(xml: string): EcbDay[] {
  const blocks = [...xml.matchAll(/<Cube\s+time=['"]([^'"]+)['"]>([\s\S]*?)<\/Cube>/g)].map((m) => ({
    date: m[1],
    body: m[2],
  }));
  return blocks
    .map((b) => {
      const rates: Record<string, number> = {};
      for (const m of b.body.matchAll(/<Cube\s+currency=['"]([A-Z]{3})['"]\s+rate=['"]([^'"]+)['"]\s*\/>/g)) {
        rates[m[1]] = Number(m[2]);
      }
      return { date: b.date, rates };
    })
    .filter((x) => Object.keys(x.rates).length > 0);
}

export function clip(v: number): number {
  return Math.max(0, Math.min(1, v));
}

export function sign(v: number): number {
  return v > 0 ? 1 : v < 0 ? -1 : 0;
}
