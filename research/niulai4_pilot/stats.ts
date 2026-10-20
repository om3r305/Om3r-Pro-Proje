// Cluster bootstrap (pure, deterministic). Observations inside one cluster (a 5-minute window, or
// a calendar day) are resampled together, so dependent rows never count as independent evidence.

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface ClusterValue {
  cluster: string;
  value: number;
}

export interface BootstrapResult {
  n: number;
  clusters: number;
  mean: number | null;
  lower: number | null;
  upper: number | null;
  level: number;
}

export function clusterBootstrapMean(rows: ClusterValue[], options: { resamples?: number; level?: number; seed?: number } = {}): BootstrapResult {
  const level = options.level ?? 0.95;
  const valid = rows.filter((r) => Number.isFinite(r.value) && r.cluster !== "");
  const groups = new Map<string, number[]>();
  for (const r of valid) {
    const g = groups.get(r.cluster) ?? [];
    g.push(r.value);
    groups.set(r.cluster, g);
  }
  if (!valid.length) return { n: 0, clusters: 0, mean: null, lower: null, upper: null, level };
  const mean = valid.reduce((s, r) => s + r.value, 0) / valid.length;
  if (groups.size < 2) return { n: valid.length, clusters: groups.size, mean, lower: null, upper: null, level };
  const list = [...groups.values()];
  const rand = mulberry32(options.seed ?? 1);
  const resamples = Math.max(200, Math.min(20_000, options.resamples ?? 2000));
  const stats: number[] = [];
  for (let b = 0; b < resamples; b++) {
    let sum = 0;
    let n = 0;
    for (let i = 0; i < list.length; i++) {
      const g = list[Math.floor(rand() * list.length)];
      for (const v of g) {
        sum += v;
        n++;
      }
    }
    stats.push(sum / n);
  }
  stats.sort((a, b) => a - b);
  const q = (p: number) => {
    const pos = (stats.length - 1) * p;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    return stats[lo] + (stats[hi] - stats[lo]) * (pos - lo);
  };
  const a = (1 - level) / 2;
  return { n: valid.length, clusters: groups.size, mean, lower: q(a), upper: q(1 - a), level };
}

export function quantiles(xs: number[], ps: number[]): (number | null)[] {
  const s = xs.filter(Number.isFinite).sort((a, b) => a - b);
  return ps.map((p) => (s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : null));
}
