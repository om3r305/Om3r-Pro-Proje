export const DEFAULT_WS_BASE = "wss://stream.binance.com:9443";
export const DEFAULT_REST_BASE = "https://api.binance.com";
export const DEFAULT_SNAPSHOT_LIMIT = 1000;
export const DEFAULT_BATCH_MESSAGES = 200;
export const DEFAULT_FLUSH_MS = 2_000;
export const SNAPSHOT_RETRY_MIN_MS = 250;
export const SNAPSHOT_RETRY_MAX_MS = 5_000;
export const RECONNECT_MIN_MS = 1_000;
export const RECONNECT_MAX_MS = 30_000;
export const PROACTIVE_CONNECTION_ROTATION_MS = 23 * 60 * 60 * 1000 + 50 * 60 * 1000;

export interface L2WorkerConfig {
  supabaseUrl: string;
  serviceRoleKey: string;
  symbols: string[];
  wsBase: string;
  restBase: string;
  snapshotLimit: number;
  batchMessages: number;
  flushMs: number;
}

function requiredEnv(name: string): string {
  const value = Deno.env.get(name)?.trim() ?? "";
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function positiveIntEnv(name: string, fallback: number): number {
  const raw = Deno.env.get(name)?.trim();
  if (!raw) return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`);
  return value;
}

export function parseSymbols(raw: string): string[] {
  const symbols = raw.split(",").map((value) => value.trim().toUpperCase()).filter(Boolean);
  const unique = [...new Set(symbols)];
  if (!unique.length) throw new Error("BRIAN_L2_SYMBOLS must contain at least one symbol");
  if (unique.length !== symbols.length) throw new Error("BRIAN_L2_SYMBOLS contains duplicates");
  if (unique.length > 1024) throw new Error("BRIAN_L2_SYMBOLS exceeds Binance's 1024-stream connection limit");
  for (const symbol of unique) {
    if (!/^[A-Z0-9]+$/.test(symbol)) throw new Error(`invalid Binance symbol: ${symbol}`);
  }
  return unique.sort();
}

export function combinedDepthWsUrl(wsBase: string, symbols: string[]): string {
  if (!symbols.length) throw new Error("at least one depth symbol is required");
  const streams = symbols.map((symbol) => `${symbol.toLowerCase()}@depth@100ms`).join("/");
  return `${wsBase.replace(/\/$/, "")}/stream?streams=${streams}`;
}

export function depthSnapshotUrl(restBase: string, symbol: string, limit: number): string {
  const query = new URLSearchParams({ symbol, limit: String(limit) });
  return `${restBase.replace(/\/$/, "")}/api/v3/depth?${query.toString()}`;
}

export function loadConfigFromEnv(): L2WorkerConfig {
  const snapshotLimit = positiveIntEnv("BRIAN_L2_SNAPSHOT_LIMIT", DEFAULT_SNAPSHOT_LIMIT);
  if (![100, 500, 1000, 5000].includes(snapshotLimit)) {
    throw new Error("BRIAN_L2_SNAPSHOT_LIMIT must be one of 100, 500, 1000, 5000");
  }
  return {
    supabaseUrl: requiredEnv("SUPABASE_URL"),
    serviceRoleKey: requiredEnv("SUPABASE_SERVICE_ROLE_KEY"),
    symbols: parseSymbols(requiredEnv("BRIAN_L2_SYMBOLS")),
    wsBase: (Deno.env.get("BINANCE_L2_WS_BASE") ?? DEFAULT_WS_BASE).replace(/\/$/, ""),
    restBase: (Deno.env.get("BINANCE_L2_REST_BASE") ?? DEFAULT_REST_BASE).replace(/\/$/, ""),
    snapshotLimit,
    batchMessages: positiveIntEnv("BRIAN_L2_BATCH_MESSAGES", DEFAULT_BATCH_MESSAGES),
    flushMs: positiveIntEnv("BRIAN_L2_FLUSH_MS", DEFAULT_FLUSH_MS),
  };
}
