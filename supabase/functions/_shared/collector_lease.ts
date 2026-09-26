import postgres from "npm:postgres@3.4.7";

// Collector-level atomic lease/mutex for Brian collectors.
//
// The database lease remains the source of truth. This client wrapper adds bounded retry and
// structured error normalization around the three idempotent lease RPCs so a transient
// PostgREST/schema-cache/connection wobble does not immediately take every collector down with
// an opaque "[object Object]" failure. The normal path is PostgREST RPC first; direct Postgres
// is a bounded fallback only after RPC exhaustion because Edge Runtime direct port-5432
// connectivity can be intermittently unavailable and must not add latency to every heartbeat.

export interface RpcClient {
  rpc(fn: string, params: Record<string, unknown>): PromiseLike<{ data: unknown; error: unknown }>;
}

export interface LeaseResult<T> {
  contended: boolean;
  ownerToken: string;
  value?: T;
}

type RpcResult = { data: unknown; error: unknown };
type AbortableRpcRequest = PromiseLike<RpcResult> & {
  abortSignal?: (signal: AbortSignal) => PromiseLike<RpcResult>;
};

const RPC_ATTEMPTS = 2;
const RPC_BACKOFF_MS = [250];
const RPC_TIMEOUT_MS = 5_000;

let directSql: ReturnType<typeof postgres> | null = null;

function readDbUrl(): string {
  try {
    return typeof Deno !== "undefined" ? (Deno.env.get("SUPABASE_DB_URL") ?? "") : "";
  } catch {
    // Deno test sandboxes may intentionally omit --allow-env. In that case
    // preserve the PostgREST fallback instead of failing during module import.
    return "";
  }
}

function sqlClient() {
  const dbUrl = readDbUrl();
  if (!dbUrl) return null;
  if (!directSql) {
    directSql = postgres(dbUrl, {
      prepare: false,
      max: 1,
      connect_timeout: 3,
      idle_timeout: 5,
      max_lifetime: 30,
    });
  }
  return directSql;
}

async function directLeaseCall(
  fn: "acquire" | "renew" | "release",
  collectorId: string,
  ownerToken: string,
  leaseSeconds?: number,
): Promise<boolean | null> {
  const sql = sqlClient();
  if (!sql) return null;

  try {
    if (fn === "acquire") {
      const rows = await sql`
        select public.brian_acquire_collector_lease(
          ${collectorId}, ${ownerToken}, ${Number(leaseSeconds ?? 1)}
        ) as value
      `;
      return rows?.[0]?.value === true;
    }
    if (fn === "renew") {
      const rows = await sql`
        select public.brian_renew_collector_lease(
          ${collectorId}, ${ownerToken}, ${Number(leaseSeconds ?? 1)}
        ) as value
      `;
      return rows?.[0]?.value === true;
    }
    const rows = await sql`
      select public.brian_release_collector_lease(
        ${collectorId}, ${ownerToken}
      ) as value
    `;
    return rows?.[0]?.value === true;
  } catch (error) {
    console.error(`collector lease direct postgres ${fn} failed for ${collectorId}: ${errorText(error)}`);
    return null;
  }
}

export function randomOwnerToken(): string {
  return crypto.randomUUID();
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function errorText(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  if (error && typeof error === "object") {
    const row = error as Record<string, unknown>;
    const fields = ["code", "message", "details", "hint", "status", "statusText"]
      .filter((key) => row[key] != null)
      .map((key) => `${key}=${String(row[key])}`);
    if (fields.length) return fields.join(" | ");
    try { return JSON.stringify(error); } catch { /* fall through */ }
  }
  return String(error);
}

async function rpcWithRetry(
  client: RpcClient,
  fn: string,
  params: Record<string, unknown>,
): Promise<unknown> {
  let lastError: unknown = null;
  for (let attempt = 1; attempt <= RPC_ATTEMPTS; attempt++) {
    let result: RpcResult;
    try {
      const request = client.rpc(fn, params) as AbortableRpcRequest;
      if (typeof request?.abortSignal === "function") {
        result = await request.abortSignal(AbortSignal.timeout(RPC_TIMEOUT_MS));
      } else {
        result = await new Promise<RpcResult>((resolve, reject) => {
          const timer = setTimeout(
            () => reject(new Error(`${fn} timeout after ${RPC_TIMEOUT_MS}ms`)),
            RPC_TIMEOUT_MS,
          );
          Promise.resolve(request).then(
            (value) => {
              clearTimeout(timer);
              resolve(value);
            },
            (error) => {
              clearTimeout(timer);
              reject(error);
            },
          );
        });
      }
    } catch (error) {
      lastError = error;
      if (attempt === RPC_ATTEMPTS) break;
      await sleep(RPC_BACKOFF_MS[attempt - 1] ?? 1_500);
      continue;
    }

    if (!result.error) return result.data;
    lastError = result.error;
    if (attempt === RPC_ATTEMPTS) break;
    await sleep(RPC_BACKOFF_MS[attempt - 1] ?? 1_500);
  }
  throw new Error(`${fn} failed after ${RPC_ATTEMPTS} attempts: ${errorText(lastError)}`);
}

export async function acquireCollectorLease(
  client: RpcClient,
  collectorId: string,
  ownerToken: string,
  leaseSeconds: number,
): Promise<boolean> {
  try {
    const data = await rpcWithRetry(client, "brian_acquire_collector_lease", {
      p_collector_id: collectorId,
      p_owner_token: ownerToken,
      p_lease_seconds: leaseSeconds,
    });
    return data === true;
  } catch (rpcError) {
    const direct = await directLeaseCall("acquire", collectorId, ownerToken, leaseSeconds);
    if (direct !== null) return direct;
    throw rpcError;
  }
}

export async function renewCollectorLease(
  client: RpcClient,
  collectorId: string,
  ownerToken: string,
  leaseSeconds: number,
): Promise<boolean> {
  try {
    const data = await rpcWithRetry(client, "brian_renew_collector_lease", {
      p_collector_id: collectorId,
      p_owner_token: ownerToken,
      p_lease_seconds: leaseSeconds,
    });
    return data === true;
  } catch (rpcError) {
    const direct = await directLeaseCall("renew", collectorId, ownerToken, leaseSeconds);
    if (direct !== null) return direct;
    throw rpcError;
  }
}

export async function releaseCollectorLease(
  client: RpcClient,
  collectorId: string,
  ownerToken: string,
): Promise<boolean> {
  try {
    const data = await rpcWithRetry(client, "brian_release_collector_lease", {
      p_collector_id: collectorId,
      p_owner_token: ownerToken,
    });
    return data === true;
  } catch (rpcError) {
    const direct = await directLeaseCall("release", collectorId, ownerToken);
    if (direct !== null) return direct;
    throw rpcError;
  }
}

export async function withCollectorLease<T>(
  client: RpcClient,
  collectorId: string,
  leaseSeconds: number,
  work: (ownerToken: string) => Promise<T>,
): Promise<LeaseResult<T>> {
  const ownerToken = randomOwnerToken();
  const acquired = await acquireCollectorLease(client, collectorId, ownerToken, leaseSeconds);
  if (!acquired) return { contended: true, ownerToken };

  const renewIntervalMs = Math.max(1, Math.floor(leaseSeconds / 3)) * 1000;
  const heartbeat = setInterval(() => {
    renewCollectorLease(client, collectorId, ownerToken, leaseSeconds)
      .then((renewed) => {
        if (!renewed) {
          console.error(
            `collector lease renewal lost ownership for ${collectorId} (owner ${ownerToken}); stopping renewal`,
          );
          clearInterval(heartbeat);
        }
      })
      .catch((renewError) => {
        console.error(`collector lease renewal failed for ${collectorId}: ${errorText(renewError)}`);
      });
  }, renewIntervalMs);

  try {
    const value = await work(ownerToken);
    return { contended: false, ownerToken, value };
  } finally {
    clearInterval(heartbeat);
    try {
      await releaseCollectorLease(client, collectorId, ownerToken);
    } catch (releaseError) {
      console.error(`collector lease release failed for ${collectorId}: ${errorText(releaseError)}`);
    }
  }
}
