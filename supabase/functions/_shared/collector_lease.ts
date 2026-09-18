// Collector-level atomic lease/mutex for Brian collectors.
//
// The database lease remains the source of truth. This client wrapper adds bounded retry and
// structured error normalization around the three idempotent lease RPCs so a transient
// PostgREST/schema-cache/connection wobble does not immediately take every collector down with
// an opaque "[object Object]" failure.

export interface RpcClient {
  rpc(fn: string, params: Record<string, unknown>): PromiseLike<{ data: unknown; error: unknown }>;
}

export interface LeaseResult<T> {
  contended: boolean;
  ownerToken: string;
  value?: T;
}

type RpcResult = { data: unknown; error: unknown };

const RPC_ATTEMPTS = 2;
const RPC_BACKOFF_MS = [250];
const RPC_TIMEOUT_MS = 5_000;

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
      const request: any = client.rpc(fn, params);
      result = typeof request?.abortSignal === "function"
        ? await request.abortSignal(AbortSignal.timeout(RPC_TIMEOUT_MS))
        : await Promise.race([
            Promise.resolve(request),
            new Promise<RpcResult>((_, reject) => setTimeout(() => reject(new Error(`${fn} timeout after ${RPC_TIMEOUT_MS}ms`)), RPC_TIMEOUT_MS)),
          ]);
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
  const data = await rpcWithRetry(client, "brian_acquire_collector_lease", {
    p_collector_id: collectorId,
    p_owner_token: ownerToken,
    p_lease_seconds: leaseSeconds,
  });
  return data === true;
}

export async function renewCollectorLease(
  client: RpcClient,
  collectorId: string,
  ownerToken: string,
  leaseSeconds: number,
): Promise<boolean> {
  const data = await rpcWithRetry(client, "brian_renew_collector_lease", {
    p_collector_id: collectorId,
    p_owner_token: ownerToken,
    p_lease_seconds: leaseSeconds,
  });
  return data === true;
}

export async function releaseCollectorLease(
  client: RpcClient,
  collectorId: string,
  ownerToken: string,
): Promise<boolean> {
  const data = await rpcWithRetry(client, "brian_release_collector_lease", {
    p_collector_id: collectorId,
    p_owner_token: ownerToken,
  });
  return data === true;
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
