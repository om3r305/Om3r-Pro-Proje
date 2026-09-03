// Collector-level atomic lease/mutex (Item 1, brian-2026 issue #32). Wraps the
// brian_acquire_collector_lease / brian_release_collector_lease Postgres RPCs (see
// supabase/migrations/202609030009_brian_collector_lease.sql) with the
// acquire -> work -> release-in-finally pattern every Phase 3.8-4.0 collector needs.
//
// A contended lease returns before any collector work begins -- callers must not fetch market
// data, write sensor observations, or write a second virtual-book chain in that case. See
// collector_lease.test.ts for the caller-side contract tests (mocked RPC client, no network/DB);
// the migration file plus tests/test_brian2026_collector_lease_postgres.py cover real Postgres
// atomicity under concurrency.
//
// RpcClient is deliberately a minimal structural interface (not the full supabase-js
// SupabaseClient type) so a plain mock object can stand in for tests without any network or
// Supabase dependency, while the real client -- whose `.rpc()` already returns a thenable --
// still satisfies it unchanged.

export interface RpcClient {
  rpc(fn: string, params: Record<string, unknown>): PromiseLike<{ data: unknown; error: unknown }>;
}

export interface LeaseResult<T> {
  contended: boolean;
  ownerToken: string;
  value?: T;
}

export function randomOwnerToken(): string {
  return crypto.randomUUID();
}

export async function acquireCollectorLease(
  client: RpcClient,
  collectorId: string,
  ownerToken: string,
  leaseSeconds: number,
): Promise<boolean> {
  const { data, error } = await client.rpc("brian_acquire_collector_lease", {
    p_collector_id: collectorId,
    p_owner_token: ownerToken,
    p_lease_seconds: leaseSeconds,
  });
  if (error) throw error;
  return data === true;
}

export async function releaseCollectorLease(
  client: RpcClient,
  collectorId: string,
  ownerToken: string,
): Promise<boolean> {
  const { data, error } = await client.rpc("brian_release_collector_lease", {
    p_collector_id: collectorId,
    p_owner_token: ownerToken,
  });
  if (error) throw error;
  return data === true;
}

/**
 * Acquires the named collector's mutex, runs `work` if acquired, and releases the lease in a
 * finally block regardless of whether `work` throws or resolves. If the lease is contended,
 * `work` is never invoked and `{ contended: true }` is returned immediately.
 *
 * If `work` throws, that error still propagates out of this function (after the release attempt
 * completes) so the caller's own try/catch and failure-accounting path still run -- a lease
 * acquisition does not swallow collector failures. A failure to release, on the other hand, is
 * logged and swallowed so it can never mask the real outcome of `work`.
 */
export async function withCollectorLease<T>(
  client: RpcClient,
  collectorId: string,
  leaseSeconds: number,
  work: (ownerToken: string) => Promise<T>,
): Promise<LeaseResult<T>> {
  const ownerToken = randomOwnerToken();
  const acquired = await acquireCollectorLease(client, collectorId, ownerToken, leaseSeconds);
  if (!acquired) return { contended: true, ownerToken };
  try {
    const value = await work(ownerToken);
    return { contended: false, ownerToken, value };
  } finally {
    try {
      await releaseCollectorLease(client, collectorId, ownerToken);
    } catch (releaseError) {
      console.error(`collector lease release failed for ${collectorId}`, releaseError);
    }
  }
}
